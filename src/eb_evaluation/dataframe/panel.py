"""
Panel-style evaluation output (DataFrame utilities).

This module provides a convenience wrapper that evaluates a DataFrame at multiple hierarchy
levels and returns a long-form (tidy) panel suitable for reporting, plotting, and downstream
aggregation.

The implementation delegates the core computation to
``eb_evaluation.dataframe.hierarchy.evaluate_hierarchy_df`` and then reshapes the wide
per-level outputs into a single stacked table with:

- a ``level`` column (which hierarchy level produced the row)
- optional grouping key columns (depending on the level)
- ``metric`` / ``value`` columns for tidy analysis
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .hierarchy import evaluate_hierarchy_df


def evaluate_panel_df(
    df: pd.DataFrame,
    levels: dict[str, Sequence[str]],
    actual_col: str,
    forecast_col: str,
    cu,
    co,
    tau: float | None = None,
) -> pd.DataFrame:
    """Evaluate metrics at multiple levels and return a long-form panel DataFrame.

    This is a convenience wrapper around
    ``eb_evaluation.dataframe.hierarchy.evaluate_hierarchy_df`` that:

    1. Computes a wide metrics DataFrame per hierarchy level.
    2. Stacks them into a single table with a ``level`` column.
    3. Melts metrics into ``metric`` / ``value`` pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least ``actual_col`` and ``forecast_col`` plus any
        grouping columns referenced in ``levels``.
    levels : dict[str, Sequence[str]]
        Mapping of level name to the column names used to group at that level.

        Example:

        >>> levels = {
        ...     "overall": [],
        ...     "by_store": ["store_id"],
        ...     "by_item": ["item_id"],
        ...     "by_store_item": ["store_id", "item_id"],
        ... }
    actual_col : str
        Column name for actual demand / realized values.
    forecast_col : str
        Column name for forecast values.
    cu
        Underbuild (shortfall) cost coefficient passed through to CWSL/FRS evaluations.
    co
        Overbuild (excess) cost coefficient passed through to CWSL/FRS evaluations.
    tau : float | None, default=None
        Tolerance parameter for HR@tau. If ``None``, HR@tau is omitted.

    Returns
    -------
    pandas.DataFrame
        Long-form (tidy) panel with columns:

        - ``level`` : hierarchy level name
        - ``<group cols>`` : the grouping keys for that level (may be empty for overall)
        - ``metric`` : metric name
        - ``value`` : metric value

        Each row corresponds to a single metric evaluated at a specific level/group.

    Notes
    -----
    - The set of metric columns is derived from the outputs of
      ``eb_evaluation.dataframe.hierarchy.evaluate_hierarchy_df``. Only metrics present in
      the combined wide table are melted.
    - Grouping key columns vary by level. The returned panel includes the union of all
      grouping key columns across levels; levels that do not use a given key will have NaN
      in that column.
    """
    hier = evaluate_hierarchy_df(
        df=df,
        levels=levels,
        actual_col=actual_col,
        forecast_col=forecast_col,
        cu=cu,
        co=co,
        tau=tau,
    )

    stacked_frames: list[pd.DataFrame] = []
    for level_name, level_df in hier.items():
        tmp = level_df.copy()
        tmp["level"] = level_name
        stacked_frames.append(tmp)

    combined = pd.concat(stacked_frames, ignore_index=True)

    # Put 'level' first for readability
    cols = ["level"] + [c for c in combined.columns if c != "level"]
    combined = combined[cols]

    # Decide which columns are metrics vs grouping keys
    candidate_metric_cols = [
        "n_intervals",
        "total_demand",
        "cwsl",
        "nsl",
        "ud",
        "wmape",
        "hr_at_tau",
        "frs",
    ]
    metric_cols = [c for c in candidate_metric_cols if c in combined.columns]

    # Everything else (besides 'level') is treated as a grouping key
    group_cols = [c for c in combined.columns if c not in metric_cols and c != "level"]

    panel = combined.melt(
        id_vars=["level", *group_cols],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    # Reorder for readability
    panel = panel[["level", *group_cols, "metric", "value"]]

    return panel

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

from .hierarchy import evaluate_hierarchy_df


def evaluate_panel_df(
    df: pd.DataFrame,
    levels: Dict[str, Sequence[str]],
    actual_col: str,
    forecast_col: str,
    cu,
    co,
    tau: float | None = None,
) -> pd.DataFrame:
    """
    Evaluate CWSL and related diagnostics at multiple levels and return
    a long-form (tidy) panel DataFrame.

    This is a convenience wrapper around ``evaluate_hierarchy_df`` that
    stacks all levels into a single table with a ``level`` column and
    metric/value pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the actual and forecast columns,
        plus any grouping columns referenced in ``levels``.

    levels : dict[str, Sequence[str]]
        Mapping of level name -> list/tuple of column names to group by.

        Examples
        --------
        levels = {
            "overall": [],
            "by_store": ["store_id"],
            "by_item": ["item_id"],
            "by_store_item": ["store_id", "item_id"],
        }

    actual_col : str
        Column name for actual demand.

    forecast_col : str
        Column name for forecasted demand.

    cu : float or array-like
        Underbuild cost parameter passed through to ``cwsl``.

    co : float or array-like
        Overbuild cost parameter passed through to ``cwsl``.

    tau : float, optional
        Tolerance passed to ``hr_at_tau``. If None, HR@τ is omitted.

    Returns
    -------
    pandas.DataFrame
        Long-form panel with columns like:

            level | <group cols> | metric | value

        where each row is a single metric at a specific level/group.
    """
    # First get wide DataFrames per level
    hier = evaluate_hierarchy_df(
        df=df,
        levels=levels,
        actual_col=actual_col,
        forecast_col=forecast_col,
        cu=cu,
        co=co,
        tau=tau,
    )

    # Stack them with a 'level' column
    stacked_frames: list[pd.DataFrame] = []
    for level_name, level_df in hier.items():
        tmp = level_df.copy()
        tmp["level"] = level_name
        stacked_frames.append(tmp)

    combined = pd.concat(stacked_frames, ignore_index=True)

    # Put 'level' first
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

    # Melt to long form: one row per level/group/metric
    panel = combined.melt(
        id_vars=["level"] + group_cols,
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    # Reorder for readability
    panel = panel[["level"] + group_cols + ["metric", "value"]]

    return panel
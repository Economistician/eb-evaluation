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

from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fpc import FPCThresholds
from eb_evaluation.diagnostics.presets import GovernancePreset
from eb_evaluation.diagnostics.run import run_governance_gate

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

    # Put 'level' first for readability.
    # Use `loc` to keep the type a DataFrame for type checkers.
    cols = ["level"] + [c for c in combined.columns if c != "level"]
    combined = combined.loc[:, cols]

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

    # Reorder for readability.
    # Use `loc` so the return is always a DataFrame (not inferred as Series).
    panel = panel.loc[:, ["level", *group_cols, "metric", "value"]]

    return panel


def run_governance_panel_df(
    *,
    df: pd.DataFrame,
    group_cols: Sequence[str],
    actual_col: str,
    forecast_base_col: str,
    forecast_ral_col: str,
    tau: float,
    cwsl_r: float | None = None,
    dqc_thresholds: DQCThresholds | None = None,
    fpc_thresholds: FPCThresholds | None = None,
    preset: GovernancePreset | str | None = None,
) -> pd.DataFrame:
    """
    Run the governance gate per panel stream and return a tidy results DataFrame.

    This function is the governance analogue of evaluate_panel_df: it operates
    on a panel-like dataframe and emits one row per stream (group).

    Parameters
    ----------
    df:
        Input frame containing actuals and forecasts.
    group_cols:
        Columns defining a stream identity (e.g., ["site_id", "forecast_entity_id"]).
    actual_col:
        Actual demand/usage column.
    forecast_base_col:
        Baseline forecast column.
    forecast_ral_col:
        RAL-adjusted forecast column (can be identical to baseline if no RAL is applied).
    tau:
        Tolerance parameter passed to governance gating. Note governance may direct
        downstream interpretation as raw vs grid units.
    cwsl_r:
        Optional asymmetry ratio for CWSL-sensitive gating.
    dqc_thresholds, fpc_thresholds, preset:
        Same semantics as diagnostics.run.run_governance_gate: do not mix preset with
        explicit thresholds.

    Returns
    -------
    pandas.DataFrame
        One row per stream with governance artifacts and routing recommendations.
    """
    required = set(group_cols) | {actual_col, forecast_base_col, forecast_ral_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for governance panel: {missing}")

    results: list[dict[str, object]] = []

    # Pandas groupby wants list[str] for stable typing.
    group_cols_list = list(group_cols)

    for keys, g in df.groupby(group_cols_list, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys_tuple: tuple[object, ...] = (keys,)
        else:
            keys_tuple = keys

        row: dict[str, object] = dict(zip(group_cols_list, keys_tuple, strict=True))

        # Align series by dropping any row where any input is null.
        sub = g.loc[:, [actual_col, forecast_base_col, forecast_ral_col]].dropna(how="any")
        n_used = len(sub)
        row["n_points_used"] = n_used

        if n_used == 0:
            row["warnings"] = "empty_series_after_dropna"
            results.append(row)
            continue

        y_list = sub[actual_col].astype(float).tolist()
        yhat_base_list = sub[forecast_base_col].astype(float).tolist()
        yhat_ral_list = sub[forecast_ral_col].astype(float).tolist()

        gate = run_governance_gate(
            y=y_list,
            yhat_base=yhat_base_list,
            yhat_ral=yhat_ral_list,
            tau=float(tau),
            cwsl_r=cwsl_r,
            dqc_thresholds=dqc_thresholds,
            fpc_thresholds=fpc_thresholds,
            preset=preset,
        )

        row["dqc_class"] = gate.dqc.dqc_class.value
        row["dqc_granularity"] = gate.dqc.signals.granularity
        row["fpc_raw_class"] = gate.fpc_raw.fpc_class.value
        row["fpc_snapped_class"] = gate.fpc_snapped.fpc_class.value

        # GovernanceDecision fields are stable outputs; expose them directly.
        row["snap_required"] = gate.decision.snap_required
        row["snap_unit"] = gate.decision.snap_unit
        row["tau_policy"] = gate.decision.tau_policy.value
        row["ral_policy"] = gate.decision.ral_policy.value
        row["status"] = gate.decision.status.value

        row["recommended_mode"] = gate.recommended_mode
        row["recommendations"] = ", ".join(gate.recommendations)

        results.append(row)

    return pd.DataFrame(results)

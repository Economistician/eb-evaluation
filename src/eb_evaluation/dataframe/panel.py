"""Long-form panel evaluation across hierarchy levels.

Runs ``evaluate_hierarchy_df`` and melts wide per-level metrics into
``level`` / ``metric`` / ``value`` rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from eb_evaluation.diagnostics.dqc import DQCClass, DQCThresholds
from eb_evaluation.diagnostics.fas import FASClass, resolve_panel_fas_class
from eb_evaluation.diagnostics.fpc import FPCClass, FPCThresholds
from eb_evaluation.diagnostics.governance import GovernanceStatus, RALPolicy, TauPolicy
from eb_evaluation.diagnostics.presets import GovernancePreset
from eb_evaluation.diagnostics.run import run_governance_gate

from .governance_panel import fas_review_is_missing, finite_coverage_is_insufficient
from .hierarchy import evaluate_hierarchy_df


def evaluate_panel_df(
    df: pd.DataFrame,
    levels: dict[str, Sequence[str]],
    actual_col: str,
    forecast_col: str,
    cu,
    co,
    tau: float | None = None,
    *,
    cwsl_max: float,
    sample_weight_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate metrics at multiple levels and return a long-form panel DataFrame.

    Runs ``evaluate_hierarchy_df``, then melts to ``level`` / ``metric`` / ``value``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input with ``actual_col``, ``forecast_col``, and grouping columns from ``levels``.
    levels : dict[str, Sequence[str]]
        Mapping of level name to grouping columns.

        Example:

        >>> levels = {
        ...     "overall": [],
        ...     "by_store": ["store_id"],
        ...     "by_item": ["item_id"],
        ...     "by_store_item": ["store_id", "item_id"],
        ... }
    actual_col : str
        Column of actual demand / realized values.
    forecast_col : str
        Column of forecast values.
    cu
        Underbuild cost passed through to CWSL/FRS evaluations.
    co
        Overbuild cost passed through to CWSL/FRS evaluations.
    tau : float | None, default=None
        Tolerance for HR@tau. If ``None``, HR@tau is omitted.
    cwsl_max : float
        Finite upper bound for FRS scaling; must be ``> 0``.
    sample_weight_col : str | None, default=None
        Optional non-negative weights, passed through to ``evaluate_hierarchy_df``.

    Returns
    -------
    pandas.DataFrame
        Long-form panel with columns:

        - ``level`` : hierarchy level name
        - ``<group cols>`` : grouping keys for that level (may be empty for overall)
        - ``metric`` : metric name
        - ``value`` : metric value

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
        cwsl_max=cwsl_max,
        sample_weight_col=sample_weight_col,
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
    fas_class: FASClass | str | pd.Series | None = None,
    fas_class_col: str | None = None,
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
    fas_class:
        Required upstream Forecast Admissibility Surface class. A scalar is
        broadcast to every stream. A row-aligned Series is resolved per stream.
        Omitted or null FAS fail-closes the stream (``fas_class=BLOCKED``,
        ``status=red``, ``ral_policy=disallow``).
    fas_class_col:
        Optional panel column of per-row FAS classes. Mixed values within a
        stream raise. When set, this column takes precedence over ``fas_class``.
        A column that resolves to all-null for a stream is treated as omitted.

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

        # Align series by dropping any row where any input is null or non-finite.
        stream_cols = [actual_col, forecast_base_col, forecast_ral_col]
        sub = g.loc[:, stream_cols].copy()
        for col in stream_cols:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        finite_mask = np.isfinite(sub.to_numpy(dtype=float)).all(axis=1)
        sub = sub.loc[finite_mask]
        n_used = len(sub)
        row["n_points_used"] = n_used
        row["n_finite"] = n_used
        row["finite_coverage"] = (float(n_used) / float(len(g))) if len(g) else 0.0

        stream_fas = resolve_panel_fas_class(g, fas_class=fas_class, fas_class_col=fas_class_col)
        if fas_review_is_missing(stream_fas):
            row["warnings"] = "fas_required_fail_closed"
            row["dqc_class"] = DQCClass.UNKNOWN.value
            row["dqc_granularity"] = None
            row["fpc_raw_class"] = FPCClass.INCOMPATIBLE.value
            row["fpc_snapped_class"] = FPCClass.INCOMPATIBLE.value
            row["snap_required"] = False
            row["snap_unit"] = None
            row["tau_policy"] = TauPolicy.RAW_UNITS.value
            row["ral_policy"] = RALPolicy.DISALLOW.value
            row["status"] = GovernanceStatus.RED.value
            row["fas_class"] = FASClass.BLOCKED.value
            row["recommended_mode"] = "reroute_discrete"
            row["recommendations"] = "fas_required_fail_closed"
            results.append(row)
            continue

        if finite_coverage_is_insufficient(len(g), n_used):
            row["warnings"] = (
                "empty_series_after_dropna"
                if n_used == 0
                else "insufficient_finite_coverage_fail_closed"
            )
            row["dqc_class"] = DQCClass.UNKNOWN.value
            row["dqc_granularity"] = None
            row["fpc_raw_class"] = FPCClass.INCOMPATIBLE.value
            row["fpc_snapped_class"] = FPCClass.INCOMPATIBLE.value
            row["snap_required"] = False
            row["snap_unit"] = None
            row["tau_policy"] = TauPolicy.RAW_UNITS.value
            row["ral_policy"] = RALPolicy.DISALLOW.value
            row["status"] = GovernanceStatus.RED.value
            if isinstance(stream_fas, FASClass):
                row["fas_class"] = stream_fas.value
            else:
                row["fas_class"] = str(stream_fas)
            row["recommended_mode"] = "reroute_discrete"
            row["recommendations"] = (
                "empty_series_fail_closed"
                if n_used == 0
                else "insufficient_finite_coverage_fail_closed"
            )
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
            fas_class=stream_fas,
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
        row["fas_class"] = (
            None if gate.decision.fas_class is None else gate.decision.fas_class.value
        )

        row["recommended_mode"] = gate.recommended_mode
        row["recommendations"] = ", ".join(gate.recommendations)

        results.append(row)

    return pd.DataFrame(results)

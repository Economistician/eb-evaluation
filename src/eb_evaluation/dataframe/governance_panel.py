"""Per-stream governance gate over a panel DataFrame.

Groups by caller keys, runs ``run_governance_gate`` on each stream, and returns
one summary row per stream (status, DQC/FPC classes, recommended mode, reasons).
Expects base and RAL forecast columns; reuse one column name when only a single
series exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from eb_evaluation.diagnostics.dqc import DQCClass, DQCThresholds
from eb_evaluation.diagnostics.fas import FASClass, resolve_panel_fas_class
from eb_evaluation.diagnostics.fpc import FPCClass, FPCThresholds
from eb_evaluation.diagnostics.governance import GovernanceStatus, RALPolicy, TauPolicy
from eb_evaluation.diagnostics.presets import GovernancePreset
from eb_evaluation.diagnostics.results import GovernanceResult
from eb_evaluation.diagnostics.run import run_governance_gate

__all__ = ["evaluate_governance_panel_df"]

_RECOMMENDATION_SEP = ", "


def _as_reason_string(reasons: Sequence[str] | None) -> str:
    if not reasons:
        return ""
    return "|".join(str(r) for r in reasons)


def _as_recommendation_string(reasons: Sequence[str] | None) -> str:
    if not reasons:
        return ""
    return _RECOMMENDATION_SEP.join(str(r) for r in reasons)


def _safe_getattr(obj: object, name: str) -> Any:
    """
    Best-effort attribute access.

    We use this to keep the panel summary stable even if diagnostic result
    objects evolve (e.g., adding/removing optional fields).
    """
    return getattr(obj, name, None)


_FAIL_CLOSED_TOKEN = "empty_series_fail_closed"


def _finite_aligned_subset(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Keep rows where every named column is numeric and finite."""
    work = frame.loc[:, list(cols)].copy()
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    values = work.to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    return work.loc[mask]


def _fas_value(stream_fas: FASClass | str | None) -> str | None:
    if stream_fas is None:
        return None
    if isinstance(stream_fas, FASClass):
        return stream_fas.value
    return str(stream_fas)


def _fail_closed_panel_row(
    *,
    keys_list: Sequence[str],
    key_vals: tuple[object, ...],
    n: int,
    stream_fas: FASClass | str | None,
) -> dict[str, Any]:
    """Emit a RED / DISALLOW / FPC INCOMPATIBLE row without calling eb-metrics."""
    row: dict[str, Any] = dict(zip(keys_list, key_vals, strict=True))
    row["n"] = n
    row["recommended_mode"] = "reroute_discrete"
    row["snap_required"] = False
    row["snap_unit"] = None
    row["tau_policy"] = TauPolicy.RAW_UNITS.value
    row["ral_policy"] = RALPolicy.DISALLOW.value
    row["status"] = GovernanceStatus.RED.value
    row["fas_class"] = _fas_value(stream_fas)
    row["dqc_class"] = DQCClass.UNKNOWN.value
    row["fpc_raw_class"] = FPCClass.INCOMPATIBLE.value
    row["fpc_snapped_class"] = FPCClass.INCOMPATIBLE.value
    row["dqc_granularity"] = None
    row["dqc_multiple_rate"] = None
    row["dqc_offgrid_mad_ratio"] = None
    row["dqc_nonzero_obs"] = None
    row["nsl_base_raw"] = None
    row["nsl_ral_raw"] = None
    row["delta_nsl_raw"] = None
    row["ud_raw"] = None
    row["nsl_base_snapped"] = None
    row["nsl_ral_snapped"] = None
    row["delta_nsl_snapped"] = None
    row["ud_snapped"] = None
    row["dqc_reasons"] = _FAIL_CLOSED_TOKEN
    row["fpc_raw_reasons"] = _FAIL_CLOSED_TOKEN
    row["fpc_snapped_reasons"] = _FAIL_CLOSED_TOKEN
    row["recommendations"] = _FAIL_CLOSED_TOKEN
    return row


def evaluate_governance_panel_df(
    *,
    df: pd.DataFrame,
    keys: Sequence[str],
    actual_col: str,
    base_forecast_col: str,
    ral_forecast_col: str,
    tau: float,
    cwsl_r: float | None = None,
    preset: GovernancePreset | str | None = None,
    dqc_thresholds: DQCThresholds | None = None,
    fpc_thresholds: FPCThresholds | None = None,
    dropna_keys: bool = True,
    fas_class: FASClass | str | pd.Series | None = None,
    fas_class_col: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate governance across a panel of streams and return a per-stream summary table.

    Parameters
    ----------
    df:
        Input panel DataFrame containing at least `keys`, `actual_col`,
        `base_forecast_col`, and `ral_forecast_col`.
    keys:
        Column names defining independent streams (e.g., ["site_id", "forecast_entity_id"]).
    actual_col:
        Column name containing realized demand / usage (y).
    base_forecast_col:
        Column name containing baseline predictions (yhat_base).
    ral_forecast_col:
        Column name containing readiness-adjusted predictions (yhat_ral).
    tau:
        HR@tau tolerance band in raw units. Governance may recommend interpreting
        tau in grid units when snapping is required.
    cwsl_r:
        Optional cost ratio r for CWSL response signals. When provided, governance
        will include CWSL response in FPC signals.
    preset:
        Optional governance preset (name or object). If provided, explicit thresholds
        must not be provided.
    dqc_thresholds, fpc_thresholds:
        Optional explicit threshold objects. If `preset` is provided, these must be None.
    dropna_keys:
        If True, drop rows with NA in any key column before grouping. This is usually
        desired for stable grouping semantics.
    fas_class:
        Optional upstream Forecast Admissibility Surface class. A scalar is
        broadcast to every stream. A row-aligned Series is resolved per stream.
    fas_class_col:
        Optional panel column of per-row FAS classes. Mixed values within a
        stream raise. When set, this column takes precedence over ``fas_class``.

    Returns
    -------
    pandas.DataFrame
        One row per stream (key combination) summarizing governance results.

    Notes
    -----
    Empty streams and streams with no finite aligned ``y`` / ``yhat`` points
    (NaN or ±inf) fail closed (``status=red``, ``ral_policy=disallow``, FPC
    ``incompatible``) instead of calling ``eb-metrics``.
    """
    keys_list = list(keys)
    if len(keys_list) == 0:
        raise ValueError("`keys` must contain at least one grouping column.")

    if preset is not None and (dqc_thresholds is not None or fpc_thresholds is not None):
        raise ValueError("Provide either `preset` or explicit thresholds, not both.")

    required = set(keys_list) | {actual_col, base_forecast_col, ral_forecast_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df
    if dropna_keys:
        work = work.dropna(subset=keys_list)

    out_rows: list[dict[str, Any]] = []

    grouped = work.groupby(keys_list, dropna=not dropna_keys, sort=False)

    for key_vals, g in grouped:
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)

        stream_fas = resolve_panel_fas_class(g, fas_class=fas_class, fas_class_col=fas_class_col)
        finite = _finite_aligned_subset(g, (actual_col, base_forecast_col, ral_forecast_col))
        if len(finite) == 0:
            out_rows.append(
                _fail_closed_panel_row(
                    keys_list=keys_list,
                    key_vals=key_vals,
                    n=len(g),
                    stream_fas=stream_fas,
                )
            )
            continue

        row: dict[str, Any] = dict(zip(keys_list, key_vals, strict=True))
        row["n"] = len(g)

        y = finite[actual_col].to_numpy(dtype=float)
        yhat_base = finite[base_forecast_col].to_numpy(dtype=float)
        yhat_ral = finite[ral_forecast_col].to_numpy(dtype=float)

        gate = run_governance_gate(
            y=y,
            yhat_base=yhat_base,
            yhat_ral=yhat_ral,
            tau=float(tau),
            cwsl_r=cwsl_r,
            preset=preset,
            dqc_thresholds=dqc_thresholds,
            fpc_thresholds=fpc_thresholds,
            fas_class=stream_fas,
        )

        # Use the portable, stable representation for core policy/class fields.
        result = GovernanceResult.from_gate_result(gate=gate)

        # Primary routing + policies
        row["recommended_mode"] = result.recommended_mode
        row["snap_required"] = bool(result.snap_required)
        row["snap_unit"] = result.snap_unit
        row["tau_policy"] = result.tau_policy.value
        row["ral_policy"] = result.ral_policy.value
        row["status"] = result.status.value
        row["fas_class"] = (
            None if gate.decision.fas_class is None else gate.decision.fas_class.value
        )

        # Classes
        row["dqc_class"] = result.dqc_class.value
        row["fpc_raw_class"] = result.fpc_raw_class.value
        row["fpc_snapped_class"] = result.fpc_snapped_class.value

        # Compact DQC signals (best effort)
        dqc_sig = gate.dqc.signals
        row["dqc_granularity"] = _safe_getattr(dqc_sig, "granularity")
        row["dqc_multiple_rate"] = _safe_getattr(dqc_sig, "multiple_rate")
        row["dqc_offgrid_mad_ratio"] = _safe_getattr(dqc_sig, "offgrid_mad_ratio")
        row["dqc_nonzero_obs"] = _safe_getattr(dqc_sig, "nonzero_obs")

        # Compact FPC signals (raw + snapped) (best effort)
        raw_sig = gate.fpc_raw.signals
        snap_sig = gate.fpc_snapped.signals

        row["nsl_base_raw"] = _safe_getattr(raw_sig, "nsl_base")
        row["nsl_ral_raw"] = _safe_getattr(raw_sig, "nsl_ral")
        row["delta_nsl_raw"] = _safe_getattr(raw_sig, "delta_nsl")
        row["ud_raw"] = _safe_getattr(raw_sig, "ud")

        row["nsl_base_snapped"] = _safe_getattr(snap_sig, "nsl_base")
        row["nsl_ral_snapped"] = _safe_getattr(snap_sig, "nsl_ral")
        row["delta_nsl_snapped"] = _safe_getattr(snap_sig, "delta_nsl")
        row["ud_snapped"] = _safe_getattr(snap_sig, "ud")

        # Reasons / recommendations (keep existing string columns for backward compatibility)
        row["dqc_reasons"] = _as_reason_string(result.dqc_reasons)
        row["fpc_raw_reasons"] = _as_reason_string(result.fpc_raw_reasons)
        row["fpc_snapped_reasons"] = _as_reason_string(result.fpc_snapped_reasons)
        row["recommendations"] = _as_recommendation_string(result.recommendations)

        out_rows.append(row)

    return pd.DataFrame(out_rows)

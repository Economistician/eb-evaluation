"""
Governance-oriented panel evaluation (DataFrame utilities).

This module provides a DataFrame-oriented orchestration layer for running the
governance gate (DQC x FPC) across a panel of independent demand streams.

Motivation
----------
In most operational forecasting settings, evaluation is not performed on a single
series. Instead, we evaluate many independent streams, typically defined by a key
set such as:

- site_id x forecast_entity_id
- store_id x item_id
- any other caller-defined grouping grain

The governance layer (DQC/FPC) is fundamentally *per-stream*: it needs aligned
(y, yhat_base, yhat_ral) series for a single stream to classify:

- whether snapping is required (DQC)
- whether readiness adjustment is structurally valid (FPC)
- how τ should be interpreted (TauPolicy)
- whether to allow RAL (RALPolicy)
- an overall governance status (green/yellow/red)
- a recommended evaluation routing mode

This module:
- groups a DataFrame into per-stream series,
- runs `eb_evaluation.diagnostics.run.run_governance_gate` per stream,
- returns a tidy (long-form) summary DataFrame suitable for reporting and joins.

Design goals
------------
- Keep diagnostics pure: this module orchestrates; it does not implement DQC/FPC math.
- Keep the output auditable: include key identifiers + decision fields + a compact reasons string.
- Be robust to real-world inputs: tolerate numpy/pandas-backed sequences, missing rows,
  and NaN-demand intervals by letting diagnostics handle NaN semantics.
- Avoid policy duplication: callers can pass presets/thresholds through unchanged.

Notes
-----
- This module expects `yhat_base` and `yhat_ral` columns to be present for each row.
  If you only have a single forecast series, you can pass the same column name for
  both base and ral, or build a second series upstream.
- The governance gate is run per stream. This means group sizes matter: extremely
  small streams may produce "continuous-like" or "empty_series" reasons depending
  on your thresholds.

Output schema
-------------
The returned DataFrame is one row per stream, with:

- keys (as provided by caller)
- n (group length)
- recommended_mode
- snap_required, snap_unit, tau_policy, ral_policy, status
- dqc_class, fpc_raw_class, fpc_snapped_class
- selected compact signal values when available (e.g., granularity, multiple_rate, nsl_base, nsl_ral, delta_nsl, ud)
- reasons (pipe-delimited string)

This is intended as a *summary* table. If you need full diagnostic artifacts per stream,
persist them separately (future "artifacts" layer).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fpc import FPCThresholds
from eb_evaluation.diagnostics.presets import GovernancePreset
from eb_evaluation.diagnostics.run import run_governance_gate


def _as_reason_string(reasons: Sequence[str] | None) -> str:
    if not reasons:
        return ""
    return "|".join(str(r) for r in reasons)


def _safe_getattr(obj: object, name: str) -> Any:
    """
    Best-effort attribute access.

    We use this to keep the panel summary stable even if diagnostic result
    objects evolve (e.g., adding/removing optional fields).
    """
    return getattr(obj, name, None)


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

    Returns
    -------
    pandas.DataFrame
        One row per stream (key combination) summarizing governance results.
    """
    if len(keys) == 0:
        raise ValueError("`keys` must contain at least one grouping column.")

    required = set(keys) | {actual_col, base_forecast_col, ral_forecast_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df
    if dropna_keys:
        work = work.dropna(subset=list(keys))

    out_rows: list[dict[str, Any]] = []

    grouped = work.groupby(list(keys), dropna=not dropna_keys, sort=False)

    for key_vals, g in grouped:
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)

        row: dict[str, Any] = dict(zip(keys, key_vals, strict=True))
        row["n"] = len(g)

        y = g[actual_col].to_numpy(dtype=float)
        yhat_base = g[base_forecast_col].to_numpy(dtype=float)
        yhat_ral = g[ral_forecast_col].to_numpy(dtype=float)

        gate = run_governance_gate(
            y=y,
            yhat_base=yhat_base,
            yhat_ral=yhat_ral,
            tau=tau,
            cwsl_r=cwsl_r,
            preset=preset,
            dqc_thresholds=dqc_thresholds,
            fpc_thresholds=fpc_thresholds,
        )

        # Primary routing + policies
        row["recommended_mode"] = gate.recommended_mode
        row["snap_required"] = gate.decision.snap_required
        row["snap_unit"] = gate.decision.snap_unit
        row["tau_policy"] = gate.decision.tau_policy.value
        row["ral_policy"] = gate.decision.ral_policy.value
        row["status"] = gate.decision.status.value

        # Classes
        row["dqc_class"] = gate.dqc.dqc_class.value
        row["fpc_raw_class"] = gate.fpc_raw.fpc_class.value
        row["fpc_snapped_class"] = gate.fpc_snapped.fpc_class.value

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

        # Reasons / recommendations
        row["dqc_reasons"] = _as_reason_string(gate.dqc.reasons)
        row["fpc_raw_reasons"] = _as_reason_string(gate.fpc_raw.reasons)
        row["fpc_snapped_reasons"] = _as_reason_string(gate.fpc_snapped.reasons)
        row["recommendations"] = _as_reason_string(gate.recommendations)

        out_rows.append(row)

    return pd.DataFrame(out_rows)

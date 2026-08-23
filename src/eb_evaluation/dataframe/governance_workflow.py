"""Compose governance decisions with RAL panel application.

Runs ``evaluate_governance_panel_df`` then ``apply_ral``: decisions table in,
governed panel (and decisions) out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from eb_evaluation.adjustment.ral import (
    REQUIRED_DECISION_COLUMNS,
    NonnegPolicy,
    SnapMode,
    apply_ral,
)
from eb_evaluation.dataframe.governance_panel import evaluate_governance_panel_df
from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fas import FASClass
from eb_evaluation.diagnostics.fpc import FPCThresholds
from eb_evaluation.diagnostics.presets import GovernancePreset

__all__ = ["run_governance_workflow_df", "run_governance_workflow_df_dict"]

_FAIL_CLOSED_RAL = "disallow"
_FAIL_CLOSED_STATUS = "red"
_FAIL_CLOSED_DQC = "unknown"
_FAIL_CLOSED_FAS = "BLOCKED"
_SNAP_DQC_CLASSES = ("quantized", "piecewise_packed")
_STATUS_RANK = {"green": 0, "yellow": 1, "red": 2}
_RAL_POLICY_RANK = {
    "allow": 0,
    "allow_after_snap": 1,
    "caution_after_snap": 2,
    "disallow": 3,
}


def _ctrl_token(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (ValueError, TypeError):
        pass
    raw = getattr(value, "value", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return str(value).strip().lower().rsplit(".", 1)[-1]


def _ral_policy_rank(token: str) -> int:
    """Total order: disallow > caution_* > allow_after_snap > allow. Unknown is disallow."""
    key = str(token).strip().lower()
    if key in _RAL_POLICY_RANK:
        return _RAL_POLICY_RANK[key]
    if key.startswith("caution"):
        return _RAL_POLICY_RANK["caution_after_snap"]
    if key.startswith("allow"):
        return (
            _RAL_POLICY_RANK["allow_after_snap"]
            if "after_snap" in key
            else _RAL_POLICY_RANK["allow"]
        )
    return _RAL_POLICY_RANK["disallow"]


def _as_series(frame: pd.DataFrame, col: str) -> pd.Series:
    obj = frame[col]
    if isinstance(obj, pd.Series):
        return obj
    return pd.Series(obj, index=frame.index, name=col)


def _fail_close_incomplete_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Fill missing governance control columns as DISALLOW / RED / UNKNOWN.

    An injected override that omits any required control column cannot be
    treated as a complete gate result. Missing columns, and rows with NA in
    ral_policy / status / fas_class / dqc_class / snap_required, fail closed.
    """
    out = decisions.copy()
    missing_cols = [c for c in REQUIRED_DECISION_COLUMNS if c not in out.columns]
    if missing_cols:
        out["ral_policy"] = _FAIL_CLOSED_RAL
        out["status"] = _FAIL_CLOSED_STATUS
        if "fas_class" not in out.columns:
            out["fas_class"] = _FAIL_CLOSED_FAS
        if "dqc_class" not in out.columns:
            out["dqc_class"] = _FAIL_CLOSED_DQC
        if "snap_required" not in out.columns:
            out["snap_required"] = False

    ral_obj = out["ral_policy"]
    status_obj = out["status"]
    dqc_obj = out["dqc_class"]
    snap_obj = out["snap_required"]
    fas_obj = out["fas_class"]
    ral_s = ral_obj if isinstance(ral_obj, pd.Series) else pd.Series(ral_obj, index=out.index)
    status_s = (
        status_obj if isinstance(status_obj, pd.Series) else pd.Series(status_obj, index=out.index)
    )
    dqc_s = dqc_obj if isinstance(dqc_obj, pd.Series) else pd.Series(dqc_obj, index=out.index)
    snap_s = snap_obj if isinstance(snap_obj, pd.Series) else pd.Series(snap_obj, index=out.index)
    fas_s = fas_obj if isinstance(fas_obj, pd.Series) else pd.Series(fas_obj, index=out.index)
    incomplete = ral_s.isna() | status_s.isna() | dqc_s.isna() | snap_s.isna() | fas_s.isna()
    if bool(incomplete.any()):
        out.loc[incomplete, "ral_policy"] = _FAIL_CLOSED_RAL
        out.loc[incomplete, "status"] = _FAIL_CLOSED_STATUS
        out.loc[incomplete, "dqc_class"] = _FAIL_CLOSED_DQC
        out.loc[incomplete, "snap_required"] = False
        out.loc[incomplete, "fas_class"] = _FAIL_CLOSED_FAS
    return out


def _reconcile_injected_with_gate(
    injected: pd.DataFrame,
    gate: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Keep injected tightenings; never loosen snap, DQC, RAL, or status vs the gate."""
    key_list = list(keys)
    merged = injected.merge(gate, on=key_list, how="left", suffixes=("", "_gate"))

    if "dqc_class_gate" in merged.columns:
        gate_dqc = _as_series(merged, "dqc_class_gate").map(_ctrl_token)
        use_gate_dqc = gate_dqc.isin((*_SNAP_DQC_CLASSES, "unknown"))
        inj_dqc = _as_series(merged, "dqc_class")
        merged["dqc_class"] = inj_dqc.where(~use_gate_dqc, _as_series(merged, "dqc_class_gate"))
    else:
        gate_dqc = pd.Series("", index=merged.index)
        use_gate_dqc = pd.Series(False, index=merged.index)

    inj_snap = _as_series(merged, "snap_required").fillna(False).astype(bool)
    if "snap_required_gate" in merged.columns:
        gate_snap = _as_series(merged, "snap_required_gate").fillna(False).astype(bool)
    else:
        gate_snap = pd.Series(False, index=merged.index)
    dqc_implies_snap = gate_dqc.isin(_SNAP_DQC_CLASSES)
    merged["snap_required"] = inj_snap | gate_snap | dqc_implies_snap

    if "snap_unit_gate" in merged.columns:
        need_unit = merged["snap_required"].astype(bool)
        inj_unit = (
            _as_series(merged, "snap_unit")
            if "snap_unit" in merged.columns
            else pd.Series(np.nan, index=merged.index)
        )
        merged["snap_unit"] = inj_unit.where(~need_unit, _as_series(merged, "snap_unit_gate"))

    if "ral_policy_gate" in merged.columns:
        gate_ral = _as_series(merged, "ral_policy_gate").map(_ctrl_token)
        inj_ral = _as_series(merged, "ral_policy").map(_ctrl_token)
        gate_rr = gate_ral.map(_ral_policy_rank)
        inj_rr = inj_ral.map(_ral_policy_rank)
        use_inj_ral = inj_rr > gate_rr
        chosen_ral = inj_ral.where(use_inj_ral, gate_ral)
        empty_ral = chosen_ral.map(_ctrl_token) == ""
        merged["ral_policy"] = chosen_ral.where(~empty_ral, _FAIL_CLOSED_RAL)

    if "status_gate" in merged.columns:
        gate_status = _as_series(merged, "status_gate").map(_ctrl_token)
        inj_status = _as_series(merged, "status").map(_ctrl_token)
        gate_rank = gate_status.map(lambda t: _STATUS_RANK.get(t, 2))
        inj_rank = inj_status.map(lambda t: _STATUS_RANK.get(t, 2))
        use_gate_status = gate_rank >= inj_rank
        merged["status"] = _as_series(merged, "status").where(
            ~use_gate_status, _as_series(merged, "status_gate")
        )

    chosen_ral_token = _as_series(merged, "ral_policy").map(_ctrl_token)
    merged.loc[chosen_ral_token == "disallow", "status"] = _FAIL_CLOSED_STATUS
    caution_mask = chosen_ral_token.str.startswith("caution")
    status_token = _as_series(merged, "status").map(_ctrl_token)
    merged.loc[caution_mask & (status_token == "green"), "status"] = "yellow"

    if "fas_class_gate" in merged.columns:
        gate_fas = _as_series(merged, "fas_class_gate").map(_ctrl_token)
        inj_fas = _as_series(merged, "fas_class").map(_ctrl_token)
        blocked = (gate_fas == "blocked") | (inj_fas == "blocked")
        if bool(blocked.any()):
            merged.loc[blocked, "fas_class"] = "BLOCKED"
            merged.loc[blocked, "ral_policy"] = _FAIL_CLOSED_RAL
            merged.loc[blocked, "status"] = _FAIL_CLOSED_STATUS

    if "recommended_mode_gate" in merged.columns and (
        bool(use_gate_dqc.any()) or bool((gate_snap | dqc_implies_snap).any())
    ):
        merged["recommended_mode"] = _as_series(merged, "recommended_mode_gate")
    if "recommendations_gate" in merged.columns and (
        bool(use_gate_dqc.any()) or bool((gate_snap | dqc_implies_snap).any())
    ):
        merged["recommendations"] = _as_series(merged, "recommendations_gate")

    drop_gate = [c for c in merged.columns if c.endswith("_gate")]
    if drop_gate:
        merged = merged.drop(columns=drop_gate)
    return merged


def run_governance_workflow_df(
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
    # decisions override (optional)
    decisions_df: pd.DataFrame | None = None,
    # apply step
    snap_mode: SnapMode = "ceil",
    nonneg_policy: NonnegPolicy | None = None,
    infer_policy_from_recommendations: bool = True,
    out_base_col: str | None = None,
    out_ral_col: str | None = None,
    out_audit_prefix: str = "ral_apply_",
    require_complete_decisions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end governance workflow for a panel DataFrame.

    Returns
    -------
    (panel_governed_df, decisions_df)

    Notes
    -----
    - `decisions_df` is a per-stream summary table and is directly joinable back to `df`
      on `keys`.
    - `panel_governed_df` includes governed prediction columns plus audit columns.
    - The governance gate always runs. A custom ``decisions_df`` is schema-fail-closed
      (missing or NA ``ral_policy`` / ``status`` / ``fas_class`` / ``dqc_class`` /
      ``snap_required`` become ``DISALLOW`` / ``RED`` / DQC ``UNKNOWN``), then
      reconciled against the gate: injected rows may tighten, never loosen snap,
      DQC class, RAL policy, or status. ``ral_policy`` uses the total order
      ``disallow > caution_* > allow_*``; injected ``allow`` cannot upgrade
      gate ``caution_after_snap``.
    - Empty, missing, non-finite, or thinly covered ``y`` / ``yhat`` streams
      fail closed in ``evaluate_governance_panel_df`` (``status=red``,
      ``ral_policy=disallow``, FPC ``incompatible``) instead of raising from
      ``eb-metrics`` or silently governing a finite remainder. More than 20%
      non-finite rows, or fewer than 8 finite aligned rows, fail closed.
    - FAS review is mandatory. Omitted or null ``fas_class`` / ``fas_class_col``
      fail closed as ``fas_class=BLOCKED``, ``status=red``, ``ral_policy=disallow``.
    """
    keys_list = list(keys)
    if len(keys_list) == 0:
        raise ValueError("`keys` must contain at least one grouping column.")

    required = set(keys_list) | {actual_col, base_forecast_col, ral_forecast_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gate_decisions = evaluate_governance_panel_df(
        df=df,
        keys=keys_list,
        actual_col=actual_col,
        base_forecast_col=base_forecast_col,
        ral_forecast_col=ral_forecast_col,
        tau=tau,
        cwsl_r=cwsl_r,
        preset=preset,
        dqc_thresholds=dqc_thresholds,
        fpc_thresholds=fpc_thresholds,
        dropna_keys=dropna_keys,
        fas_class=fas_class,
        fas_class_col=fas_class_col,
    )
    if decisions_df is None:
        decisions_df = gate_decisions
    else:
        missing_decision_keys = sorted(set(keys_list) - set(decisions_df.columns))
        if missing_decision_keys:
            raise ValueError(
                f"Provided decisions_df is missing required key columns: {missing_decision_keys}"
            )
        decisions_df = _reconcile_injected_with_gate(
            _fail_close_incomplete_decisions(decisions_df),
            gate_decisions,
            keys_list,
        )

    # Defaults for governed output columns.
    out_base = out_base_col if out_base_col is not None else f"{base_forecast_col}_governed"
    out_ral = out_ral_col if out_ral_col is not None else f"{ral_forecast_col}_governed"

    try:
        panel_governed = apply_ral(
            df=df,
            decisions=decisions_df,
            key_cols=keys_list,
            yhat_base_col=base_forecast_col,
            yhat_ral_col=ral_forecast_col,
            snap_mode=snap_mode,
            nonneg_policy=nonneg_policy,
            infer_policy_from_recommendations=infer_policy_from_recommendations,
            out_base_col=out_base,
            out_ral_col=out_ral,
            out_audit_prefix=out_audit_prefix,
        )
    except ValueError as e:
        # Normalize missing-decision errors to a stable message for callers/tests.
        msg = str(e)
        if require_complete_decisions and (
            "missing governance decision rows" in msg
            or "missing governance decision row" in msg
            or "Missing governance decision" in msg
        ):
            raise ValueError(f"Missing governance decisions: {msg}") from e
        raise

    if require_complete_decisions:
        # Secondary guard (in case apply_ral changes to not hard-fail on missing joins).
        if "snap_required" not in panel_governed.columns:
            raise ValueError(
                "Governance decisions did not include 'snap_required' after join; "
                "cannot verify decision completeness."
            )

        snap_required = panel_governed["snap_required"]
        if bool(snap_required.isna().any()):
            missing_rows = panel_governed.loc[snap_required.isna(), keys_list]
            preview = missing_rows.drop_duplicates().head(10).to_dict(orient="records")
            raise ValueError(
                "Missing governance decisions for one or more key groups. "
                f"Missing key preview (up to 10): {preview}"
            )

        required = snap_required.fillna(False).astype(bool)
        if bool(required.any()):
            if "snap_unit" not in panel_governed.columns:
                raise ValueError(
                    "snap_required is True but snap_unit is missing; "
                    "refusing fail-open unsnapped forecasts."
                )
            units_obj = pd.to_numeric(panel_governed["snap_unit"], errors="coerce")
            units = (
                units_obj
                if isinstance(units_obj, pd.Series)
                else pd.Series(units_obj, index=panel_governed.index, name="snap_unit")
            )
            unit_vals = units.to_numpy(dtype=float)
            unit_ok = pd.Series(
                np.isfinite(unit_vals) & (unit_vals > 0.0), index=panel_governed.index
            )
            invalid = required & ~unit_ok
            if bool(invalid.any()):
                preview = (
                    panel_governed.loc[invalid, keys_list]
                    .drop_duplicates()
                    .head(10)
                    .to_dict(orient="records")
                )
                raise ValueError(
                    "snap_required is True but snap_unit is missing or not finite and > 0. "
                    f"Key preview (up to 10): {preview}"
                )

    return panel_governed, decisions_df


def run_governance_workflow_df_dict(
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
    decisions_df: pd.DataFrame | None = None,
    snap_mode: SnapMode = "ceil",
    nonneg_policy: NonnegPolicy | None = None,
    infer_policy_from_recommendations: bool = True,
    out_base_col: str | None = None,
    out_ral_col: str | None = None,
    out_audit_prefix: str = "ral_apply_",
    require_complete_decisions: bool = True,
) -> dict[str, Any]:
    """
    Same as run_governance_workflow_df, but returns a named dict payload.

    This is convenient for Retool / API usage where tuple unpacking is awkward.
    """
    panel_governed, decisions = run_governance_workflow_df(
        df=df,
        keys=keys,
        actual_col=actual_col,
        base_forecast_col=base_forecast_col,
        ral_forecast_col=ral_forecast_col,
        tau=tau,
        cwsl_r=cwsl_r,
        preset=preset,
        dqc_thresholds=dqc_thresholds,
        fpc_thresholds=fpc_thresholds,
        dropna_keys=dropna_keys,
        fas_class=fas_class,
        fas_class_col=fas_class_col,
        decisions_df=decisions_df,
        snap_mode=snap_mode,
        nonneg_policy=nonneg_policy,
        infer_policy_from_recommendations=infer_policy_from_recommendations,
        out_base_col=out_base_col,
        out_ral_col=out_ral_col,
        out_audit_prefix=out_audit_prefix,
        require_complete_decisions=require_complete_decisions,
    )
    return {"panel": panel_governed, "decisions": decisions}

"""
Governance workflow orchestration (DataFrame utilities).

This module provides a *workflow* layer that composes:

1) dataframe.governance_panel.evaluate_governance_panel_df
   -> per-stream governance summary/decision table

2) adjustment.ral.apply_ral
   -> join decisions back to a panel and apply governed post-processing
      (nonneg + snap-to-grid) to prediction columns

Design goals
------------
- Keep diagnostics pure: governance math lives in diagnostics/, not here.
- Keep orchestration explicit: this module wires panel -> decisions -> governed panel.
- Keep behavior auditable: returns both governed panel and decisions table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from eb_evaluation.adjustment.ral import NonnegPolicy, SnapMode, apply_ral
from eb_evaluation.dataframe.governance_panel import evaluate_governance_panel_df
from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fpc import FPCThresholds
from eb_evaluation.diagnostics.presets import GovernancePreset

__all__ = ["run_governance_workflow_df", "run_governance_workflow_df_dict"]


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
    - If `decisions_df` is provided, the workflow will NOT recompute decisions; it will
      apply the provided decisions and (optionally) validate completeness.
    """
    keys_list = list(keys)
    if len(keys_list) == 0:
        raise ValueError("`keys` must contain at least one grouping column.")

    required = set(keys_list) | {actual_col, base_forecast_col, ral_forecast_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if decisions_df is None:
        decisions_df = evaluate_governance_panel_df(
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
        )
    else:
        missing_decision_keys = sorted(set(keys_list) - set(decisions_df.columns))
        if missing_decision_keys:
            raise ValueError(
                f"Provided decisions_df is missing required key columns: {missing_decision_keys}"
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

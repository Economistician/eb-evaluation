"""
Public validation entrypoints for diagnostic artifacts.

This module defines stable, versioned entrypoints for running Electric Barometer
diagnostics. Consumers should prefer these functions over importing diagnostic
modules directly.

The goal is API stability: diagnostic implementations may evolve, but these
entrypoints remain stable and auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isnan
from typing import Literal

from .dqc import DQCResult, DQCThresholds, classify_dqc
from .fpc import FPCResult, FPCSignals, FPCThresholds, classify_fpc
from .governance import GovernanceDecision, decide_governance
from .presets import GovernancePreset, preset_thresholds
from .run import run_governance_gate as _run_governance_gate


def validate_fpc(
    *,
    signals: FPCSignals,
    thresholds: FPCThresholds | None = None,
) -> FPCResult:
    """
    Run Forecast Primitive Compatibility (FPC) classification.

    Parameters
    ----------
    signals:
        Precomputed observable signals used by FPC.
    thresholds:
        Optional governance thresholds. If not provided, defaults are used.

    Returns
    -------
    FPCResult
        Compatibility class + signals + rationale.
    """
    return classify_fpc(signals=signals, thresholds=thresholds)


def validate_dqc(
    *,
    y: Sequence[float],
    thresholds: DQCThresholds | None = None,
) -> DQCResult:
    """
    Run Demand Quantization Compatibility (DQC) classification.

    Parameters
    ----------
    y:
        Realized demand/usage series for a single entity.
    thresholds:
        Optional governance thresholds. If not provided, defaults are used.

    Returns
    -------
    DQCResult
        Quantization class + signals + rationale.
    """
    # Normalize to plain floats so downstream helpers never receive numpy arrays
    # that can raise on truthiness checks (e.g., `if y:`).
    y_list = _as_float_list(y)
    return classify_dqc(y=y_list, thresholds=thresholds)


def validate_governance(
    *,
    y: Sequence[float],
    fpc_signals_raw: FPCSignals,
    fpc_signals_snapped: FPCSignals | None = None,
    dqc_thresholds: DQCThresholds | None = None,
    fpc_thresholds: FPCThresholds | None = None,
    preset: str | GovernancePreset | None = None,
) -> GovernanceDecision:
    """
    Run the governance decision contract (DQC x FPC) for a single entity.

    This is the stable public entrypoint for governance-oriented evaluation.
    It combines:
    - Demand Quantization Compatibility (DQC): whether snapping to a demand grid is required
    - Forecast Primitive Compatibility (FPC): whether scale-based readiness adjustment (e.g., RAL)
      is structurally valid, evaluated in raw space and (when required) snapped space.

    Presets
    -------
    If `preset` is provided, callers MUST NOT also pass explicit `dqc_thresholds`
    and/or `fpc_thresholds`. Mixing a preset with explicit thresholds is
    ambiguous and is rejected.

    Parameters
    ----------
    y:
        Realized demand/usage series for a single entity. This should be the *raw*
        realized series (not pre-snapped).
    fpc_signals_raw:
        Observable FPC signals computed in raw units (no snapping).
    fpc_signals_snapped:
        Observable FPC signals computed after applying the snap-to-grid policy.
        Required when DQC indicates snapping is required; optional otherwise.
    dqc_thresholds:
        Optional thresholds for DQC classification.
    fpc_thresholds:
        Optional thresholds for FPC classification (applied to both raw and snapped signals).
    preset:
        Optional governance preset. Provide either:
        - preset name: {"conservative", "balanced", "aggressive"}, or
        - a GovernancePreset instance.

    Returns
    -------
    GovernanceDecision
        A deterministic, auditable decision artifact containing:
        - DQC result (structure / snap requirement)
        - FPC results (raw + snapped)
        - snapping and τ interpretation policy
        - readiness adjustment allowability policy (traffic-light status)

    Raises
    ------
    ValueError
        If `preset` is provided together with explicit thresholds.
    TypeError
        If `preset` is not a str or GovernancePreset.
    ValueError
        If `preset` is a str but not a known preset name.
    """
    if preset is not None and (dqc_thresholds is not None or fpc_thresholds is not None):
        raise ValueError(
            "Ambiguous governance configuration: provide either `preset` OR explicit "
            "`dqc_thresholds`/`fpc_thresholds`, not both."
        )

    dqc_thr = dqc_thresholds
    fpc_thr = fpc_thresholds
    if preset is not None:
        dqc_thr, fpc_thr = preset_thresholds(preset)

    # Normalize realized demand to plain floats to avoid numpy truthiness traps
    # in downstream DQC helpers invoked by decide_governance.
    y_list = _as_float_list(y)

    return decide_governance(
        y=y_list,
        fpc_signals_raw=fpc_signals_raw,
        fpc_signals_snapped=fpc_signals_snapped,
        dqc_thresholds=dqc_thr,
        fpc_thresholds=fpc_thr,
    )


RecommendedEvaluationMode = Literal["continuous", "pack_aware", "reroute_discrete"]


@dataclass(frozen=True)
class GateResult:
    """
    Result of running the governance gate from common raw inputs.

    This is a convenience orchestrator that:
    - computes DQC from realized demand y,
    - computes raw FPC signals from (y, yhat_base, yhat_ral, tau, cwsl_r),
    - optionally computes snapped FPC signals when DQC requires snapping,
    - runs the governance decision contract,
    - returns an explicit recommended evaluation mode for downstream routing.

    The authoritative decision is always `decision` (GovernanceDecision).
    """

    dqc: DQCResult
    fpc_raw: FPCResult
    fpc_snapped: FPCResult
    decision: GovernanceDecision
    recommended_mode: RecommendedEvaluationMode
    recommendations: tuple[str, ...] = ()


def _as_float_list(x: Sequence[float]) -> list[float]:
    return [float(v) for v in x]


def _validate_same_length(y: Sequence[float], yhat: Sequence[float], name: str) -> None:
    if len(y) != len(yhat):
        raise ValueError(
            f"Length mismatch: len(y)={len(y)} but len({name})={len(yhat)}. "
            "Inputs must be aligned by interval."
        )


def _is_nan(v: float) -> bool:
    return isinstance(v, float) and isnan(v)


def _coverage_rate(y: Sequence[float], yhat: Sequence[float]) -> float:
    """
    Simple service-coverage proxy: fraction of intervals where yhat >= y.
    """
    n = len(y)
    if n == 0:
        return 0.0
    ok = 0
    for yt, yh in zip(y, yhat, strict=True):
        if _is_nan(yt) or _is_nan(yh):
            continue
        if yh >= yt:
            ok += 1
    return ok / float(n)


def _hit_rate_at_tau(y: Sequence[float], yhat: Sequence[float], tau: float) -> float:
    """
    Tolerance hit rate: fraction of intervals where |yhat - y| <= tau.
    """
    n = len(y)
    if n == 0:
        return 0.0
    if tau < 0 or _is_nan(tau):
        return 0.0
    ok = 0
    for yt, yh in zip(y, yhat, strict=True):
        if _is_nan(yt) or _is_nan(yh):
            continue
        if abs(yh - yt) <= tau:
            ok += 1
    return ok / float(n)


def _conditional_shortfall_depth(y: Sequence[float], yhat: Sequence[float]) -> float:
    """
    UD-like severity proxy: mean shortfall among shortfall intervals.
    """
    total = 0.0
    k = 0
    for yt, yh in zip(y, yhat, strict=True):
        if _is_nan(yt) or _is_nan(yh):
            continue
        if yt > yh:
            total += yt - yh
            k += 1
    if k == 0:
        return 0.0
    return total / float(k)


def _cwsl_proxy(y: Sequence[float], yhat: Sequence[float], r: float) -> float:
    """
    Lightweight CWSL proxy: mean( r * shortfall + overbuild ).

    This preserves the key asymmetry knob for gating/orchestration without
    requiring a full metric dependency at the validate layer.
    """
    n = len(y)
    if n == 0:
        return 0.0
    if _is_nan(r) or r < 0:
        return 0.0
    total = 0.0
    k = 0
    for yt, yh in zip(y, yhat, strict=True):
        if _is_nan(yt) or _is_nan(yh):
            continue
        short = yt - yh if yt > yh else 0.0
        over = yh - yt if yh > yt else 0.0
        total += (r * short) + over
        k += 1
    if k == 0:
        return 0.0
    return total / float(k)


def _build_signals_from_series(
    *,
    y: Sequence[float],
    yhat_base: Sequence[float],
    yhat_ral: Sequence[float],
    tau: float,
    cwsl_r: float | None,
) -> FPCSignals:
    nsl_base = _coverage_rate(y, yhat_base)
    nsl_ral = _coverage_rate(y, yhat_ral)

    hr_base_tau = _hit_rate_at_tau(y, yhat_base, tau)
    hr_ral_tau = _hit_rate_at_tau(y, yhat_ral, tau)

    # Severity: use RAL series (post-intervention) as the "experienced" depth.
    ud = _conditional_shortfall_depth(y, yhat_ral)

    cwsl_base: float | None = None
    cwsl_ral: float | None = None
    if cwsl_r is not None:
        cwsl_base = _cwsl_proxy(y, yhat_base, cwsl_r)
        cwsl_ral = _cwsl_proxy(y, yhat_ral, cwsl_r)

    delta_nsl = nsl_ral - nsl_base
    delta_hr = hr_ral_tau - hr_base_tau
    delta_cwsl: float | None = None
    if cwsl_base is not None and cwsl_ral is not None:
        delta_cwsl = cwsl_ral - cwsl_base

    return FPCSignals(
        nsl_base=nsl_base,
        nsl_ral=nsl_ral,
        delta_nsl=delta_nsl,
        hr_base_tau=hr_base_tau,
        hr_ral_tau=hr_ral_tau,
        delta_hr_tau=delta_hr,
        ud=ud,
        cwsl_base=cwsl_base,
        cwsl_ral=cwsl_ral,
        delta_cwsl=delta_cwsl,
        intervals=len(y),
        shortfall_intervals=None,
    )


def run_governance_gate(
    *,
    y: Sequence[float],
    yhat_base: Sequence[float],
    yhat_ral: Sequence[float],
    tau: float,
    cwsl_r: float | None = None,
    dqc_thresholds: DQCThresholds | None = None,
    fpc_thresholds: FPCThresholds | None = None,
    preset: str | GovernancePreset | None = None,
) -> GateResult:
    """
    Orchestrate governance from common raw inputs.

    This helper computes FPC signals directly from series inputs and produces
    an explicit recommended evaluation mode:

    - "continuous": demand is continuous-like; evaluate in raw units.
    - "pack_aware": demand is quantized/packed; τ and adjustments should be interpreted
      in grid units; forecasts/adjustments should be snapped as required.
    - "reroute_discrete": forecast primitive is incompatible for readiness control; consider
      rerouting to a discrete/event/state decision model.

    Preset rules match validate_governance: do not mix preset with explicit thresholds.

    Notes
    -----
    This entrypoint delegates orchestration to diagnostics/run.py to ensure there
    is a single auditable "choke point" for governance gating behavior. This wrapper
    preserves the historical validate-layer signature and behavior (snap_mode="ceil",
    no post-prediction constraints).
    """
    # Keep the validate-layer ambiguity check for a clear, stable error message.
    if preset is not None and (dqc_thresholds is not None or fpc_thresholds is not None):
        raise ValueError(
            "Ambiguous governance configuration: provide either `preset` OR explicit "
            "`dqc_thresholds`/`fpc_thresholds`, not both."
        )

    gate = _run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=tau,
        cwsl_r=cwsl_r,
        dqc_thresholds=dqc_thresholds,
        fpc_thresholds=fpc_thresholds,
        preset=preset,
        snap_mode="ceil",
        nonneg_mode="none",
    )

    return GateResult(
        dqc=gate.dqc,
        fpc_raw=gate.fpc_raw,
        fpc_snapped=gate.fpc_snapped,
        decision=gate.decision,
        recommended_mode=gate.recommended_mode,
        recommendations=gate.recommendations,
    )

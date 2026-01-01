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

from .dqc import DQCResult, DQCThresholds, classify_dqc
from .fpc import FPCResult, FPCSignals, FPCThresholds, classify_fpc
from .governance import GovernanceDecision, decide_governance
from .presets import GovernancePreset, preset_thresholds


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
    return classify_dqc(y=y, thresholds=thresholds)


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

    return decide_governance(
        y=y,
        fpc_signals_raw=fpc_signals_raw,
        fpc_signals_snapped=fpc_signals_snapped,
        dqc_thresholds=dqc_thr,
        fpc_thresholds=fpc_thr,
    )

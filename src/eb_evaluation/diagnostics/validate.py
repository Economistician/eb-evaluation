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
) -> GovernanceDecision:
    """
    Run the governance decision contract (DQC x FPC) for a single entity.

    This is the stable public entrypoint for governance-oriented evaluation.
    It combines:
    - Demand Quantization Compatibility (DQC): whether snapping to a demand grid is required
    - Forecast Primitive Compatibility (FPC): whether scale-based readiness adjustment (e.g., RAL)
      is structurally valid, evaluated in raw space and (when required) snapped space.

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

    Returns
    -------
    GovernanceDecision
        A deterministic, auditable decision artifact containing:
        - DQC result (structure / snap requirement)
        - FPC results (raw + snapped)
        - snapping and τ interpretation policy
        - readiness adjustment allowability policy (traffic-light status)
    """
    return decide_governance(
        y=y,
        fpc_signals_raw=fpc_signals_raw,
        fpc_signals_snapped=fpc_signals_snapped,
        dqc_thresholds=dqc_thresholds,
        fpc_thresholds=fpc_thresholds,
    )

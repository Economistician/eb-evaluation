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

"""
Unit tests for public diagnostic validation entrypoints.

These tests ensure that validate_* functions are stable delegation layers
over the underlying diagnostic implementations. They do NOT re-test diagnostic
logic; they test API correctness and pass-through behavior.
"""

from __future__ import annotations

from eb_evaluation.diagnostics import (
    validate_dqc,
    validate_fpc,
)
from eb_evaluation.diagnostics.dqc import DQCClass, DQCThresholds
from eb_evaluation.diagnostics.fpc import (
    FPCClass,
    FPCSignals,
    FPCThresholds,
)


def test_validate_fpc_delegates_and_returns_result() -> None:
    signals = FPCSignals(
        nsl_base=0.20,
        nsl_ral=0.40,
        delta_nsl=0.20,
        hr_base_tau=0.30,
        hr_ral_tau=0.20,
        delta_hr_tau=-0.10,
        ud=5.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=1_000,
        shortfall_intervals=300,
    )

    result = validate_fpc(signals=signals)

    assert result.fpc_class is FPCClass.COMPATIBLE
    assert result.signals is signals
    assert "coverage_nontrivial_and_responsive_to_ral" in result.reasons


def test_validate_fpc_respects_threshold_override() -> None:
    signals = FPCSignals(
        nsl_base=0.04,
        nsl_ral=0.06,
        delta_nsl=0.02,
        hr_base_tau=0.06,
        hr_ral_tau=0.05,
        delta_hr_tau=-0.01,
        ud=8.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=500,
        shortfall_intervals=200,
    )

    strict = FPCThresholds(
        nsl_very_low=0.05,
        delta_nsl_tiny=0.03,
    )

    result = validate_fpc(signals=signals, thresholds=strict)
    assert result.fpc_class in (FPCClass.MARGINAL, FPCClass.INCOMPATIBLE)


def test_validate_dqc_delegates_and_returns_result() -> None:
    # Strong quantized signal (multiples of 8)
    y = [0.0] * 20 + [8.0] * 80 + [16.0] * 80

    result = validate_dqc(y=y)

    assert result.dqc_class in (DQCClass.QUANTIZED, DQCClass.PIECEWISE_PACKED)
    assert result.signals.granularity in (8.0, 4.0, 2.0, 1.0)
    assert result.signals.multiple_rate >= DQCThresholds().multiple_rate_quantized


def test_validate_dqc_respects_threshold_override() -> None:
    # Borderline grid signal
    y = [0.0] * 30 + [8.0] * 50 + [16.0] * 30 + [7.999999] * 10

    loose = DQCThresholds(
        multiple_rate_quantized=0.80,
        min_nonzero_obs=20,
    )

    result = validate_dqc(y=y, thresholds=loose)

    assert result.dqc_class in (
        DQCClass.QUANTIZED,
        DQCClass.PIECEWISE_PACKED,
        DQCClass.CONTINUOUS_LIKE,
    )

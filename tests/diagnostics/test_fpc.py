"""
Unit tests for Forecast Primitive Compatibility (FPC).

These tests validate that FPC classification behaves deterministically for
canonical signal patterns. FPC is a governance construct, so preserving
classification semantics across refactors is critical.
"""

from __future__ import annotations

from eb_evaluation.diagnostics.fpc import (
    FPCClass,
    FPCSignals,
    FPCThresholds,
    classify_fpc,
)


def test_fpc_incompatible_signature_low_coverage_tiny_response_low_hr() -> None:
    signals = FPCSignals(
        nsl_base=0.01,
        nsl_ral=0.015,
        delta_nsl=0.005,
        hr_base_tau=0.02,
        hr_ral_tau=0.01,
        delta_hr_tau=-0.01,
        ud=5.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=10_000,
        shortfall_intervals=9_900,
    )

    result = classify_fpc(signals)
    assert result.fpc_class is FPCClass.INCOMPATIBLE
    assert any("nsl_base<=" in r for r in result.reasons)
    assert any("delta_nsl<=" in r for r in result.reasons)
    assert any("hr_base_tau<=" in r for r in result.reasons)


def test_fpc_incompatible_signature_low_coverage_tiny_response_high_ud() -> None:
    signals = FPCSignals(
        nsl_base=0.02,
        nsl_ral=0.03,
        delta_nsl=0.01,
        hr_base_tau=0.10,  # not very low; UD drives the mismatch signature
        hr_ral_tau=0.06,
        delta_hr_tau=-0.04,
        ud=25.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=10_000,
        shortfall_intervals=9_500,
    )

    result = classify_fpc(signals)
    assert result.fpc_class is FPCClass.INCOMPATIBLE
    assert any("ud>=" in r for r in result.reasons)


def test_fpc_incompatible_signature_cost_penalty_supports_mismatch() -> None:
    # Here, cost escalation reinforces mismatch even if HR/UD are not extreme.
    signals = FPCSignals(
        nsl_base=0.02,
        nsl_ral=0.035,
        delta_nsl=0.015,
        hr_base_tau=0.08,
        hr_ral_tau=0.05,
        delta_hr_tau=-0.03,
        ud=8.0,
        cwsl_base=1.0,
        cwsl_ral=2.0,
        delta_cwsl=1.0,
        intervals=10_000,
        shortfall_intervals=9_600,
    )

    result = classify_fpc(signals)
    assert result.fpc_class is FPCClass.INCOMPATIBLE
    assert any("delta_cwsl>=" in r for r in result.reasons)


def test_fpc_compatible_signature_nontrivial_and_responsive() -> None:
    signals = FPCSignals(
        nsl_base=0.20,
        nsl_ral=0.40,
        delta_nsl=0.20,
        hr_base_tau=0.25,
        hr_ral_tau=0.15,  # HR@tau may drop with RAL; not disqualifying
        delta_hr_tau=-0.10,
        ud=6.0,
        cwsl_base=1.2,
        cwsl_ral=1.6,
        delta_cwsl=0.4,
        intervals=10_000,
        shortfall_intervals=8_000,
    )

    result = classify_fpc(signals)
    assert result.fpc_class is FPCClass.COMPATIBLE
    assert "coverage_nontrivial_and_responsive_to_ral" in result.reasons


def test_fpc_marginal_default_mixed_signature() -> None:
    signals = FPCSignals(
        nsl_base=0.10,
        nsl_ral=0.14,
        delta_nsl=0.04,  # some gain, but not strongly "material" under defaults
        hr_base_tau=0.12,
        hr_ral_tau=0.09,
        delta_hr_tau=-0.03,
        ud=9.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=10_000,
        shortfall_intervals=9_000,
    )

    result = classify_fpc(signals)
    assert result.fpc_class is FPCClass.MARGINAL


def test_fpc_incompatible_signature_prophet_like_negative_and_clipped() -> None:
    """
    Regression (Option C): Prophet-like incompatibility signature.

    In our stress test, Prophet produced a meaningful fraction of negative predictions
    for a nonnegative demand series. If a naive "fix" clips negatives to zero, the
    forecast primitive exhibits:
      - very low base coverage (NSL),
      - tiny/near-zero response under the control-like transform (delta NSL),
      - very low HR@tau at reference tau.

    This test uses a canonical, deterministic signature to lock in the expected FPC
    classification for that pattern.
    """
    signals = FPCSignals(
        # "Base" behaves like negative forecasts against positive demand
        nsl_base=0.01,
        # "RAL" behaves like clipping (but without meaningful readiness improvement)
        nsl_ral=0.011,
        delta_nsl=0.001,
        hr_base_tau=0.02,
        hr_ral_tau=0.02,
        delta_hr_tau=0.0,
        # UD is not required to drive the mismatch here; low coverage + tiny response + low HR does
        ud=6.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=10_000,
        shortfall_intervals=9_900,
    )

    result = classify_fpc(signals)
    assert result.fpc_class is FPCClass.INCOMPATIBLE
    assert any("nsl_base<=" in r for r in result.reasons)
    assert any("delta_nsl<=" in r for r in result.reasons)
    assert any("hr_base_tau<=" in r for r in result.reasons)


def test_fpc_thresholds_can_shift_borderline_case() -> None:
    # Same signals, but stricter/tighter thresholds can change classification.
    signals = FPCSignals(
        nsl_base=0.04,
        nsl_ral=0.06,
        delta_nsl=0.02,
        hr_base_tau=0.06,
        hr_ral_tau=0.04,
        delta_hr_tau=-0.02,
        ud=11.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=5_000,
        shortfall_intervals=4_700,
    )

    # Default: likely marginal (not enough for strong mismatch signature)
    default_result = classify_fpc(signals, thresholds=FPCThresholds())
    assert default_result.fpc_class in (FPCClass.MARGINAL, FPCClass.INCOMPATIBLE)

    # More aggressive mismatch thresholds should push toward INCOMPATIBLE
    aggressive = FPCThresholds(
        nsl_very_low=0.05,
        delta_nsl_tiny=0.03,
        hr_very_low=0.07,
        ud_high=10.0,
    )
    aggressive_result = classify_fpc(signals, thresholds=aggressive)
    assert aggressive_result.fpc_class is FPCClass.INCOMPATIBLE

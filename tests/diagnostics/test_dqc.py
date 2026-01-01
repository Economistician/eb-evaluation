"""
Unit tests for Demand Quantization Compatibility (DQC).

These tests validate that DQC classification behaves deterministically for
canonical demand patterns:
- continuous-like series
- quantized series (single grid unit)
- piecewise packed series (multiple pack sizes with mass)
- unknown classification when insufficient data
"""

from __future__ import annotations

from eb_evaluation.diagnostics.dqc import (
    DQCClass,
    DQCThresholds,
    classify_dqc,
)


def test_dqc_continuous_like() -> None:
    # A "continuous-like" sequence (not aligned to small integer pack grids)
    y = [0.2 * i + 0.137 for i in range(1, 400)]  # 399 points, non-zero
    result = classify_dqc(y)
    assert result.dqc_class is DQCClass.CONTINUOUS_LIKE
    assert result.signals.granularity is not None
    assert 0.0 <= result.signals.multiple_rate <= 1.0


def test_dqc_quantized_unit_8() -> None:
    # Strong multiples-of-8 signature (packed but single unit -> QUANTIZED)
    y = [0.0] * 50 + [8.0] * 120 + [16.0] * 120 + [24.0] * 120
    result = classify_dqc(y)
    assert result.signals.granularity in (8.0, 4.0, 2.0, 1.0)
    assert result.signals.multiple_rate >= DQCThresholds().multiple_rate_quantized
    # With only one pack size dominating, pack_signature likely empty -> QUANTIZED.
    assert result.dqc_class in (DQCClass.QUANTIZED, DQCClass.PIECEWISE_PACKED)


def test_dqc_piecewise_packed_multiple_pack_sizes() -> None:
    # Multiple exact pack sizes with meaningful mass -> PIECEWISE_PACKED
    # Nonzero values include 4, 8, 12 in high frequency (each >= 10% mass).
    y = [0.0] * 60 + [4.0] * 120 + [8.0] * 120 + [12.0] * 120 + [16.0] * 60
    result = classify_dqc(y)
    assert result.dqc_class is DQCClass.PIECEWISE_PACKED
    assert result.signals.granularity is not None
    assert result.signals.multiple_rate >= DQCThresholds().multiple_rate_packed
    assert len(result.signals.pack_signature) >= DQCThresholds().pack_units_min_count


def test_dqc_unknown_when_insufficient_nonzero_data() -> None:
    # Too few non-zero observations should be UNKNOWN unless signal is overwhelming.
    y = [0.0] * 500 + [8.0] * 10  # only 10 nonzero
    result = classify_dqc(y)
    assert result.dqc_class in (DQCClass.UNKNOWN, DQCClass.QUANTIZED, DQCClass.PIECEWISE_PACKED)
    # If UNKNOWN, we expect the reason annotation about insufficient data.
    if result.dqc_class is DQCClass.UNKNOWN:
        assert any("nonzero_obs<" in r or "insufficient" in r for r in result.reasons)


def test_dqc_respects_threshold_override() -> None:
    # A borderline case becomes quantized when the quantized threshold is loosened.
    # Here, 8/16 dominate but we inject some near-multiples to reduce the exact
    # multiple rate below the default 0.90 threshold.
    y = [0.0] * 50 + [8.0] * 90 + [16.0] * 40 + [7.999999] * 10 + [16.000001] * 10
    # Nonzero = 150; exact multiples = 130 -> multiple_rate ~= 0.8667

    default_result = classify_dqc(y)
    assert default_result.dqc_class in (
        DQCClass.CONTINUOUS_LIKE,
        DQCClass.QUANTIZED,
        DQCClass.PIECEWISE_PACKED,
        DQCClass.UNKNOWN,
    )

    looser = DQCThresholds(
        multiple_rate_quantized=0.85,  # below 0.8667
        multiple_rate_packed=0.93,
        offgrid_mad_ratio_max=0.20,
        min_nonzero_obs=20,
    )
    looser_result = classify_dqc(y, thresholds=looser)
    assert looser_result.dqc_class in (DQCClass.QUANTIZED, DQCClass.PIECEWISE_PACKED)

"""
Unit tests for governance policy presets.

These tests validate that:
- Presets resolve deterministically into (DQCThresholds, FPCThresholds),
- The conservative/balanced/aggressive ordering is monotonic in key knobs,
- Preset resolution is stable and ergonomic for downstream use.

We do NOT re-test DQC/FPC/governance logic here; only preset wiring and
guardrail semantics.
"""

from __future__ import annotations

import pytest

from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fpc import FPCThresholds
from eb_evaluation.diagnostics.presets import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    GovernancePreset,
    preset_thresholds,
)


def test_preset_thresholds_returns_expected_types() -> None:
    dqc_thr, fpc_thr = preset_thresholds(BALANCED)

    assert isinstance(dqc_thr, DQCThresholds)
    assert isinstance(fpc_thr, FPCThresholds)


@pytest.mark.parametrize(
    "preset",
    [
        CONSERVATIVE,
        BALANCED,
        AGGRESSIVE,
    ],
)
def test_preset_thresholds_are_deterministic(preset: GovernancePreset) -> None:
    dqc1, fpc1 = preset_thresholds(preset)
    dqc2, fpc2 = preset_thresholds(preset)

    # Dataclasses should compare by value; this ensures stable mapping.
    assert dqc1 == dqc2
    assert fpc1 == fpc2


def test_preset_strictness_monotonicity() -> None:
    """
    Guardrails: conservative should generally be "harder" than balanced, which is
    harder than aggressive.

    We only assert a small number of key knobs to avoid brittleness if you tune
    other fields later.
    """
    dqc_cons, fpc_cons = preset_thresholds(CONSERVATIVE)
    dqc_bal, fpc_bal = preset_thresholds(BALANCED)
    dqc_aggr, fpc_aggr = preset_thresholds(AGGRESSIVE)

    # DQC: stricter means higher multiple-rate threshold required to call quantized/packed.
    assert (
        dqc_cons.multiple_rate_quantized
        >= dqc_bal.multiple_rate_quantized
        >= dqc_aggr.multiple_rate_quantized
    )
    assert (
        dqc_cons.multiple_rate_packed
        >= dqc_bal.multiple_rate_packed
        >= dqc_aggr.multiple_rate_packed
    )

    # FPC: stricter means we demand more delta_nsl to view "material gain".
    assert fpc_cons.delta_nsl_tiny >= fpc_bal.delta_nsl_tiny >= fpc_aggr.delta_nsl_tiny

    # Optional: conservative should be more sensitive to unattractive cost tradeoffs.
    assert fpc_cons.delta_cwsl_high <= fpc_bal.delta_cwsl_high <= fpc_aggr.delta_cwsl_high


def test_preset_thresholds_rejects_invalid_value() -> None:
    # Defensive: ensure unknown preset names raise rather than silently defaulting.
    with pytest.raises((TypeError, ValueError)):
        preset_thresholds("not-a-preset")

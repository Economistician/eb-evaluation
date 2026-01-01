"""
Unit tests for public diagnostic validation entrypoints.

These tests ensure that validate_* functions are stable delegation layers
over the underlying diagnostic implementations. They do NOT re-test diagnostic
logic; they test API correctness and pass-through behavior.
"""

from __future__ import annotations

import pytest

from eb_evaluation.diagnostics import (
    preset_thresholds,
    validate_dqc,
    validate_fpc,
    validate_governance,
)
from eb_evaluation.diagnostics.dqc import DQCClass, DQCThresholds
from eb_evaluation.diagnostics.fpc import (
    FPCClass,
    FPCSignals,
    FPCThresholds,
)
from eb_evaluation.diagnostics.governance import (
    GovernancePreset,
    GovernanceStatus,
    RALPolicy,
    TauPolicy,
)
from eb_evaluation.diagnostics.validate import run_governance_gate


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


def test_validate_governance_delegates_and_returns_decision_artifact() -> None:
    # Strongly quantized demand: snapping should be required (by default thresholds).
    y = [0.0] * 20 + [4.0] * 60 + [8.0] * 60 + [12.0] * 40

    # Raw FPC: mismatch-like (should not govern allowability when snapping is required).
    fpc_raw = FPCSignals(
        nsl_base=0.02,
        nsl_ral=0.021,
        delta_nsl=0.001,
        hr_base_tau=0.03,
        hr_ral_tau=0.03,
        delta_hr_tau=0.0,
        ud=12.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=100,
        shortfall_intervals=None,
    )

    # Snapped FPC: unambiguously compatible under default thresholds:
    # nsl_base > 0.03 and delta_nsl > 0.05
    fpc_snapped = FPCSignals(
        nsl_base=0.06,
        nsl_ral=0.16,
        delta_nsl=0.10,
        hr_base_tau=0.03,
        hr_ral_tau=0.02,
        delta_hr_tau=-0.01,
        ud=3.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=100,
        shortfall_intervals=None,
    )

    decision = validate_governance(
        y=y,
        fpc_signals_raw=fpc_raw,
        fpc_signals_snapped=fpc_snapped,
    )

    # Contract-level assertions: we got a decision artifact with consistent policy fields.
    assert decision.snap_required is True
    assert decision.snap_unit is not None
    assert decision.tau_policy == TauPolicy.GRID_UNITS
    assert decision.ral_policy == RALPolicy.ALLOW_AFTER_SNAP
    assert decision.status == GovernanceStatus.GREEN
    assert decision.fpc_raw.fpc_class in (FPCClass.MARGINAL, FPCClass.INCOMPATIBLE)
    assert decision.fpc_snapped.fpc_class is FPCClass.COMPATIBLE


def test_validate_governance_respects_threshold_overrides() -> None:
    # Borderline quantized signal; we force snapping with explicit DQC thresholds
    # and loosen FPC threshold so a small gain can qualify.
    y = [0.0] * 10 + [4.0] * 60 + [8.0] * 60 + [12.0] * 60

    fpc = FPCSignals(
        nsl_base=0.08,
        nsl_ral=0.12,
        delta_nsl=0.04,
        hr_base_tau=0.10,
        hr_ral_tau=0.08,
        delta_hr_tau=-0.02,
        ud=3.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=100,
        shortfall_intervals=None,
    )

    dqc_thr = DQCThresholds(
        multiple_rate_quantized=0.70,
        multiple_rate_packed=0.85,
        offgrid_mad_ratio_max=0.20,
        min_nonzero_obs=20,
    )
    fpc_thr = FPCThresholds(delta_nsl_tiny=0.01)

    decision = validate_governance(
        y=y,
        fpc_signals_raw=fpc,
        fpc_signals_snapped=fpc,
        dqc_thresholds=dqc_thr,
        fpc_thresholds=fpc_thr,
    )

    assert decision.snap_required is True
    assert decision.tau_policy == TauPolicy.GRID_UNITS
    assert decision.fpc_snapped.fpc_class in (FPCClass.COMPATIBLE, FPCClass.MARGINAL)
    assert decision.status in (GovernanceStatus.GREEN, GovernanceStatus.YELLOW)


def test_validate_governance_accepts_preset_and_matches_preset_thresholds() -> None:
    """
    The validate_governance entrypoint should accept a preset and wire it into
    the underlying thresholds resolution, producing the same behavior as
    explicitly passing the preset's thresholds.
    """
    # Moderately grid-ish series that we can push over/under the snap boundary by preset.
    y = [0.0] * 20 + [4.0] * 50 + [8.0] * 40 + [12.0] * 30

    # Use compatible signals so policy outcome is driven by snap_required differences
    # rather than FPC ambiguity.
    fpc_ok = FPCSignals(
        nsl_base=0.08,
        nsl_ral=0.18,
        delta_nsl=0.10,
        hr_base_tau=0.10,
        hr_ral_tau=0.08,
        delta_hr_tau=-0.02,
        ud=3.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=140,
        shortfall_intervals=None,
    )

    preset = GovernancePreset.BALANCED
    dqc_thr, fpc_thr = preset_thresholds(preset)

    by_preset = validate_governance(
        y=y,
        fpc_signals_raw=fpc_ok,
        fpc_signals_snapped=fpc_ok,
        preset=preset,
    )
    explicit = validate_governance(
        y=y,
        fpc_signals_raw=fpc_ok,
        fpc_signals_snapped=fpc_ok,
        dqc_thresholds=dqc_thr,
        fpc_thresholds=fpc_thr,
    )

    assert by_preset.dqc.dqc_class == explicit.dqc.dqc_class
    assert by_preset.snap_required == explicit.snap_required
    assert by_preset.tau_policy == explicit.tau_policy
    assert by_preset.ral_policy == explicit.ral_policy
    assert by_preset.status == explicit.status


def test_validate_governance_rejects_preset_and_explicit_thresholds_together() -> None:
    """
    Contract: callers should choose either a preset or explicit thresholds.
    Mixing is ambiguous and should raise.
    """
    y = [0.0] * 10 + [4.0] * 20 + [8.0] * 20
    fpc = FPCSignals(
        nsl_base=0.08,
        nsl_ral=0.18,
        delta_nsl=0.10,
        hr_base_tau=0.10,
        hr_ral_tau=0.08,
        delta_hr_tau=-0.02,
        ud=3.0,
        cwsl_base=None,
        cwsl_ral=None,
        delta_cwsl=None,
        intervals=50,
        shortfall_intervals=None,
    )

    with pytest.raises(ValueError):
        validate_governance(
            y=y,
            fpc_signals_raw=fpc,
            fpc_signals_snapped=fpc,
            preset=GovernancePreset.BALANCED,
            dqc_thresholds=DQCThresholds(),
        )


def test_run_governance_gate_recommended_modes() -> None:
    # 1) Continuous-like demand => continuous
    y_cont = [0.1 * i for i in range(1, 101)]
    # Baseline covers about half the intervals; RAL covers (almost) all.
    yhat_base_cont = [v if (i % 2 == 0) else (v * 0.90) for i, v in enumerate(y_cont)]
    yhat_ral_cont = [v * 1.01 for v in y_cont]
    gate = run_governance_gate(
        y=y_cont,
        yhat_base=yhat_base_cont,
        yhat_ral=yhat_ral_cont,
        tau=2.0,
        cwsl_r=None,
    )
    assert gate.recommended_mode == "continuous"

    # 2) Quantized demand => pack_aware (snapping required)
    y_q = [0.0] * 10 + [4.0] * 30 + [8.0] * 30 + [12.0] * 30
    yhat_base_q = [0.0] * 10 + [4.0] * 30 + [4.0] * 30 + [8.0] * 30  # underbuild in later blocks
    yhat_ral_q = y_q[:]  # perfect cover
    gate_q = run_governance_gate(
        y=y_q,
        yhat_base=yhat_base_q,
        yhat_ral=yhat_ral_q,
        tau=1.0,
        cwsl_r=None,
    )
    assert gate_q.recommended_mode in ("pack_aware", "reroute_discrete")
    assert gate_q.decision.snap_required is True

    # 3) Incompatible primitive => reroute_discrete
    y_bad = [10.0] * 80
    yhat_base_bad = [0.0] * 80
    yhat_ral_bad = [0.0] * 80
    gate_bad = run_governance_gate(
        y=y_bad,
        yhat_base=yhat_base_bad,
        yhat_ral=yhat_ral_bad,
        tau=0.5,
        cwsl_r=None,
    )
    assert gate_bad.recommended_mode == "reroute_discrete"

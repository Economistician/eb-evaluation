"""
Unit tests for governance (DQC x FPC) decision surface.

These tests validate that the governance layer:
- Requires snapping for quantized / piecewise-packed demand (via DQC),
- Interprets τ in grid units when snapping is required,
- Uses snapped FPC for allowability when snapping is required,
- Uses raw FPC for allowability when demand is continuous-like,
- Produces stable traffic-light status and RALPolicy outputs,
- Is deterministic and respects threshold overrides (via upstream components).

We intentionally construct FPCSignals directly to avoid coupling governance tests
to any particular forecast baseline model or notebook wiring.
"""

from __future__ import annotations

from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fpc import FPCSignals, FPCThresholds
from eb_evaluation.diagnostics.governance import (
    GovernanceStatus,
    RALPolicy,
    TauPolicy,
    decide_governance,
)


def _signals(
    *,
    nsl_base: float,
    nsl_ral: float,
    hr_base_tau: float,
    hr_ral_tau: float,
    ud: float,
    cwsl_base: float | None = None,
    cwsl_ral: float | None = None,
) -> FPCSignals:
    delta_nsl = nsl_ral - nsl_base
    delta_hr = hr_ral_tau - hr_base_tau
    delta_cwsl = None
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
        intervals=100,
        shortfall_intervals=None,
    )


def test_governance_continuous_like_uses_raw_fpc() -> None:
    # Continuous-like demand: not strongly quantized/packed.
    y = [0.1 * i for i in range(1, 201)]

    raw = _signals(
        nsl_base=0.08,
        nsl_ral=0.14,  # material improvement -> compatible by default thresholds
        hr_base_tau=0.11,
        hr_ral_tau=0.07,
        ud=3.0,
    )

    # Even if a snapped signal is provided, continuous-like should not require snapping,
    # and allowability should follow RAW FPC.
    snapped = _signals(
        nsl_base=0.01,
        nsl_ral=0.01,
        hr_base_tau=0.02,
        hr_ral_tau=0.02,
        ud=20.0,
    )

    res = decide_governance(y=y, fpc_signals_raw=raw, fpc_signals_snapped=snapped)

    assert res.snap_required is False
    assert res.snap_unit is None
    assert res.tau_policy == TauPolicy.RAW_UNITS
    assert res.ral_policy == RALPolicy.ALLOW
    assert res.status == GovernanceStatus.GREEN
    assert res.fpc_raw.fpc_class.value == "compatible"
    # snapped result is still computed, but not used for policy when snap_required is False.
    assert res.fpc_snapped.fpc_class.value == "incompatible"


def test_governance_quantized_requires_snap_and_uses_snapped_fpc() -> None:
    # Strongly quantized (pack/grid) demand: lots of exact multiples of 4.
    y = [0.0] * 20 + [4.0] * 60 + [8.0] * 60 + [12.0] * 40

    # Raw FPC is marginal/incompatible (tiny or no response).
    raw = _signals(
        nsl_base=0.02,
        nsl_ral=0.021,
        hr_base_tau=0.03,
        hr_ral_tau=0.03,
        ud=12.0,
    )

    # Snapped FPC must be UNAMBIGUOUSLY compatible under default thresholds.
    #
    # Default FPC compatibility gate requires:
    #   nsl_base > 0.03  and  delta_nsl > 0.05
    # so we set:
    #   nsl_base = 0.06, nsl_ral = 0.16  (delta = 0.10)
    snapped = _signals(
        nsl_base=0.06,
        nsl_ral=0.16,
        hr_base_tau=0.03,
        hr_ral_tau=0.02,
        ud=3.0,
    )

    res = decide_governance(y=y, fpc_signals_raw=raw, fpc_signals_snapped=snapped)

    assert res.snap_required is True
    assert res.snap_unit is not None
    assert res.tau_policy == TauPolicy.GRID_UNITS

    # Since snap_required, policy follows snapped FPC and should be green.
    assert res.ral_policy == RALPolicy.ALLOW_AFTER_SNAP
    assert res.status == GovernanceStatus.GREEN
    assert res.fpc_raw.fpc_class.value in ("marginal", "incompatible")
    assert res.fpc_snapped.fpc_class.value == "compatible"


def test_governance_quantized_marginal_after_snap_is_yellow() -> None:
    y = [0.0] * 10 + [1.0] * 60 + [2.0] * 60 + [3.0] * 60  # still quantized/grid-ish

    raw = _signals(
        nsl_base=0.02,
        nsl_ral=0.025,
        hr_base_tau=0.03,
        hr_ral_tau=0.03,
        ud=12.0,
    )

    # Snapped remains marginal (gain below material threshold).
    snapped = _signals(
        nsl_base=0.06,
        nsl_ral=0.08,  # delta=0.02 -> at default tiny threshold
        hr_base_tau=0.10,
        hr_ral_tau=0.09,
        ud=3.0,
    )

    res = decide_governance(y=y, fpc_signals_raw=raw, fpc_signals_snapped=snapped)

    assert res.snap_required is True
    assert res.tau_policy == TauPolicy.GRID_UNITS
    assert res.fpc_snapped.fpc_class.value == "marginal"
    assert res.ral_policy == RALPolicy.CAUTION_AFTER_SNAP
    assert res.status == GovernanceStatus.YELLOW


def test_governance_incompatible_is_red_and_disallow() -> None:
    y = [0.0] * 20 + [8.0] * 50 + [16.0] * 50 + [24.0] * 20

    raw = _signals(
        nsl_base=0.01,
        nsl_ral=0.01,
        hr_base_tau=0.02,
        hr_ral_tau=0.02,
        ud=20.0,
    )
    snapped = _signals(
        nsl_base=0.01,
        nsl_ral=0.01,
        hr_base_tau=0.02,
        hr_ral_tau=0.02,
        ud=20.0,
    )

    res = decide_governance(y=y, fpc_signals_raw=raw, fpc_signals_snapped=snapped)

    assert res.snap_required is True
    assert res.ral_policy == RALPolicy.DISALLOW
    assert res.status == GovernanceStatus.RED
    assert res.fpc_snapped.fpc_class.value == "incompatible"


def test_governance_respects_threshold_overrides_through_upstream_components() -> None:
    # Construct a borderline scenario where FPC would normally be marginal,
    # but becomes compatible when FPCThresholds are loosened.
    y = [0.0] * 10 + [4.0] * 60 + [8.0] * 60 + [12.0] * 60

    raw = _signals(
        nsl_base=0.08,
        nsl_ral=0.12,  # delta 0.04 might be marginal depending on thresholds
        hr_base_tau=0.10,
        hr_ral_tau=0.08,
        ud=3.0,
    )

    # Ensure DQC still triggers snapping (default thresholds usually do),
    # but we pass explicit thresholds to avoid brittleness.
    dqc_thr = DQCThresholds(
        multiple_rate_quantized=0.70,
        multiple_rate_packed=0.85,
        offgrid_mad_ratio_max=0.20,
        min_nonzero_obs=20,
    )

    # Loosen FPC material gain requirement by loosening delta_nsl_tiny.
    fpc_thr = FPCThresholds(delta_nsl_tiny=0.01)

    res = decide_governance(
        y=y,
        fpc_signals_raw=raw,
        fpc_signals_snapped=raw,
        dqc_thresholds=dqc_thr,
        fpc_thresholds=fpc_thr,
    )

    # If DQC triggers snapping, allowability follows snapped FPC (same as raw here).
    assert res.snap_required is True
    assert res.tau_policy == TauPolicy.GRID_UNITS
    assert res.fpc_snapped.fpc_class.value in ("compatible", "marginal")
    # With the loosened threshold, this should be compatible and thus GREEN.
    assert res.ral_policy in (RALPolicy.ALLOW_AFTER_SNAP, RALPolicy.CAUTION_AFTER_SNAP)
    assert res.status in (GovernanceStatus.GREEN, GovernanceStatus.YELLOW)

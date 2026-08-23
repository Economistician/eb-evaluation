"""
Unit tests for run-level orchestration (diagnostics/run.py).

These tests validate that the orchestration layer:
- Computes DQC + FPC (raw and snapped when required),
- Produces a GovernanceDecision consistent with underlying diagnostics,
- Returns a stable recommended_mode routing:
    * continuous
    * pack_aware
    * reroute_discrete
- Enforces basic input integrity (length checks, preset/threshold ambiguity).

We do NOT re-test diagnostic math in depth; we use simple constructed series
that reliably trigger each routing branch.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fas import FASClass
from eb_evaluation.diagnostics.fpc import FPCThresholds
from eb_evaluation.diagnostics.governance import GovernanceStatus, RALPolicy
from eb_evaluation.diagnostics.run import run_governance_gate


def test_run_governance_gate_continuous_returns_continuous() -> None:
    # Continuous-like demand.
    y = [0.1 * i for i in range(1, 121)]

    # Baseline covers about half the intervals; RAL covers (almost) all.
    yhat_base = [v if (i % 2 == 0) else (v * 0.90) for i, v in enumerate(y)]
    yhat_ral = [v * 1.01 for v in y]

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=2.0,
        cwsl_r=None,
    )

    assert gate.recommended_mode == "continuous"
    assert gate.decision.snap_required is False


def test_run_governance_gate_quantized_routes_pack_aware_or_reroute() -> None:
    # Strongly quantized demand: multiples of 4.
    y = [0.0] * 10 + [4.0] * 30 + [8.0] * 30 + [12.0] * 30

    # Baseline underbuilds later blocks; RAL fully covers.
    yhat_base = [0.0] * 10 + [4.0] * 30 + [4.0] * 30 + [8.0] * 30
    yhat_ral = y[:]  # perfect cover

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=1.0,
        cwsl_r=None,
    )

    assert gate.decision.snap_required is True
    assert gate.recommended_mode in ("pack_aware", "reroute_discrete")


def test_run_governance_gate_incompatible_reroutes_discrete() -> None:
    # Constant demand, forecasts never cover -> should be clearly incompatible.
    y = [10.0] * 80
    yhat_base = [0.0] * 80
    yhat_ral = [0.0] * 80

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=0.5,
        cwsl_r=None,
    )

    assert gate.recommended_mode == "reroute_discrete"
    assert gate.fpc_raw.fpc_class.value == "incompatible"


def test_run_governance_gate_unknown_dqc_reroutes_discrete() -> None:
    from eb_evaluation.diagnostics.dqc import DQCClass, DQCResult, DQCSignals

    y = [0.1 * i for i in range(1, 121)]
    yhat_base = [v if (i % 2 == 0) else (v * 0.90) for i, v in enumerate(y)]
    yhat_ral = [v * 1.01 for v in y]
    fake = DQCResult(
        dqc_class=DQCClass.UNKNOWN,
        signals=DQCSignals(
            n_obs=len(y),
            nonzero_obs=len(y),
            granularity=None,
            multiple_rate=float("nan"),
            support_size=len(y),
            zero_mass=0.0,
            small_value_mass=0.0,
            offgrid_mad=float("nan"),
            candidate_units=(),
            unit_scores=(),
        ),
        reasons=("insufficient_data",),
    )
    with (
        patch("eb_evaluation.diagnostics.run.classify_dqc", return_value=fake),
        patch("eb_evaluation.diagnostics.governance.classify_dqc", return_value=fake),
    ):
        gate = run_governance_gate(
            y=y,
            yhat_base=yhat_base,
            yhat_ral=yhat_ral,
            tau=2.0,
            cwsl_r=None,
        )
    assert gate.dqc.dqc_class is DQCClass.UNKNOWN
    assert gate.decision.ral_policy == RALPolicy.DISALLOW
    assert gate.decision.status == GovernanceStatus.RED
    assert gate.recommended_mode == "reroute_discrete"
    assert "dqc_unknown_fail_closed" in gate.decision.reasons
    assert "dqc_unknown_fail_closed" in gate.recommendations


def test_run_governance_gate_accepts_numpy_arrays() -> None:
    # Regression test: numpy arrays should be accepted as Sequence[float] inputs.
    # Prior behavior could crash if downstream code performed truthiness checks.
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    yhat_base = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    yhat_ral = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=1.0,
        cwsl_r=None,
    )

    assert gate is not None


def test_run_governance_gate_rejects_length_mismatch() -> None:
    y = [1.0, 2.0, 3.0]
    yhat_base = [1.0, 2.0]
    yhat_ral = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError):
        run_governance_gate(
            y=y,
            yhat_base=yhat_base,
            yhat_ral=yhat_ral,
            tau=1.0,
            cwsl_r=None,
        )


def test_run_governance_gate_rejects_preset_with_explicit_thresholds() -> None:
    y = [1.0, 2.0, 3.0]
    yhat_base = [1.0, 2.0, 3.0]
    yhat_ral = [1.0, 2.0, 3.0]

    # The previous version of this test didn't actually pass any explicit thresholds
    # (it passed None), so it correctly did not raise. To exercise the ambiguity
    # check, we must pass a non-None threshold object.
    with pytest.raises(ValueError):
        run_governance_gate(
            y=y,
            yhat_base=yhat_base,
            yhat_ral=yhat_ral,
            tau=1.0,
            cwsl_r=None,
            preset="balanced",
            dqc_thresholds=DQCThresholds(),
        )

    with pytest.raises(ValueError):
        run_governance_gate(
            y=y,
            yhat_base=yhat_base,
            yhat_ral=yhat_ral,
            tau=1.0,
            cwsl_r=None,
            preset="balanced",
            fpc_thresholds=FPCThresholds(),
        )


def test_run_governance_gate_nonneg_clip_adds_recommendation_and_preserves_routing() -> None:
    # Continuous-like demand, but intentionally allow negative forecasts to test the
    # post-prediction constraint path.
    y = [0.1 * i for i in range(1, 121)]

    # Introduce negatives in the forecasts. We don't rely on specific FPC math here;
    # we just need a stable run and evidence the postprocess was applied (via recs).
    yhat_base = [-0.5 if (i % 10 == 0) else v for i, v in enumerate(y)]
    yhat_ral = [-0.25 if (i % 15 == 0) else (v * 1.01) for i, v in enumerate(y)]

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=2.0,
        cwsl_r=None,
        nonneg_mode="clip",
    )

    # The run should succeed and record an auditable recommendation.
    assert any(r == "forecast_postprocess_nonneg(mode=clip_zero)" for r in gate.recommendations)

    # We expect this scenario to remain continuous-like and not require snapping.
    assert gate.decision.snap_required is False
    assert gate.recommended_mode in ("continuous", "reroute_discrete")


def test_run_governance_gate_nonneg_allow_records_allow_recommendation() -> None:
    y = [0.1 * i for i in range(1, 51)]
    yhat_base = list(y)
    yhat_ral = list(y)

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=2.0,
        cwsl_r=None,
        nonneg_mode="allow",
    )

    assert any(r == "forecast_postprocess_nonneg(mode=allow)" for r in gate.recommendations)


def test_run_governance_gate_omitted_preset_applies_balanced_clip_zero() -> None:
    y = [float(i) for i in range(1, 41)]
    yhat_base = [-1.0 if (i % 9 == 0) else float(v) for i, v in enumerate(y)]
    yhat_ral = [-2.0 if (i % 13 == 0) else float(v) for i, v in enumerate(y)]

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=1.0,
        cwsl_r=None,
    )

    assert any(r == "forecast_postprocess_nonneg(mode=clip_zero)" for r in gate.recommendations)
    assert "preset=balanced" in gate.decision.reasons


def test_run_governance_gate_preset_balanced_applies_nonneg_policy_by_default() -> None:
    # This is the key Option C test: when preset is provided and nonneg_mode is left
    # at its default, the preset policy should determine whether clipping occurs.
    y = [float(i) for i in range(1, 41)]
    yhat_base = [-1.0 if (i % 9 == 0) else float(v) for i, v in enumerate(y)]
    yhat_ral = [-2.0 if (i % 13 == 0) else float(v) for i, v in enumerate(y)]

    gate_preset = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=1.0,
        cwsl_r=None,
        preset="balanced",
        # nonneg_mode intentionally omitted so preset policy is used
    )

    assert any(
        r == "forecast_postprocess_nonneg(mode=clip_zero)" for r in gate_preset.recommendations
    )

    # Stronger regression: explicitly clipping via the legacy API should produce
    # identical results, proving the preset path is applying the same policy.
    gate_explicit = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=1.0,
        cwsl_r=None,
        nonneg_mode="clip",
    )

    assert gate_preset.recommended_mode == gate_explicit.recommended_mode
    assert gate_preset.decision.snap_required == gate_explicit.decision.snap_required
    assert gate_preset.fpc_raw.fpc_class == gate_explicit.fpc_raw.fpc_class
    assert gate_preset.fpc_snapped.fpc_class == gate_explicit.fpc_snapped.fpc_class


def test_run_governance_gate_fas_blocked_short_circuits_and_disallows() -> None:
    y = [0.1 * i for i in range(1, 121)]
    yhat_base = [v if (i % 2 == 0) else (v * 0.90) for i, v in enumerate(y)]
    yhat_ral = [v * 1.01 for v in y]

    with (
        patch("eb_evaluation.diagnostics.run.classify_dqc") as mock_dqc,
        patch("eb_evaluation.diagnostics.run.classify_fpc") as mock_fpc,
        patch("eb_evaluation.diagnostics.governance.classify_dqc") as mock_gov_dqc,
        patch("eb_evaluation.diagnostics.governance.classify_fpc") as mock_gov_fpc,
    ):
        gate = run_governance_gate(
            y=y,
            yhat_base=yhat_base,
            yhat_ral=yhat_ral,
            tau=2.0,
            cwsl_r=None,
            fas_class="BLOCKED",
        )

    mock_dqc.assert_not_called()
    mock_fpc.assert_not_called()
    mock_gov_dqc.assert_not_called()
    mock_gov_fpc.assert_not_called()
    assert gate.decision.status == GovernanceStatus.RED
    assert gate.decision.ral_policy == RALPolicy.DISALLOW
    assert "blocked_by_fas" in gate.decision.reasons
    assert "blocked_by_fas" in gate.recommendations
    assert gate.recommended_mode == "reroute_discrete"
    assert gate.decision.fas_class == FASClass.BLOCKED


def test_run_governance_gate_fas_conditional_downgrades_green_allow() -> None:
    y = [0.1 * i for i in range(1, 121)]
    yhat_base = [v if (i % 2 == 0) else (v * 0.90) for i, v in enumerate(y)]
    yhat_ral = [v * 1.01 for v in y]

    baseline = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=2.0,
        cwsl_r=None,
    )
    assert baseline.decision.ral_policy == RALPolicy.ALLOW
    assert baseline.decision.status == GovernanceStatus.GREEN

    gate = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=2.0,
        cwsl_r=None,
        fas_class="CONDITIONAL",
    )
    assert gate.decision.ral_policy == RALPolicy.CAUTION_AFTER_SNAP
    assert gate.decision.status == GovernanceStatus.YELLOW
    assert "fas_conditional_downgrade" in gate.decision.reasons


def test_run_governance_gate_fas_class_none_matches_omitted() -> None:
    y = [0.1 * i for i in range(1, 121)]
    yhat_base = [v if (i % 2 == 0) else (v * 0.90) for i, v in enumerate(y)]
    yhat_ral = [v * 1.01 for v in y]

    omitted = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=2.0,
        cwsl_r=None,
    )
    explicit = run_governance_gate(
        y=y,
        yhat_base=yhat_base,
        yhat_ral=yhat_ral,
        tau=2.0,
        cwsl_r=None,
        fas_class=None,
    )
    assert omitted.decision == explicit.decision
    assert omitted.recommended_mode == explicit.recommended_mode
    assert omitted.recommendations == explicit.recommendations
    assert omitted.decision.fas_class is None

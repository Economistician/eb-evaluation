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

import pytest

from eb_evaluation.diagnostics.dqc import DQCThresholds
from eb_evaluation.diagnostics.fpc import FPCThresholds
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

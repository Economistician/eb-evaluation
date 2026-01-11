"""
Unit tests for portable diagnostic result artifacts (diagnostics/results.py).

These tests validate the stability of the "portable contract" layer:
- GovernanceResult serializes deterministically via to_dict()
- GovernanceResult.from_gate_result() can adapt a GateResult-like object
  without tight coupling to internal dataclass layouts.

We intentionally avoid re-testing DQC/FPC/governance math here. The goal is
API/contract stability and future-proofing.
"""

from __future__ import annotations

from dataclasses import dataclass

from eb_evaluation.diagnostics.dqc import DQCClass
from eb_evaluation.diagnostics.fpc import FPCClass
from eb_evaluation.diagnostics.governance import GovernanceStatus, RALPolicy, TauPolicy
from eb_evaluation.diagnostics.results import GovernanceResult


@dataclass(frozen=True)
class _FakeDQC:
    dqc_class: DQCClass
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _FakeFPC:
    fpc_class: FPCClass
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _FakeDecision:
    snap_required: bool
    snap_unit: float | None
    tau_policy: TauPolicy
    ral_policy: RALPolicy
    status: GovernanceStatus


@dataclass(frozen=True)
class _FakeGate:
    """
    Minimal GateResult-like object that matches the attribute surface
    consumed by GovernanceResult.from_gate_result().
    """

    recommended_mode: str
    recommendations: tuple[str, ...]
    dqc: _FakeDQC
    fpc_raw: _FakeFPC
    fpc_snapped: _FakeFPC
    decision: _FakeDecision


def test_governance_result_to_dict_serializes_expected_fields() -> None:
    res = GovernanceResult(
        recommended_mode="pack_aware",
        snap_required=True,
        snap_unit=4.0,
        tau_policy=TauPolicy.GRID_UNITS,
        ral_policy=RALPolicy.ALLOW_AFTER_SNAP,
        status=GovernanceStatus.GREEN,
        dqc_class=DQCClass.QUANTIZED,
        fpc_raw_class=FPCClass.MARGINAL,
        fpc_snapped_class=FPCClass.COMPATIBLE,
        dqc_reasons=("multiple_rate_high",),
        fpc_raw_reasons=("coverage_low",),
        fpc_snapped_reasons=("coverage_nontrivial_and_responsive_to_ral",),
        recommendations=(
            "snap_required_interpret_tau_in_grid_units",
            "snap_forecasts_to_grid(mode=ceil)",
        ),
    )

    d = res.to_dict()

    # Core fields exist
    assert d["recommended_mode"] == "pack_aware"
    assert d["snap_required"] is True
    assert d["snap_unit"] == 4.0

    # Enum value serialization
    assert d["tau_policy"] == TauPolicy.GRID_UNITS.value
    assert d["ral_policy"] == RALPolicy.ALLOW_AFTER_SNAP.value
    assert d["status"] == GovernanceStatus.GREEN.value
    assert d["dqc_class"] == DQCClass.QUANTIZED.value
    assert d["fpc_raw_class"] == FPCClass.MARGINAL.value
    assert d["fpc_snapped_class"] == FPCClass.COMPATIBLE.value

    # Reasons included as both tuples and compact strings
    assert d["dqc_reasons"] == ("multiple_rate_high",)
    assert d["dqc_reasons_str"] == "multiple_rate_high"
    assert d["fpc_raw_reasons_str"] == "coverage_low"
    assert d["fpc_snapped_reasons_str"] == "coverage_nontrivial_and_responsive_to_ral"
    assert (
        d["recommendations_str"]
        == "snap_required_interpret_tau_in_grid_units|snap_forecasts_to_grid(mode=ceil)"
    )


def test_governance_result_from_gate_result_adapts_gate_like_object() -> None:
    gate = _FakeGate(
        recommended_mode="continuous",
        recommendations=("evaluate_in_raw_units",),
        dqc=_FakeDQC(dqc_class=DQCClass.CONTINUOUS_LIKE, reasons=("continuous_like",)),
        fpc_raw=_FakeFPC(fpc_class=FPCClass.COMPATIBLE, reasons=("ok",)),
        fpc_snapped=_FakeFPC(fpc_class=FPCClass.COMPATIBLE, reasons=("ok",)),
        decision=_FakeDecision(
            snap_required=False,
            snap_unit=None,
            tau_policy=TauPolicy.RAW_UNITS,
            ral_policy=RALPolicy.ALLOW,
            status=GovernanceStatus.GREEN,
        ),
    )

    res = GovernanceResult.from_gate_result(gate=gate)

    assert res.recommended_mode == "continuous"
    assert res.snap_required is False
    assert res.snap_unit is None
    assert res.tau_policy == TauPolicy.RAW_UNITS
    assert res.ral_policy == RALPolicy.ALLOW
    assert res.status == GovernanceStatus.GREEN

    assert res.dqc_class == DQCClass.CONTINUOUS_LIKE
    assert res.fpc_raw_class == FPCClass.COMPATIBLE
    assert res.fpc_snapped_class == FPCClass.COMPATIBLE

    assert res.dqc_reasons == ("continuous_like",)
    assert res.fpc_raw_reasons == ("ok",)
    assert res.fpc_snapped_reasons == ("ok",)
    assert res.recommendations == ("evaluate_in_raw_units",)


def test_governance_result_from_gate_result_allows_reason_overrides() -> None:
    gate = _FakeGate(
        recommended_mode="reroute_discrete",
        recommendations=("fpc_incompatible_reroute_to_discrete_decision_model",),
        dqc=_FakeDQC(dqc_class=DQCClass.QUANTIZED, reasons=("multiple_rate_high",)),
        fpc_raw=_FakeFPC(fpc_class=FPCClass.INCOMPATIBLE, reasons=("incompatible",)),
        fpc_snapped=_FakeFPC(fpc_class=FPCClass.INCOMPATIBLE, reasons=("incompatible",)),
        decision=_FakeDecision(
            snap_required=True,
            snap_unit=4.0,
            tau_policy=TauPolicy.GRID_UNITS,
            ral_policy=RALPolicy.DISALLOW,
            status=GovernanceStatus.RED,
        ),
    )

    res = GovernanceResult.from_gate_result(
        gate=gate,
        dqc_reasons=("override_dqc",),
        fpc_raw_reasons="override_raw",
        fpc_snapped_reasons=None,  # fallback to gate
        recommendations=("override_reco_1", "override_reco_2"),
    )

    assert res.dqc_reasons == ("override_dqc",)
    assert res.fpc_raw_reasons == ("override_raw",)
    assert res.fpc_snapped_reasons == ("incompatible",)
    assert res.recommendations == ("override_reco_1", "override_reco_2")

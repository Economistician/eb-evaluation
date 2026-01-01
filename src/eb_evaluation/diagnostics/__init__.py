"""
Diagnostics for Electric Barometer.

This subpackage provides derived, governance-oriented diagnostics that evaluate
the structural compatibility of forecast primitives with observed demand
processes and the validity of evaluation/control resolution under demand
quantization. These diagnostics consume Forecast Readiness Framework (FRF)
signals (e.g., NSL, UD, HR@τ, CWSL response) and produce interpretable
classifications used for evaluation, reporting, and policy gating.

Public API
----------
FPCClass
    Enumeration of forecast primitive compatibility states.
FPCSignals
    Container for observable diagnostic signals used in FPC classification.
FPCThresholds
    Interpretable threshold parameters governing FPC classification logic.
FPCResult
    Structured result containing compatibility class and rationale.
classify_fpc
    Deterministic classification of forecast primitive compatibility from
    observable signals.
validate_fpc
    Stable public entrypoint for running FPC diagnostics.

DQCClass
    Enumeration of demand quantization compatibility states.
DQCSignals
    Container for observable signals describing demand quantization.
DQCThresholds
    Interpretable threshold parameters governing DQC classification logic.
DQCResult
    Structured result containing quantization class and rationale.
classify_dqc
    Deterministic classification of demand quantization compatibility from a
    realized demand series.
validate_dqc
    Stable public entrypoint for running DQC diagnostics.
dqc_to_dict
    JSON-friendly serialization for DQCResult.

GovernanceStatus
    Traffic-light governance status (green/yellow/red).
RALPolicy
    Policy decision for whether RAL is allowed (and under what conditions).
TauPolicy
    Policy decision for interpreting τ in raw units vs grid units.
GovernanceDecision
    Authoritative governance decision bundle (DQC + FPC + policies).
decide_governance
    Stable public entrypoint for running combined DQC x FPC governance.

Notes
-----
Forecast Primitive Compatibility (FPC) is a **diagnostic and governance
construct**, not a performance metric and not an optimization objective. Its
role is to assess whether scale-based readiness adjustments (e.g., RAL) are a
structurally valid control lever for a given demand process before such
adjustments are applied or interpreted.

Demand Quantization Compatibility (DQC) is a companion governance diagnostic that
assesses whether realized demand behaves as continuous-like at the evaluation
resolution or is strongly quantized/packed (e.g., piecewise items sold in pack
sizes). When demand is quantized, tolerance-based diagnostics and readiness
adjustments should be interpreted or applied in quantized units (e.g., snap-to-
grid).

The governance layer composes DQC and FPC to produce an **authoritative**
downstream policy decision (snap required? τ in grid units? allow RAL?).
"""

from __future__ import annotations

from .dqc import (
    DQCClass,
    DQCResult,
    DQCSignals,
    DQCThresholds,
    classify_dqc,
    dqc_to_dict,
)
from .fpc import (
    FPCClass,
    FPCResult,
    FPCSignals,
    FPCThresholds,
    classify_fpc,
)
from .governance import (
    GovernanceDecision,
    GovernanceStatus,
    RALPolicy,
    TauPolicy,
    decide_governance,
)
from .validate import (
    validate_dqc,
    validate_fpc,
)

__all__ = [
    "DQCClass",
    "DQCResult",
    "DQCSignals",
    "DQCThresholds",
    "FPCClass",
    "FPCResult",
    "FPCSignals",
    "FPCThresholds",
    "GovernanceDecision",
    "GovernanceStatus",
    "RALPolicy",
    "TauPolicy",
    "classify_dqc",
    "classify_fpc",
    "decide_governance",
    "dqc_to_dict",
    "validate_dqc",
    "validate_fpc",
]

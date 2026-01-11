"""
Portable result artifacts for Electric Barometer diagnostics.

This module defines small, stable, JSON-friendly result containers intended for:

- returning governance outcomes from orchestration entrypoints,
- producing consistent panel outputs (DataFrame summaries),
- persisting or exporting governance decisions without depending on internal
  diagnostic object layouts.

Design goals
------------
- Keep this module *pure* (no orchestration, no pandas dependency).
- Provide stable, auditable fields with minimal redundancy.
- Avoid importing orchestration layers (e.g., diagnostics/run.py) to prevent cycles.
- Support both:
    * "single-stream" usage (per y / yhat stream)
    * "panel" summarization workflows (one row per stream)

Notes
-----
- The authoritative governance logic lives in `diagnostics/governance.py`
  and `diagnostics/run.py`. This module is a *representation* layer only.
- These dataclasses are intended to be used as stable return types from
  orchestration functions and as the source for DataFrame row construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from .dqc import DQCClass
from .fpc import FPCClass
from .governance import GovernanceStatus, RALPolicy, TauPolicy

_REASON_SEP: Final[str] = "|"


class _DQCResultLike(Protocol):
    """
    Minimal structural protocol for a DQC result artifact.
    """

    @property
    def dqc_class(self) -> DQCClass: ...

    @property
    def reasons(self) -> Sequence[str]: ...


class _FPCResultLike(Protocol):
    """
    Minimal structural protocol for an FPC result artifact.
    """

    @property
    def fpc_class(self) -> FPCClass: ...

    @property
    def reasons(self) -> Sequence[str]: ...


class _GovernanceDecisionLike(Protocol):
    """
    Minimal structural protocol for a governance decision artifact.
    """

    @property
    def snap_required(self) -> bool: ...

    @property
    def snap_unit(self) -> float | None: ...

    @property
    def tau_policy(self) -> TauPolicy: ...

    @property
    def ral_policy(self) -> RALPolicy: ...

    @property
    def status(self) -> GovernanceStatus: ...


class _GateLike(Protocol):
    """
    Minimal structural protocol for objects produced by run-level governance.

    We avoid importing diagnostics/run.py to prevent cycles, but we still want
    type-checker visibility into the attributes used by `from_gate_result`.

    Important: this protocol is intentionally *structural* (duck-typed). It does
    NOT require concrete DQCResult/FPCResult/GovernanceDecision instances. This
    keeps tests free to use light fake objects, and it keeps adapters resilient
    to internal object layout evolution.
    """

    @property
    def recommended_mode(self) -> str: ...

    @property
    def recommendations(self) -> Sequence[str]: ...

    @property
    def dqc(self) -> _DQCResultLike: ...

    @property
    def fpc_raw(self) -> _FPCResultLike: ...

    @property
    def fpc_snapped(self) -> _FPCResultLike: ...

    @property
    def decision(self) -> _GovernanceDecisionLike: ...


def _as_tuple_str(x: object) -> tuple[str, ...]:
    """
    Best-effort normalization to a tuple[str, ...].

    Accepts:
    - None -> ()
    - str -> (str,)
    - Sequence[str] -> tuple(...)
    - Any other -> (str(x),)
    """
    if x is None:
        return ()
    if isinstance(x, tuple) and all(isinstance(v, str) for v in x):
        return x
    if isinstance(x, str):
        return (x,)

    try:
        # Avoid importing collections.abc.Sequence just for isinstance checks.
        # We'll try iterability and coerce to strings.
        return tuple(str(v) for v in x)  # type: ignore[arg-type]
    except TypeError:
        return (str(x),)


def _reason_string(reasons: tuple[str, ...]) -> str:
    """
    Join reasons into a compact, pipe-delimited string for tables/logging.
    """
    if not reasons:
        return ""
    return _REASON_SEP.join(reasons)


@dataclass(frozen=True, slots=True)
class GovernanceResult:
    """
    Portable governance result for a single demand stream.

    This is the stable "summary contract" for governance outcomes and is designed
    to be detached from internal diagnostic object layouts. It intentionally mirrors
    the key fields consumers care about:

    - routing recommendation (recommended_mode)
    - snap requirement and grid unit (snap_required / snap_unit)
    - tau interpretation policy (tau_policy)
    - readiness adjustment policy (ral_policy)
    - overall traffic-light status (status)
    - diagnostic classes (dqc_class, fpc_raw_class, fpc_snapped_class)
    - compact reasons (for auditability / debugging)

    Fields
    ------
    recommended_mode:
        One of {"continuous", "pack_aware", "reroute_discrete"}.
    snap_required:
        Whether forecasts should be snapped to a grid prior to evaluation.
    snap_unit:
        Grid unit (granularity) when snapping is required; otherwise None.
    tau_policy:
        Whether tau should be interpreted in raw units or grid units downstream.
    ral_policy:
        Whether a readiness adjustment layer (RAL) is allowed (and under what conditions).
    status:
        Governance traffic-light status.
    dqc_class:
        Demand Quantization Compatibility class.
    fpc_raw_class:
        Forecast Primitive Compatibility class in raw space.
    fpc_snapped_class:
        Forecast Primitive Compatibility class in snapped space (equals raw when no snapping).
    dqc_reasons / fpc_raw_reasons / fpc_snapped_reasons / recommendations:
        Human-readable rationale strings from diagnostics/orchestration.
    """

    # Routing
    recommended_mode: str

    # Policies
    snap_required: bool
    snap_unit: float | None
    tau_policy: TauPolicy
    ral_policy: RALPolicy
    status: GovernanceStatus

    # Classes
    dqc_class: DQCClass
    fpc_raw_class: FPCClass
    fpc_snapped_class: FPCClass

    # Rationale
    dqc_reasons: tuple[str, ...] = ()
    fpc_raw_reasons: tuple[str, ...] = ()
    fpc_snapped_reasons: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-friendly dict.

        Enadrum conventions:
        - enums are serialized via `.value`
        - reasons are provided both as tuples and compact strings
        """
        return {
            "recommended_mode": self.recommended_mode,
            "snap_required": self.snap_required,
            "snap_unit": self.snap_unit,
            "tau_policy": self.tau_policy.value,
            "ral_policy": self.ral_policy.value,
            "status": self.status.value,
            "dqc_class": self.dqc_class.value,
            "fpc_raw_class": self.fpc_raw_class.value,
            "fpc_snapped_class": self.fpc_snapped_class.value,
            "dqc_reasons": self.dqc_reasons,
            "fpc_raw_reasons": self.fpc_raw_reasons,
            "fpc_snapped_reasons": self.fpc_snapped_reasons,
            "recommendations": self.recommendations,
            "dqc_reasons_str": _reason_string(self.dqc_reasons),
            "fpc_raw_reasons_str": _reason_string(self.fpc_raw_reasons),
            "fpc_snapped_reasons_str": _reason_string(self.fpc_snapped_reasons),
            "recommendations_str": _reason_string(self.recommendations),
        }

    @classmethod
    def from_gate_result(
        cls,
        *,
        gate: _GateLike,
        dqc_reasons: object | None = None,
        fpc_raw_reasons: object | None = None,
        fpc_snapped_reasons: object | None = None,
        recommendations: object | None = None,
    ) -> GovernanceResult:
        """
        Build a GovernanceResult from a run-level GateResult-like object.

        This is a *best-effort* adapter intended to avoid tight coupling to any
        specific GateResult dataclass layout. It expects the following attribute
        structure (as in diagnostics/run.py):

        - gate.recommended_mode: str
        - gate.recommendations: Sequence[str]
        - gate.dqc.dqc_class: DQCClass
        - gate.dqc.reasons: Sequence[str]
        - gate.fpc_raw.fpc_class: FPCClass
        - gate.fpc_raw.reasons: Sequence[str]
        - gate.fpc_snapped.fpc_class: FPCClass
        - gate.fpc_snapped.reasons: Sequence[str]
        - gate.decision.snap_required: bool
        - gate.decision.snap_unit: float | None
        - gate.decision.tau_policy: TauPolicy
        - gate.decision.ral_policy: RALPolicy
        - gate.decision.status: GovernanceStatus

        If you supply explicit *_reasons arguments, those override values read from `gate`.
        """
        recommended_mode = gate.recommended_mode

        decision = gate.decision
        dqc = gate.dqc
        fpc_raw = gate.fpc_raw
        fpc_snapped = gate.fpc_snapped

        dqc_reasons_eff = _as_tuple_str(
            dqc_reasons if dqc_reasons is not None else getattr(dqc, "reasons", ())
        )
        fpc_raw_reasons_eff = _as_tuple_str(
            fpc_raw_reasons if fpc_raw_reasons is not None else getattr(fpc_raw, "reasons", ())
        )
        fpc_snapped_reasons_eff = _as_tuple_str(
            fpc_snapped_reasons
            if fpc_snapped_reasons is not None
            else getattr(fpc_snapped, "reasons", ())
        )
        recs_eff = _as_tuple_str(
            recommendations if recommendations is not None else getattr(gate, "recommendations", ())
        )

        return cls(
            recommended_mode=str(recommended_mode),
            snap_required=bool(decision.snap_required),
            snap_unit=decision.snap_unit,
            tau_policy=decision.tau_policy,
            ral_policy=decision.ral_policy,
            status=decision.status,
            dqc_class=dqc.dqc_class,
            fpc_raw_class=fpc_raw.fpc_class,
            fpc_snapped_class=fpc_snapped.fpc_class,
            dqc_reasons=dqc_reasons_eff,
            fpc_raw_reasons=fpc_raw_reasons_eff,
            fpc_snapped_reasons=fpc_snapped_reasons_eff,
            recommendations=recs_eff,
        )

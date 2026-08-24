"""
Forecast governance entrypoints for Electric Barometer diagnostics.

This module unifies governance-oriented diagnostics into a single, stable
decision surface:

- Demand Quantization Compatibility (DQC) diagnoses whether realized demand is
  continuous-like or strongly quantized/packed (piecewise items, pack sizes).
- Forecast Primitive Compatibility (FPC) diagnoses whether a scale-based
  readiness adjustment (e.g., RAL) is a structurally valid control lever for the
  demand process at the evaluation resolution.

The resulting GovernanceDecision is an auditable, deterministic artifact used to
drive:
- snapping requirements (raw units vs grid units),
- tolerance policy interpretation (τ in raw units vs τ in grid units),
- readiness adjustment policy (allow / caution / disallow),
- reporting and downstream policy gating.

Decision contract (authoritative)
---------------------------------
Inputs:
- y (realized demand series) is used ONLY for DQC.
- fpc_signals_raw is REQUIRED and represents FPC signals computed on raw units.
- fpc_signals_snapped is REQUIRED when snap_required is true. Omitting it
  does not reuse raw FPC; snapped FPC is marked INCOMPATIBLE and RAL is
  disallowed.

Outputs:
- snap_required:
    True iff DQC class ∈ {quantized, piecewise_packed}.
- snap_unit:
    DQC granularity when snap_required else None.
- tau_policy:
    grid_units when snap_required else raw_units.
- ral_policy & status:
    Determined from FPC on:
      * snapped FPC when snap_required
      * raw FPC when continuous-like
    DQC UNKNOWN is fail-closed (DISALLOW / RED) and is never treated as
    continuous-like.

Policy presets:
- conservative / balanced / aggressive provide small, stable presets for
  governance thresholds. Explicit threshold overrides always win.

Architecture note
-----------------
Governance presets are owned by `eb_evaluation.diagnostics.presets`.

This module consumes presets only via:
- preset name strings (e.g., "balanced"), or
- `GovernancePreset` instances from `diagnostics.presets`.

This file should not define its own preset enums or duplicate preset containers,
as doing so can introduce type-identity conflicts in static analysis and
downstream call sites.

Notes
-----
This module is a governance layer. It is not a performance metric and is not an
optimization objective.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
import logging
from math import ceil, floor, isfinite, isnan

from .dqc import DQCClass, DQCResult, DQCSignals, DQCThresholds, classify_dqc
from .fas import FASClass
from .fpc import FPCClass, FPCResult, FPCSignals, FPCThresholds, classify_fpc
from .presets import GovernancePreset, preset_thresholds


class GovernanceStatus(StrEnum):
    """Traffic-light status for downstream gating."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class TauPolicy(StrEnum):
    """How to interpret tolerance τ downstream.

    Distinct from ``eb_optimization.TauPolicyArtifact``, the frozen τ-calibration
    policy produced by optimization.
    """

    RAW_UNITS = "raw_units"
    GRID_UNITS = "grid_units"


class RALPolicy(StrEnum):
    """Whether readiness adjustment is allowed downstream.

    Distinct from ``eb_optimization.RALPolicyArtifact``, the multiplicative
    uplift artifact produced by optimization.
    """

    ALLOW = "allow"
    ALLOW_AFTER_SNAP = "allow_after_snap"
    CAUTION_AFTER_SNAP = "caution_after_snap"
    DISALLOW = "disallow"


# Explicit names for importers that also bind eb-optimization policy artifacts.
GovernanceTauPolicy = TauPolicy
GovernanceRALPolicy = RALPolicy


@dataclass(frozen=True)
class GovernanceDecision:
    """
    Authoritative governance decision.

    Fields are designed to be stable and auditable. Downstream systems should
    use these values directly rather than re-implementing policy logic.
    """

    # Underlying diagnostics
    dqc: DQCResult
    fpc_raw: FPCResult
    fpc_snapped: FPCResult

    # Enforcement knobs
    snap_required: bool
    snap_unit: float | None
    tau_policy: TauPolicy
    ral_policy: RALPolicy
    status: GovernanceStatus

    # Optional convenience annotations
    max_delta_nsl_raw: float | None = None
    max_delta_nsl_snap: float | None = None

    # Optional upstream FAS class (None = not supplied; contract unchanged)
    fas_class: FASClass | None = None

    # Audit trail
    reasons: tuple[str, ...] = ()


def _as_list(y: Sequence[float] | Iterable[float]) -> list[float]:
    # Accept numpy arrays, pandas series, etc.
    return list(y)


def _require_finite_positive_snap_unit(unit: float | None) -> float:
    """Require a usable grid unit when DQC says snapping is required."""
    if unit is None:
        raise ValueError(
            "DQC requires snapping but snap_unit is missing; "
            "refusing fail-open unsnapped forecasts."
        )
    try:
        value = float(unit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DQC requires snapping but snap_unit is invalid ({unit!r}); "
            "refusing fail-open unsnapped forecasts."
        ) from exc
    if not isfinite(value) or value <= 0.0:
        raise ValueError(
            f"DQC requires snapping but snap_unit must be finite and > 0; got {unit!r}."
        )
    return value


def snap_to_grid(values: Sequence[float], unit: float, *, mode: str = "ceil") -> list[float]:
    """
    Snap values to the detected demand grid.

    We default to ceil snapping because readiness is a "build to cover" control in most
    operational settings (i.e., avoid underbuild). Downstream systems may choose
    alternate snapping (round/floor) but governance should be conservative.

    Parameters
    ----------
    values:
        Forecast values to snap.
    unit:
        Grid unit (granularity) to snap to. Must be > 0.
    mode:
        One of {"ceil", "round", "floor"}.

    Returns
    -------
    list[float]
        Snapped values.

    Raises
    ------
    ValueError
        If ``unit`` is NaN or ``unit <= 0``, or if a value is ``None``,
        non-numeric, or non-finite (NaN / ±inf).
    """
    if unit <= 0 or isnan(unit):
        raise ValueError(f"snap unit must be > 0; got {unit!r}")

    if mode not in {"ceil", "round", "floor"}:
        raise ValueError(f"Invalid snap mode: {mode}")

    snapped: list[float] = []
    inv = 1.0 / unit
    for v in values:
        if v is None:
            raise ValueError("snap_to_grid values must be numeric; got None")
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"snap_to_grid values must be numeric; got {v!r}") from exc
        if not isfinite(fv):
            raise ValueError(
                "snap_to_grid values must be finite; refusing fail-open NaN/inf forecasts."
            )

        q = fv * inv
        if mode == "ceil":
            qi = ceil(q)
        elif mode == "floor":
            qi = floor(q)
        else:
            # Half away from zero (the signed analogue of half-up).
            qi = floor(q + 0.5) if q >= 0.0 else ceil(q - 0.5)

        snapped.append(float(qi) * unit)
    return snapped


def build_fpc_signals(
    *,
    nsl_base: float,
    nsl_ral: float,
    hr_base_tau: float,
    hr_ral_tau: float,
    ud: float,
    cwsl_base: float | None = None,
    cwsl_ral: float | None = None,
    intervals: int | None = None,
    shortfall_intervals: int | None = None,
) -> FPCSignals:
    """
    Convenience builder for FPCSignals with derived deltas.

    This keeps the public governance entrypoint stable even if the FPCSignals
    dataclass grows fields later.
    """
    delta_nsl = nsl_ral - nsl_base
    delta_hr = hr_ral_tau - hr_base_tau
    delta_cwsl: float | None = None
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
        intervals=intervals,
        shortfall_intervals=shortfall_intervals,
    )


_LOGGER = logging.getLogger(__name__)
_KNOWN_FAS_TOKENS = frozenset(member.value.upper() for member in FASClass)
_UNKNOWN_FAS_FAIL_CLOSED = "unknown_fas_fail_closed"
_unknown_fas_fail_closed_count = 0


def unknown_fas_fail_closed_count() -> int:
    """Return how many unparseable FAS tokens have been fail-closed this process."""
    return _unknown_fas_fail_closed_count


def record_unknown_fas(raw: object) -> None:
    """Log and count an unparseable FAS token so batches can continue."""
    global _unknown_fas_fail_closed_count
    _unknown_fas_fail_closed_count += 1
    _LOGGER.warning(
        "Unknown fas_class %r; fail-closing stream as BLOCKED (count=%s).",
        raw,
        _unknown_fas_fail_closed_count,
    )


def fas_class_is_unparseable(fas_class: FASClass | str | None) -> bool:
    """Return True for a non-empty FAS token that is not a known class."""
    if fas_class is None or isinstance(fas_class, FASClass):
        return False
    token = str(fas_class).strip()
    if not token:
        return False
    return token.upper() not in _KNOWN_FAS_TOKENS


def _coerce_fas_class(fas_class: FASClass | str | None) -> FASClass | None:
    """Normalize an optional FAS class input to ``FASClass`` or ``None``.

    Unparseable tokens are logged and counted, then treated as missing so the
    caller can fail-close that stream without aborting a panel batch.
    """
    if fas_class is None:
        return None
    if isinstance(fas_class, FASClass):
        return fas_class
    token = str(fas_class).strip().upper()
    if not token:
        return None
    try:
        return FASClass(token)
    except ValueError:
        record_unknown_fas(fas_class)
        return None


def _incompatible_placeholder_fpc(reason: str) -> FPCResult:
    """Fail-closed FPC artifact when snapped evidence cannot be evaluated."""
    return FPCResult(
        fpc_class=FPCClass.INCOMPATIBLE,
        signals=FPCSignals(
            nsl_base=float("nan"),
            nsl_ral=float("nan"),
            delta_nsl=float("nan"),
            hr_base_tau=float("nan"),
            hr_ral_tau=float("nan"),
            delta_hr_tau=float("nan"),
            ud=float("nan"),
        ),
        reasons=(reason,),
    )


def _skipped_fas_blocked_diagnostics() -> tuple[DQCResult, FPCResult]:
    """Placeholder DQC/FPC artifacts when FAS blocks before evaluation."""
    dqc = DQCResult(
        dqc_class=DQCClass.UNKNOWN,
        signals=DQCSignals(
            n_obs=0,
            nonzero_obs=0,
            granularity=None,
            multiple_rate=float("nan"),
            support_size=0,
            zero_mass=float("nan"),
            small_value_mass=float("nan"),
            offgrid_mad=float("nan"),
            candidate_units=(),
            unit_scores=(),
        ),
        reasons=("skipped_fas_blocked",),
    )
    return dqc, _incompatible_placeholder_fpc("skipped_fas_blocked")


def _preset_reason_value(preset: str | GovernancePreset) -> str:
    """
    Normalize a preset into a stable, human-readable reason token.

    We normalize to lowercase to keep reasons consistent across:
    - string presets ("balanced")
    - GovernancePreset instances (name may vary by implementation)
    """
    if isinstance(preset, str):
        return preset.strip().lower()

    name = getattr(preset, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip().lower()

    return str(preset).strip().lower()


def decide_governance(
    *,
    y: Sequence[float],
    fpc_signals_raw: FPCSignals,
    fpc_signals_snapped: FPCSignals | None = None,
    dqc_thresholds: DQCThresholds | None = None,
    fpc_thresholds: FPCThresholds | None = None,
    preset: str | GovernancePreset = "balanced",
    fas_class: FASClass | str | None = None,
) -> GovernanceDecision:
    """
    Produce an authoritative governance decision for a single realized series.

    Inputs
    ------
    y:
        Realized demand series (used for DQC only).
    fpc_signals_raw:
        FPC signals computed in raw units.
    fpc_signals_snapped:
        FPC signals computed after snapping forecasts to the detected demand
        grid. Required when DQC sets ``snap_required``. If omitted in that
        case, snapped FPC is INCOMPATIBLE and RAL is disallowed. On
        continuous-like demand, omitted snapped signals reuse the raw
        classification.
    dqc_thresholds:
        Optional thresholds for DQC. Overrides preset thresholds.
    fpc_thresholds:
        Optional thresholds for FPC. Overrides preset thresholds.
    preset:
        Governance preset name ("conservative" | "balanced" | "aggressive") or an
        explicit GovernancePreset instance. Used only when explicit thresholds
        are not provided. Defaults to ``"balanced"``.
    fas_class:
        Required upstream Forecast Admissibility Surface class. ``BLOCKED``
        short-circuits DQC/FPC. ``CONDITIONAL`` downgrades permissive RAL
        outcomes.         ``None``, a missing/blank value, or an unknown token fail-closes as
        ``BLOCKED`` / ``DISALLOW`` / ``RED`` without raising, so a panel
        batch can continue for sibling streams.

    Returns
    -------
    GovernanceDecision
        Deterministic policy artifact.
    """
    reasons: list[str] = []
    unknown_fas = fas_class_is_unparseable(fas_class)
    fas = _coerce_fas_class(fas_class)
    if fas is None:
        dqc_skip, fpc_skip = _skipped_fas_blocked_diagnostics()
        reason = _UNKNOWN_FAS_FAIL_CLOSED if unknown_fas else "fas_required_fail_closed"
        return GovernanceDecision(
            dqc=dqc_skip,
            fpc_raw=fpc_skip,
            fpc_snapped=fpc_skip,
            snap_required=False,
            snap_unit=None,
            tau_policy=TauPolicy.RAW_UNITS,
            ral_policy=RALPolicy.DISALLOW,
            status=GovernanceStatus.RED,
            fas_class=FASClass.BLOCKED,
            reasons=(reason,),
        )

    if fas is FASClass.BLOCKED:
        dqc_skip, fpc_skip = _skipped_fas_blocked_diagnostics()
        return GovernanceDecision(
            dqc=dqc_skip,
            fpc_raw=fpc_skip,
            fpc_snapped=fpc_skip,
            snap_required=False,
            snap_unit=None,
            tau_policy=TauPolicy.RAW_UNITS,
            ral_policy=RALPolicy.DISALLOW,
            status=GovernanceStatus.RED,
            fas_class=fas,
            reasons=("blocked_by_fas",),
        )

    # Preset thresholds (explicit overrides win)
    preset_dqc, preset_fpc = preset_thresholds(preset)
    eff_dqc = dqc_thresholds or preset_dqc
    eff_fpc = fpc_thresholds or preset_fpc

    # 1) DQC classification from realized demand
    y_list = _as_list(y)
    dqc = classify_dqc(y_list, thresholds=eff_dqc)

    # 2) Snap requirement + tolerance policy (before FPC reuse).
    snap_required = dqc.dqc_class in (DQCClass.QUANTIZED, DQCClass.PIECEWISE_PACKED)
    snap_unit = dqc.signals.granularity if snap_required else None
    if snap_required:
        snap_unit = _require_finite_positive_snap_unit(snap_unit)
    tau_policy = TauPolicy.GRID_UNITS if snap_required else TauPolicy.RAW_UNITS

    # 3) FPC classification. Snapped evidence is required when snapping is
    #    required; omitting it is INCOMPATIBLE, not a silent reuse of raw FPC.
    fpc_raw = classify_fpc(fpc_signals_raw, thresholds=eff_fpc)
    if snap_required and fpc_signals_snapped is None:
        fpc_snapped = _incompatible_placeholder_fpc("snapped_fpc_required_but_omitted")
        reasons.append("snapped_fpc_required_but_omitted")
    elif fpc_signals_snapped is not None:
        fpc_snapped = classify_fpc(fpc_signals_snapped, thresholds=eff_fpc)
    else:
        fpc_snapped = fpc_raw

    # 4) RAL policy + status
    #    - DQC UNKNOWN is fail-closed (never treated as continuous-like).
    #    - If snapping is required, judge allowability off snapped FPC.
    #    - If continuous-like, judge off raw FPC.
    target_fpc = fpc_snapped if snap_required else fpc_raw

    ral_policy: RALPolicy
    status: GovernanceStatus

    if dqc.dqc_class is DQCClass.UNKNOWN:
        ral_policy = RALPolicy.DISALLOW
        status = GovernanceStatus.RED
        reasons.append("dqc_unknown_fail_closed")
    elif target_fpc.fpc_class == FPCClass.COMPATIBLE:
        if snap_required:
            ral_policy = RALPolicy.ALLOW_AFTER_SNAP
            reasons.append("compatible_after_snap")
        else:
            ral_policy = RALPolicy.ALLOW
            reasons.append("compatible_raw")
        status = GovernanceStatus.GREEN

    elif target_fpc.fpc_class == FPCClass.MARGINAL:
        if snap_required:
            ral_policy = RALPolicy.CAUTION_AFTER_SNAP
            reasons.append("marginal_after_snap")
            status = GovernanceStatus.YELLOW
        else:
            ral_policy = RALPolicy.DISALLOW
            reasons.append("marginal_raw_disallow")
            status = GovernanceStatus.RED

    else:  # INCOMPATIBLE
        ral_policy = RALPolicy.DISALLOW
        status = GovernanceStatus.RED
        reasons.append("incompatible")

    if (
        fas is FASClass.CONDITIONAL
        and status is GovernanceStatus.GREEN
        and ral_policy in (RALPolicy.ALLOW, RALPolicy.ALLOW_AFTER_SNAP)
    ):
        ral_policy = RALPolicy.CAUTION_AFTER_SNAP
        status = GovernanceStatus.YELLOW
        reasons.append("fas_conditional_downgrade")

    # Helpful annotation for auditability:
    # Only record the preset when *no* explicit threshold override is provided.
    if dqc_thresholds is None and fpc_thresholds is None:
        reasons.append(f"preset={_preset_reason_value(preset)}")

    return GovernanceDecision(
        dqc=dqc,
        fpc_raw=fpc_raw,
        fpc_snapped=fpc_snapped,
        snap_required=snap_required,
        snap_unit=snap_unit,
        tau_policy=tau_policy,
        ral_policy=ral_policy,
        status=status,
        fas_class=fas,
        reasons=tuple(reasons),
    )

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
- fpc_signals_snapped is OPTIONAL and represents FPC signals computed after
  snapping forecasts to the demand grid. If omitted, snapped == raw.

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

Policy presets:
- conservative / balanced / aggressive provide small, stable presets for
  governance thresholds. Explicit threshold overrides always win.

Notes
-----
This module is a governance layer. It is not a performance metric and is not an
optimization objective.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isnan

from .dqc import DQCClass, DQCResult, DQCThresholds, classify_dqc
from .fpc import FPCClass, FPCResult, FPCSignals, FPCThresholds, classify_fpc
from .presets import GovernancePreset, preset_thresholds


class GovernanceStatus(str, Enum):
    """Traffic-light status for downstream gating."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class TauPolicy(str, Enum):
    """How to interpret tolerance τ downstream."""

    RAW_UNITS = "raw_units"
    GRID_UNITS = "grid_units"


class RALPolicy(str, Enum):
    """Whether readiness adjustment is allowed downstream."""

    ALLOW = "allow"
    ALLOW_AFTER_SNAP = "allow_after_snap"
    CAUTION_AFTER_SNAP = "caution_after_snap"
    DISALLOW = "disallow"


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

    # Audit trail
    reasons: tuple[str, ...] = ()


def _as_list(y: Sequence[float] | Iterable[float]) -> list[float]:
    # Accept numpy arrays, pandas series, etc.
    return list(y)


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
    """
    if unit <= 0 or isnan(unit):
        return _as_list(values)

    if mode not in {"ceil", "round", "floor"}:
        raise ValueError(f"Invalid snap mode: {mode}")

    snapped: list[float] = []
    inv = 1.0 / unit
    for v in values:
        # Guard NaNs; pass through
        if v is None or (isinstance(v, float) and isnan(v)):
            snapped.append(v)  # type: ignore[arg-type]
            continue

        q = v * inv

        if mode == "ceil":
            # ceil(v/unit)*unit without importing numpy
            qi = int(q)
            if q > float(qi):
                qi += 1

        elif mode == "floor":
            # floor(v/unit)*unit
            qi = int(q)

        else:  # mode == "round"
            # nearest integer, half-up
            qi = int(q + 0.5)

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


def decide_governance(
    *,
    y: Sequence[float],
    fpc_signals_raw: FPCSignals,
    fpc_signals_snapped: FPCSignals | None = None,
    dqc_thresholds: DQCThresholds | None = None,
    fpc_thresholds: FPCThresholds | None = None,
    preset: str | GovernancePreset = "balanced",
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
        Optional FPC signals computed after snapping forecasts to the detected
        demand grid. If not provided, the snapped decision is treated as equal
        to the raw decision.
    dqc_thresholds:
        Optional thresholds for DQC. Overrides preset thresholds.
    fpc_thresholds:
        Optional thresholds for FPC. Overrides preset thresholds.
    preset:
        Governance preset name ("conservative" | "balanced" | "aggressive") or an
        explicit GovernancePreset instance. Used only when explicit thresholds
        are not provided.

    Returns
    -------
    GovernanceDecision
        Deterministic policy artifact.
    """
    reasons: list[str] = []

    # Preset thresholds (explicit overrides win)
    preset_dqc, preset_fpc = preset_thresholds(preset)
    eff_dqc = dqc_thresholds or preset_dqc
    eff_fpc = fpc_thresholds or preset_fpc

    # 1) DQC classification from realized demand
    y_list = _as_list(y)
    dqc = classify_dqc(y_list, thresholds=eff_dqc)

    # 2) FPC classification from provided signals
    fpc_raw = classify_fpc(fpc_signals_raw, thresholds=eff_fpc)
    fpc_snapped = (
        classify_fpc(fpc_signals_snapped, thresholds=eff_fpc)
        if fpc_signals_snapped is not None
        else fpc_raw
    )

    # 3) Snap requirement + tolerance policy
    snap_required = dqc.dqc_class in (DQCClass.QUANTIZED, DQCClass.PIECEWISE_PACKED)
    snap_unit = dqc.signals.granularity if snap_required else None
    tau_policy = TauPolicy.GRID_UNITS if snap_required else TauPolicy.RAW_UNITS

    # 4) RAL policy + status
    #    - If snapping is required, judge allowability off snapped FPC.
    #    - If continuous-like, judge off raw FPC.
    target_fpc = fpc_snapped if snap_required else fpc_raw

    ral_policy: RALPolicy
    status: GovernanceStatus

    if target_fpc.fpc_class == FPCClass.COMPATIBLE:
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
        else:
            ral_policy = RALPolicy.DISALLOW
            reasons.append("marginal_raw_disallow")
        status = GovernanceStatus.YELLOW if snap_required else GovernanceStatus.RED

    else:  # INCOMPATIBLE
        ral_policy = RALPolicy.DISALLOW
        status = GovernanceStatus.RED
        reasons.append("incompatible")

    # Helpful annotation for auditability:
    # Only record the preset when *no* explicit threshold override is provided.
    if dqc_thresholds is None and fpc_thresholds is None:
        preset_name = preset if isinstance(preset, str) else preset.name
        reasons.append(f"preset={preset_name}")

    return GovernanceDecision(
        dqc=dqc,
        fpc_raw=fpc_raw,
        fpc_snapped=fpc_snapped,
        snap_required=snap_required,
        snap_unit=snap_unit,
        tau_policy=tau_policy,
        ral_policy=ral_policy,
        status=status,
        reasons=tuple(reasons),
    )

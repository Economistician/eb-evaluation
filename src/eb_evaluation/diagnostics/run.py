"""Orchestrate DQC, FPC, and governance decisions for a single stream.

Wires evaluation series into diagnostics and a recommended routing mode.
Does not persist artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .dqc import DQCClass, DQCResult, DQCThresholds, classify_dqc
from .fas import FASClass
from .fpc import (
    FPCResult,
    FPCSignals,
    FPCThresholds,
    build_signals_from_series,
    classify_fpc,
)

# NOTE: snap_to_grid is defined in diagnostics/governance.py (not adjustment/snap.py).
from .governance import GovernanceDecision, decide_governance, snap_to_grid
from .presets import GovernancePreset, preset_policy, preset_thresholds

RecommendedMode = Literal["continuous", "pack_aware", "reroute_discrete"]

FloatArrayLike: TypeAlias = Sequence[float] | Iterable[float]

# Back-compat note:
# - "none"/"clip" were used during early prototyping.
# - Preset policy uses "allow"/"clip_zero".
NonnegMode: TypeAlias = Literal["none", "clip", "allow", "clip_zero"]


@dataclass(frozen=True)
class GateResult:
    """
    Combined governance artifact (run-level).

    This is intentionally small and "portable" as an in-memory object. If you
    later decide to persist it across systems, define a contract in eb-contracts.
    """

    dqc: DQCResult
    fpc_raw: FPCResult
    fpc_snapped: FPCResult
    decision: GovernanceDecision

    recommended_mode: RecommendedMode
    recommendations: tuple[str, ...] = ()


def _ensure_equal_length(
    a: Sequence[float], b: Sequence[float], *, name_a: str, name_b: str
) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"Length mismatch: {name_a} has {len(a)} values but {name_b} has {len(b)} values."
        )


def _to_float_list(x: FloatArrayLike) -> list[float]:
    # `list(np_array)` yields numpy scalar types; we normalize to plain `float`.
    return [float(v) for v in x]


def _normalize_nonneg_mode(mode: NonnegMode) -> Literal["allow", "clip_zero"]:
    """
    Normalize nonnegativity mode to the stable policy vocabulary.

    Returns
    -------
    Literal["allow", "clip_zero"]
        Canonical mode.

    Raises
    ------
    ValueError
        If `mode` is unknown.
    """
    if mode in ("none", "allow"):
        return "allow"
    if mode in ("clip", "clip_zero"):
        return "clip_zero"
    raise ValueError(f"Unknown nonneg_mode: {mode!r}")


def _apply_nonneg(x: Sequence[float], *, mode: Literal["allow", "clip_zero"]) -> list[float]:
    """
    Apply a non-negativity post-process to forecasts.

    Notes
    -----
    - This is intentionally located in the run-level orchestration module
      (not in model adapters) so it is auditable and governed.
    - Only forecasts are post-processed. Realized demand `y` is left untouched.
    """
    if mode == "allow":
        return [float(v) for v in x]
    if mode == "clip_zero":
        return [0.0 if float(v) < 0.0 else float(v) for v in x]
    # Defensive: type checkers should prevent this, but keep runtime robust.
    raise ValueError(f"Unknown nonneg_mode: {mode!r}")


def run_governance_gate(
    *,
    y: FloatArrayLike,
    yhat_base: FloatArrayLike,
    yhat_ral: FloatArrayLike,
    tau: float,
    cwsl_r: float | None = None,
    # thresholds / presets
    dqc_thresholds: DQCThresholds | None = None,
    fpc_thresholds: FPCThresholds | None = None,
    preset: GovernancePreset | str | None = None,
    # snapping behavior
    snap_mode: Literal["ceil", "round", "floor"] = "ceil",
    # post-prediction constraints
    nonneg_mode: NonnegMode = "none",
    fas_class: FASClass | str | None = None,
) -> GateResult:
    """
    Run the minimal governance gate and return a recommended evaluation mode.

    Routing semantics
    -----------------
    - "reroute_discrete":
        When FPC is INCOMPATIBLE for the applicable space (snapped if required,
        raw otherwise). This indicates scale-based readiness adjustment is not a
        valid control lever and discrete decision modeling should be used.
    - "pack_aware":
        When snapping is required by DQC and FPC is not incompatible. Indicates
        downstream evaluation should interpret τ in grid units and (if using RAL)
        apply snap-to-grid to adjusted forecasts before scoring.
    - "continuous":
        When demand is continuous-like and FPC is not incompatible.

    Parameters
    ----------
    y:
        Realized demand series.
    yhat_base:
        Baseline forecasts (raw units).
    yhat_ral:
        Readiness-adjusted forecasts (raw units).
    tau:
        Tolerance band for HR@τ (raw units). If snapping is required, governance
        indicates τ should be interpreted in grid units downstream.
    cwsl_r:
        Optional CWSL under/over cost ratio r (>= 1 typically). When provided,
        CWSL response is included in signals and may influence classification.
    dqc_thresholds, fpc_thresholds:
        Optional explicit thresholds. If `preset` is provided, explicit thresholds
        are not allowed.
    preset:
        Optional governance preset name/enum; determines default thresholds and
        policy knobs (e.g., nonnegativity).
    snap_mode:
        Snapping mode used when snapping is required.
    nonneg_mode:
        Post-prediction constraint applied to forecasts before FPC.

        Accepted values:
        - "none": use the preset nonnegativity policy. An omitted preset
          resolves to ``"balanced"`` (``clip_zero``).
        - "allow": leave forecasts unconstrained.
        - "clip" / "clip_zero": clip negative forecasts to 0.0.

        Explicit ``allow`` / ``clip`` / ``clip_zero`` always override the preset.
    fas_class:
        Optional upstream Forecast Admissibility Surface class. ``BLOCKED``
        skips DQC/FPC evaluation and returns a red / disallow decision.

    Returns
    -------
    GateResult
        Combined diagnostic results + governance decision + recommended routing.

    Raises
    ------
    ValueError
        If series lengths mismatch, or if `preset` is mixed with explicit thresholds.
    """
    # Normalize inputs to plain lists of floats up-front.
    #
    # 1) NumPy arrays are not typed as `Sequence[float]` (Pyright),
    # 2) Some downstream helpers may do truthiness checks (e.g., `if y:`)
    #    which raise for NumPy arrays: "truth value is ambiguous".
    #
    # Converting here makes the gate robust to numpy/pandas inputs and keeps
    # downstream diagnostics operating on a simple, predictable type.
    y_list = _to_float_list(y)
    yhat_base_list = _to_float_list(yhat_base)
    yhat_ral_list = _to_float_list(yhat_ral)

    _ensure_equal_length(y_list, yhat_base_list, name_a="y", name_b="yhat_base")
    _ensure_equal_length(y_list, yhat_ral_list, name_a="y", name_b="yhat_ral")

    if preset is not None and (dqc_thresholds is not None or fpc_thresholds is not None):
        raise ValueError(
            "Ambiguous configuration: provide either `preset` OR explicit "
            "`dqc_thresholds`/`fpc_thresholds`, not both."
        )

    # Resolve thresholds:
    # - If explicit thresholds provided, use them.
    # - Else if preset provided, use preset thresholds.
    # - Else fall back to component defaults (by passing None).
    eff_dqc = dqc_thresholds
    eff_fpc = fpc_thresholds
    if preset is not None:
        eff_dqc, eff_fpc = preset_thresholds(preset)

    recommendations: list[str] = []

    # Omitted preset is balanced so thresholds, audit reasons, and nonnegativity
    # cannot disagree (unconstrained forecasts classified under balanced rules).
    effective_preset: str | GovernancePreset = preset if preset is not None else "balanced"

    # Resolve nonnegativity policy:
    #
    # - Explicit allow/clip/clip_zero always wins.
    # - "none" (the default) applies the effective preset policy (balanced => clip_zero).
    if nonneg_mode != "none":
        nonneg_policy = _normalize_nonneg_mode(nonneg_mode)
    else:
        nonneg_policy = preset_policy(effective_preset)

    # Optional post-process (governed): enforce nonnegativity on forecasts.
    # This happens *before* computing FPC signals so diagnostics reflect the
    # same constrained forecasts you would actually score downstream.
    # Always emit the resolved policy so apply_ral can reconstruct the same constraint.
    recommendations.append(f"forecast_postprocess_nonneg(mode={nonneg_policy})")
    if nonneg_policy != "allow":
        yhat_base_list = _apply_nonneg(yhat_base_list, mode=nonneg_policy)
        yhat_ral_list = _apply_nonneg(yhat_ral_list, mode=nonneg_policy)

    fas_token: str | None
    if fas_class is None:
        fas_token = None
    elif isinstance(fas_class, FASClass):
        fas_token = fas_class.value
    else:
        fas_token = str(fas_class).strip().upper()

    if fas_token == FASClass.BLOCKED.value:
        dummy_signals = FPCSignals(
            nsl_base=float("nan"),
            nsl_ral=float("nan"),
            delta_nsl=float("nan"),
            hr_base_tau=float("nan"),
            hr_ral_tau=float("nan"),
            delta_hr_tau=float("nan"),
            ud=float("nan"),
        )
        decision = decide_governance(
            y=y_list,
            fpc_signals_raw=dummy_signals,
            fpc_signals_snapped=None,
            dqc_thresholds=dqc_thresholds,
            fpc_thresholds=fpc_thresholds,
            preset=effective_preset,
            fas_class=FASClass.BLOCKED,
        )
        recommendations.append("blocked_by_fas")
        return GateResult(
            dqc=decision.dqc,
            fpc_raw=decision.fpc_raw,
            fpc_snapped=decision.fpc_snapped,
            decision=decision,
            recommended_mode="reroute_discrete",
            recommendations=tuple(recommendations),
        )

    # 1) DQC from realized demand (structure only)
    # NOTE: classify_dqc should accept thresholds=None (use its internal defaults).
    dqc = classify_dqc(y=y_list, thresholds=eff_dqc)

    # 2) FPC raw signals + classification
    # NOTE: classify_fpc should accept thresholds=None (use its internal defaults).
    raw_signals = build_signals_from_series(
        y=y_list,
        yhat_base=yhat_base_list,
        yhat_ral=yhat_ral_list,
        tau=tau,
        cwsl_r=cwsl_r,
    )
    fpc_raw = classify_fpc(raw_signals, thresholds=eff_fpc)

    # 3) If DQC indicates snapping, compute snapped FPC signals too
    snap_required = dqc.dqc_class in (DQCClass.QUANTIZED, DQCClass.PIECEWISE_PACKED)
    if snap_required:
        gran = dqc.signals.granularity
        if gran is None:
            raise ValueError(
                "DQC requires snapping but granularity is missing; "
                "refusing to pass through unsnapped forecasts."
            )
        unit = float(gran)
        yhat_base_s = snap_to_grid(yhat_base_list, unit, mode=snap_mode)
        yhat_ral_s = snap_to_grid(yhat_ral_list, unit, mode=snap_mode)

        # If snap_mode can preserve negatives (e.g., round on negative inputs),
        # re-apply nonneg constraint post-snap when enabled.
        if nonneg_policy != "allow":
            yhat_base_s = _apply_nonneg(yhat_base_s, mode=nonneg_policy)
            yhat_ral_s = _apply_nonneg(yhat_ral_s, mode=nonneg_policy)

        snapped_signals = build_signals_from_series(
            y=y_list,
            yhat_base=yhat_base_s,  # snapped forecasts, same y
            yhat_ral=yhat_ral_s,
            tau=tau,  # governance will tell downstream how to interpret τ
            cwsl_r=cwsl_r,
        )
        fpc_snapped = classify_fpc(snapped_signals, thresholds=eff_fpc)
        fpc_signals_snapped: FPCSignals | None = snapped_signals
    else:
        fpc_snapped = fpc_raw
        fpc_signals_snapped = None

    # 4) Governance decision contract
    #
    # IMPORTANT:
    # - Only pass explicit threshold overrides through to governance when the caller
    #   actually provided them. If we always pass eff_* (including preset-derived),
    #   governance will treat them as explicit overrides and will suppress the preset
    #   audit reason ("preset=..."), breaking test_governance expectations.
    decision = decide_governance(
        y=y_list,
        fpc_signals_raw=raw_signals,
        fpc_signals_snapped=fpc_signals_snapped,
        dqc_thresholds=dqc_thresholds,
        fpc_thresholds=fpc_thresholds,
        preset=effective_preset,
        fas_class=fas_class,
    )

    # 5) Recommended routing mode
    # "applicable" FPC for routing matches governance policy selection:
    # - snapped when snap_required, else raw
    applicable = fpc_snapped if decision.snap_required else fpc_raw

    if applicable.fpc_class.value == "incompatible":
        recommended_mode: RecommendedMode = "reroute_discrete"
        recommendations.append("fpc_incompatible_reroute_to_discrete_decision_model")
    elif decision.snap_required:
        recommended_mode = "pack_aware"
        recommendations.append("snap_required_interpret_tau_in_grid_units")
        recommendations.append(f"snap_forecasts_to_grid(mode={snap_mode})")
    else:
        recommended_mode = "continuous"

    return GateResult(
        dqc=dqc,
        fpc_raw=fpc_raw,
        fpc_snapped=fpc_snapped,
        decision=decision,
        recommended_mode=recommended_mode,
        recommendations=tuple(recommendations),
    )

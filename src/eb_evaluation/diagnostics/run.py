"""Orchestrate DQC, FPC, and governance decisions for a single stream.

Wires evaluation series into diagnostics and a recommended routing mode.
Does not persist artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

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
from .governance import (
    GovernanceDecision,
    _snap_to_grid_array,
    decide_governance,
    fas_class_is_unparseable,
)
from .presets import GovernancePreset, preset_policy, preset_thresholds

RecommendedMode = Literal["continuous", "pack_aware", "reroute_discrete"]

FloatArrayLike: TypeAlias = Sequence[float] | Iterable[float] | np.ndarray

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


def _ensure_equal_length(a: np.ndarray, b: np.ndarray, *, name_a: str, name_b: str) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Length mismatch: {name_a} has {a.shape[0]} values but {name_b} has {b.shape[0]} values."
        )


def _as_float64_1d(x: FloatArrayLike) -> np.ndarray:
    """Convert gate inputs to a 1D float64 ndarray without a Python float list."""
    use_asarray = isinstance(x, np.ndarray) or (
        not isinstance(x, (str, bytes)) and (hasattr(x, "__array__") or isinstance(x, Sequence))
    )
    arr = np.asarray(x, dtype=np.float64) if use_asarray else np.fromiter(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _invalid_demand_cell_count(values: np.ndarray) -> int:
    """Count NaN or negative cells in a raw demand series."""
    return int(np.count_nonzero(np.isnan(values) | (values < 0.0)))


def _placeholder_fpc_signals() -> FPCSignals:
    """NaN FPC signals used when the gate fail-closes before scoring."""
    return FPCSignals(
        nsl_base=float("nan"),
        nsl_ral=float("nan"),
        delta_nsl=float("nan"),
        hr_base_tau=float("nan"),
        hr_ral_tau=float("nan"),
        delta_hr_tau=float("nan"),
        ud=float("nan"),
    )


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


def _apply_nonneg(x: np.ndarray, *, mode: Literal["allow", "clip_zero"]) -> np.ndarray:
    """
    Apply a non-negativity post-process to forecasts.

    Notes
    -----
    - This is intentionally located in the run-level orchestration module
      (not in model adapters) so it is auditable and governed.
    - Only forecasts are post-processed. Realized demand `y` is left untouched.
    - Clip uses ``x < 0`` (not ``maximum``) so ``-0.0`` is preserved, matching
      the previous list-comprehension implementation.
    """
    if mode == "allow":
        return x
    if mode == "clip_zero":
        return np.where(x < 0.0, 0.0, x)
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
        Realized demand series. NaN or negative cells fail closed
        (``dqc_class=UNKNOWN``, ``status=red``, ``ral_policy=disallow``)
        instead of scoring a cleaned remainder.
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
        Required upstream Forecast Admissibility Surface class. ``BLOCKED``
        skips DQC/FPC evaluation and returns a red / disallow decision.
        ``None`` or a blank value fail-closes as ``BLOCKED`` / ``RED`` /
        ``DISALLOW``.

    Returns
    -------
    GateResult
        Combined diagnostic results + governance decision + recommended routing.

    Raises
    ------
    ValueError
        If series lengths mismatch, or if `preset` is mixed with explicit thresholds.
    """
    # Keep inputs as 1D float64 ndarrays. Downstream DQC/FPC helpers are
    # array-safe; converting to Python lists materializes ~70M-row series.
    y_arr = _as_float64_1d(y)
    yhat_base_arr = _as_float64_1d(yhat_base)
    yhat_ral_arr = _as_float64_1d(yhat_ral)

    _ensure_equal_length(y_arr, yhat_base_arr, name_a="y", name_b="yhat_base")
    _ensure_equal_length(y_arr, yhat_ral_arr, name_a="y", name_b="yhat_ral")

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
        yhat_base_arr = _apply_nonneg(yhat_base_arr, mode=nonneg_policy)
        yhat_ral_arr = _apply_nonneg(yhat_ral_arr, mode=nonneg_policy)

    fas_token: str | None
    if fas_class is None:
        fas_token = None
    elif isinstance(fas_class, FASClass):
        fas_token = fas_class.value
    else:
        fas_token = str(fas_class).strip().upper() or None

    unknown_fas = fas_class_is_unparseable(fas_class)
    if fas_token is None or fas_token == FASClass.BLOCKED.value or unknown_fas:
        dummy_signals = _placeholder_fpc_signals()
        fas_missing = fas_token is None and not unknown_fas
        decision = decide_governance(
            y=y_arr,
            fpc_signals_raw=dummy_signals,
            fpc_signals_snapped=None,
            dqc_thresholds=dqc_thresholds,
            fpc_thresholds=fpc_thresholds,
            preset=effective_preset,
            fas_class=fas_class if unknown_fas else (None if fas_missing else FASClass.BLOCKED),
        )
        if unknown_fas:
            recommendations.append("unknown_fas_fail_closed")
        else:
            recommendations.append("fas_required_fail_closed" if fas_missing else "blocked_by_fas")
        return GateResult(
            dqc=decision.dqc,
            fpc_raw=decision.fpc_raw,
            fpc_snapped=decision.fpc_snapped,
            decision=decision,
            recommended_mode="reroute_discrete",
            recommendations=tuple(recommendations),
        )

    if _invalid_demand_cell_count(y_arr):
        dummy_signals = _placeholder_fpc_signals()
        decision = decide_governance(
            y=y_arr,
            fpc_signals_raw=dummy_signals,
            fpc_signals_snapped=None,
            dqc_thresholds=dqc_thresholds,
            fpc_thresholds=fpc_thresholds,
            preset=effective_preset,
            fas_class=fas_class,
        )
        recommendations.append("invalid_values_fail_closed")
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
    dqc = classify_dqc(y=y_arr, thresholds=eff_dqc)

    # 2) FPC raw signals + classification
    # NOTE: classify_fpc should accept thresholds=None (use its internal defaults).
    raw_signals = build_signals_from_series(
        y=y_arr,
        yhat_base=yhat_base_arr,
        yhat_ral=yhat_ral_arr,
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
        yhat_base_s = _snap_to_grid_array(yhat_base_arr, unit, mode=snap_mode)
        yhat_ral_s = _snap_to_grid_array(yhat_ral_arr, unit, mode=snap_mode)

        # If snap_mode can preserve negatives (e.g., round on negative inputs),
        # re-apply nonneg constraint post-snap when enabled.
        if nonneg_policy != "allow":
            yhat_base_s = _apply_nonneg(yhat_base_s, mode=nonneg_policy)
            yhat_ral_s = _apply_nonneg(yhat_ral_s, mode=nonneg_policy)

        snapped_signals = build_signals_from_series(
            y=y_arr,
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
        y=y_arr,
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
    # DQC UNKNOWN is fail-closed and never routed as continuous-like.
    applicable = fpc_snapped if decision.snap_required else fpc_raw

    if dqc.dqc_class is DQCClass.UNKNOWN:
        recommended_mode: RecommendedMode = "reroute_discrete"
        recommendations.append("dqc_unknown_fail_closed")
    elif applicable.fpc_class.value == "incompatible":
        recommended_mode = "reroute_discrete"
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

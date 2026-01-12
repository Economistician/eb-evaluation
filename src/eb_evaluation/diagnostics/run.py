"""
Run-level orchestration for governance-oriented readiness evaluation.

This module is the "wiring layer" between:
- raw evaluation series (y, yhat_base, yhat_ral),
- diagnostic computations (DQC, FPC),
- governance decision surface (DQC x FPC),
- and a minimal recommended routing mode for downstream reporting.

Design goals
------------
- Keep diagnostic modules pure (no orchestration inside dqc.py / fpc.py).
- Keep validation entrypoints stable (validate.py remains a thin wrapper layer).
- Provide a single, auditable choke point to compute + gate readiness evaluation.

This module does NOT persist artifacts. Persistence/adapters should live in a
separate "artifacts" layer if/when needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .dqc import DQCClass, DQCResult, DQCThresholds, classify_dqc
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
        Optional post-prediction constraint applied to forecasts.

        Accepted values:
        - "none" / "allow": no change
        - "clip" / "clip_zero": clip negative forecasts to 0.0

        If `preset` is provided and `nonneg_mode` is left at the default ("none"),
        the preset's policy is applied.

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

    eff_dqc = dqc_thresholds
    eff_fpc = fpc_thresholds
    if preset is not None:
        eff_dqc, eff_fpc = preset_thresholds(preset)

    recommendations: list[str] = []

    # Resolve nonnegativity policy:
    #
    # - If caller explicitly sets nonneg_mode (not default "none"), it wins.
    # - Else if preset is provided, use preset policy.
    # - Else default is "allow".
    if nonneg_mode != "none":
        nonneg_policy = _normalize_nonneg_mode(nonneg_mode)
    elif preset is not None:
        nonneg_policy = preset_policy(preset)
    else:
        nonneg_policy = "allow"

    # Optional post-process (governed): enforce nonnegativity on forecasts.
    # This happens *before* computing FPC signals so diagnostics reflect the
    # same constrained forecasts you would actually score downstream.
    if nonneg_policy != "allow":
        recommendations.append(f"forecast_postprocess_nonneg(mode={nonneg_policy})")
        yhat_base_list = _apply_nonneg(yhat_base_list, mode=nonneg_policy)
        yhat_ral_list = _apply_nonneg(yhat_ral_list, mode=nonneg_policy)

    # 1) DQC from realized demand (structure only)
    dqc = classify_dqc(y=y_list, thresholds=eff_dqc)

    # 2) FPC raw signals + classification
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
    if snap_required and dqc.signals.granularity is not None:
        unit = float(dqc.signals.granularity)
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
    decision = decide_governance(
        y=y_list,
        fpc_signals_raw=raw_signals,
        fpc_signals_snapped=fpc_signals_snapped,
        dqc_thresholds=eff_dqc,
        fpc_thresholds=eff_fpc,
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

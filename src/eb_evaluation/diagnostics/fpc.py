"""
Forecast Primitive Compatibility (FPC) diagnostics.

This module defines a derived, auditable classification that diagnoses whether a
given forecast primitive (typically point forecasts with scale-based adjustment)
is structurally compatible with the observed demand process.

FPC is not a performance metric and is not intended as an optimization objective.
It consumes FRF diagnostics (e.g., NSL, UD, HR@τ, CWSL response) and produces a
small set of interpretable compatibility states used for governance and policy
gating.

Design goals:
- Deterministic and auditable (explicit signals in / signals out)
- Metric-agnostic (does not require a specific modeling approach)
- Lightweight dependencies (NumPy optional; Pandas optional)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import fabs, isnan


class FPCClass(str, Enum):
    """Compatibility taxonomy for forecast primitives."""

    COMPATIBLE = "compatible"
    MARGINAL = "marginal"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class FPCSignals:
    """
    Observable signals used to classify forecast primitive compatibility.

    All fields are intended to be computed from evaluation data and are
    interpretable on their own. None of these values should be "fitted".

    Notes:
    - nsl_* are fractions in [0, 1]
    - hr_* are fractions in [0, 1]
    - ud is a magnitude in units of y (same units as actual demand)
    - cwsl_* are dimensionless ratios (normalized by demand)
    """

    # Coverage / readiness response
    nsl_base: float
    nsl_ral: float
    delta_nsl: float

    # Tolerance hit-rate at a representative band (e.g., tau=5 units)
    hr_base_tau: float
    hr_ral_tau: float
    delta_hr_tau: float

    # Conditional shortfall depth (mean shortfall magnitude given shortfall)
    ud: float

    # Optional cost signals (can be None if not computed)
    cwsl_base: float | None = None
    cwsl_ral: float | None = None
    delta_cwsl: float | None = None

    # Contextual counts (optional but useful for auditability)
    intervals: int | None = None
    shortfall_intervals: int | None = None


@dataclass(frozen=True)
class FPCThresholds:
    """
    Thresholds for FPC classification.

    These are intentionally simple and interpretable. You can tune these to your
    operational context, but treat them as governance parameters (not a model).
    """

    nsl_very_low: float = 0.03
    delta_nsl_tiny: float = 0.02
    hr_very_low: float = 0.05
    delta_hr_large_drop: float = -0.10
    ud_high: float = 10.0
    delta_cwsl_high: float = 0.75


@dataclass(frozen=True)
class FPCResult:
    """Result container for classification + rationale."""

    fpc_class: FPCClass
    signals: FPCSignals
    reasons: tuple[str, ...]


def classify_fpc(
    signals: FPCSignals,
    thresholds: FPCThresholds | None = None,
) -> FPCResult:
    """
    Classify forecast primitive compatibility from observable signals.

    Strategy:
    - INCOMPATIBLE when we see a strong mismatch signature (very low coverage AND
      tiny response), reinforced by low tolerance hit-rate and/or high UD.
    - COMPATIBLE when coverage is meaningfully above zero AND responds to RAL in a
      material way.
    - Otherwise MARGINAL.
    """
    thr = thresholds or FPCThresholds()
    reasons: list[str] = []

    for name, val in (
        ("nsl_base", signals.nsl_base),
        ("nsl_ral", signals.nsl_ral),
        ("delta_nsl", signals.delta_nsl),
        ("hr_base_tau", signals.hr_base_tau),
        ("hr_ral_tau", signals.hr_ral_tau),
        ("delta_hr_tau", signals.delta_hr_tau),
        ("ud", signals.ud),
    ):
        if val is None or isnan(val):
            reasons.append(f"signal.{name}_missing_or_nan")

    very_low_cov = signals.nsl_base <= thr.nsl_very_low
    tiny_cov_response = signals.delta_nsl <= thr.delta_nsl_tiny
    very_low_hr = signals.hr_base_tau <= thr.hr_very_low
    high_ud = signals.ud >= thr.ud_high

    if very_low_cov:
        reasons.append(f"nsl_base<= {thr.nsl_very_low:g}")
    if tiny_cov_response:
        reasons.append(f"delta_nsl<= {thr.delta_nsl_tiny:g}")
    if very_low_hr:
        reasons.append(f"hr_base_tau<= {thr.hr_very_low:g}")
    if high_ud:
        reasons.append(f"ud>= {thr.ud_high:g}")

    cost_penalty = False
    if (
        signals.delta_cwsl is not None
        and not isnan(signals.delta_cwsl)
        and signals.delta_cwsl >= thr.delta_cwsl_high
    ):
        cost_penalty = True
        reasons.append(f"delta_cwsl>= {thr.delta_cwsl_high:g}")

    if very_low_cov and tiny_cov_response and (very_low_hr or high_ud or cost_penalty):
        return FPCResult(
            fpc_class=FPCClass.INCOMPATIBLE,
            signals=signals,
            reasons=tuple(reasons) if reasons else ("structural_mismatch_signature",),
        )

    compatible_cov = signals.nsl_base > thr.nsl_very_low
    material_gain = signals.delta_nsl > max(2.0 * thr.delta_nsl_tiny, 0.05)

    if compatible_cov and material_gain:
        reasons.append("coverage_nontrivial_and_responsive_to_ral")
        return FPCResult(
            fpc_class=FPCClass.COMPATIBLE,
            signals=signals,
            reasons=tuple(reasons),
        )

    if signals.delta_hr_tau <= thr.delta_hr_large_drop:
        reasons.append(f"delta_hr_tau<= {thr.delta_hr_large_drop:g}")

    return FPCResult(
        fpc_class=FPCClass.MARGINAL,
        signals=signals,
        reasons=tuple(reasons) if reasons else ("mixed_or_borderline_signature",),
    )


# ---------------------------------------------------------------------
# Helper utilities (pure Python)
# ---------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def compute_nsl(y: Sequence[float], yhat: Sequence[float]) -> float:
    if len(y) != len(yhat) or not y:
        return float("nan")
    return sum(1 for yi, yhi in zip(y, yhat, strict=True) if yhi >= yi) / float(len(y))


def compute_ud(y: Sequence[float], yhat: Sequence[float]) -> float:
    if len(y) != len(yhat) or not y:
        return float("nan")
    shortfalls = [yi - yhi for yi, yhi in zip(y, yhat, strict=True) if yi > yhi]
    return _mean(shortfalls) if shortfalls else 0.0


def compute_hr_tau(y: Sequence[float], yhat: Sequence[float], tau: float) -> float:
    if len(y) != len(yhat) or not y:
        return float("nan")
    return sum(1 for yi, yhi in zip(y, yhat, strict=True) if fabs(yhi - yi) <= tau) / float(len(y))


def compute_cwsl(
    y: Sequence[float],
    yhat: Sequence[float],
    *,
    r: float,
    c_over: float = 1.0,
) -> float:
    if len(y) != len(yhat) or not y:
        return float("nan")

    c_under = r * c_over
    short_sum = 0.0
    over_sum = 0.0
    y_sum = 0.0

    for yi, yhi in zip(y, yhat, strict=True):
        y_sum += yi
        if yi > yhi:
            short_sum += yi - yhi
        else:
            over_sum += yhi - yi

    if y_sum <= 0:
        return float("nan")

    return (c_under * short_sum + c_over * over_sum) / y_sum


def build_signals_from_series(
    *,
    y: Sequence[float],
    yhat_base: Sequence[float],
    yhat_ral: Sequence[float],
    tau: float,
    ud: float | None = None,
    cwsl_r: float | None = None,
) -> FPCSignals:
    nsl_base = compute_nsl(y, yhat_base)
    nsl_ral = compute_nsl(y, yhat_ral)
    delta_nsl = nsl_ral - nsl_base

    hr_base = compute_hr_tau(y, yhat_base, tau)
    hr_ral = compute_hr_tau(y, yhat_ral, tau)
    delta_hr = hr_ral - hr_base

    ud_val = ud if ud is not None else compute_ud(y, yhat_base)

    cwsl_base = cwsl_ral = delta_cwsl = None
    if cwsl_r is not None:
        cwsl_base = compute_cwsl(y, yhat_base, r=cwsl_r)
        cwsl_ral = compute_cwsl(y, yhat_ral, r=cwsl_r)
        delta_cwsl = cwsl_ral - cwsl_base

    return FPCSignals(
        nsl_base=nsl_base,
        nsl_ral=nsl_ral,
        delta_nsl=delta_nsl,
        hr_base_tau=hr_base,
        hr_ral_tau=hr_ral,
        delta_hr_tau=delta_hr,
        ud=ud_val,
        cwsl_base=cwsl_base,
        cwsl_ral=cwsl_ral,
        delta_cwsl=delta_cwsl,
        intervals=len(y),
        shortfall_intervals=None,
    )


def results_to_dict(result: FPCResult) -> dict[str, object]:
    s = result.signals
    return {
        "fpc_class": result.fpc_class.value,
        "reasons": list(result.reasons),
        "signals": {
            "nsl_base": s.nsl_base,
            "nsl_ral": s.nsl_ral,
            "delta_nsl": s.delta_nsl,
            "hr_base_tau": s.hr_base_tau,
            "hr_ral_tau": s.hr_ral_tau,
            "delta_hr_tau": s.delta_hr_tau,
            "ud": s.ud,
            "cwsl_base": s.cwsl_base,
            "cwsl_ral": s.cwsl_ral,
            "delta_cwsl": s.delta_cwsl,
            "intervals": s.intervals,
            "shortfall_intervals": s.shortfall_intervals,
        },
    }

"""
Demand Quantization Compatibility (DQC) diagnostics.

This module detects whether realized demand for an entity behaves as
continuous-like at the evaluation resolution, or is strongly quantized / packed
(e.g., items sold in pieces or pack sizes such as 4/8/12).

DQC is a governance-oriented diagnostic that complements Forecast Primitive
Compatibility (FPC):

- FPC asks: "Is a point-forecast + scale-adjustment primitive structurally valid?"
- DQC asks: "Is evaluation and control valid in continuous units, or must we
  respect discrete/pack quantization?"

If demand is quantized, tolerance-based diagnostics (HR@τ) and depth diagnostics
(UD) should be interpreted in pack units (or snapped to a grid), and readiness
policies should be quantization-aware (snap-to-grid adjustments).

Design goals:
- Deterministic and auditable (explicit signals in / signals out)
- Lightweight dependencies (pure Python)
- Robust to integer-like floats and mild measurement noise
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import fabs, isnan


class DQCClass(str, Enum):
    """Compatibility taxonomy for demand resolution / quantization."""

    CONTINUOUS_LIKE = "continuous_like"
    QUANTIZED = "quantized"
    PIECEWISE_PACKED = "piecewise_packed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DQCSignals:
    """
    Observable signals describing demand quantization.

    Fields:
    - n_obs: number of observations used for DQC (after cleaning)
    - nonzero_obs: number of strictly-positive observations used for grid tests
    - granularity: detected base unit (e.g., 1, 2, 4, 8)
    - multiple_rate: fraction of observations that lie on the granularity grid
    - support_size: number of unique (rounded) values observed
    - zero_mass: fraction of values equal to 0
    - small_value_mass: fraction of values in [1, small_value_max]
    - offgrid_mad: mean absolute distance to nearest grid multiple
    - candidate_units: the unit candidates considered
    - unit_scores: mapping of unit -> multiple_rate score
    - pack_signature: detected pack units (subset) when piecewise packed is detected
    """

    n_obs: int
    nonzero_obs: int
    granularity: float | None
    multiple_rate: float
    support_size: int
    zero_mass: float
    small_value_mass: float
    offgrid_mad: float
    candidate_units: tuple[float, ...]
    unit_scores: tuple[tuple[float, float], ...]
    pack_signature: tuple[float, ...] = ()


@dataclass(frozen=True)
class DQCThresholds:
    """
    Thresholds for DQC classification.

    These should be treated as governance parameters, not tuned per-entity.
    """

    # To consider a unit "real", a large majority of values should lie on its grid.
    multiple_rate_quantized: float = 0.90

    # Stronger threshold to assert piecewise packed (very "grid-like").
    multiple_rate_packed: float = 0.95

    # Off-grid mean absolute distance should be small relative to unit.
    offgrid_mad_ratio_max: float = 0.10

    # Minimum number of non-zero observations to classify confidently.
    min_nonzero_obs: int = 50

    # How we round for support counting & integer-like handling.
    round_decimals: int = 6

    # Small value range for "piece" behavior context (informational).
    small_value_max: float = 20.0

    # Candidate units to test for grid structure.
    candidate_units: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)

    # Pack signature detection: look for mass on a small set of pack sizes.
    # If at least this many distinct pack units appear with meaningful mass, call it piecewise packed.
    pack_units_min_count: int = 2
    pack_unit_mass_min: float = 0.10  # each pack unit must have >=10% of non-zero mass


@dataclass(frozen=True)
class DQCResult:
    """Result container for DQC classification + rationale."""

    dqc_class: DQCClass
    signals: DQCSignals
    reasons: tuple[str, ...]


def classify_dqc(
    y: Sequence[float],
    thresholds: DQCThresholds | None = None,
) -> DQCResult:
    """
    Classify demand quantization compatibility from realized demand values.

    Parameters
    ----------
    y:
        Nonnegative realized demand/usage series for a single entity.
    thresholds:
        Governance thresholds controlling classification behavior.

    Returns
    -------
    DQCResult
        Classification + signals + explicit rationale.
    """
    thr = thresholds or DQCThresholds()
    reasons: list[str] = []

    if not y:
        return DQCResult(
            dqc_class=DQCClass.UNKNOWN,
            signals=_empty_signals(thr),
            reasons=("empty_series",),
        )

    vals = _clean_nonneg(y, round_decimals=thr.round_decimals)
    if not vals:
        return DQCResult(
            dqc_class=DQCClass.UNKNOWN,
            signals=_empty_signals(thr),
            reasons=("no_valid_values",),
        )

    zero_mass = sum(1 for v in vals if v == 0.0) / float(len(vals))
    nonzero = [v for v in vals if v > 0.0]

    n_obs = len(vals)
    nonzero_obs = len(nonzero)

    support_size = len(set(vals))
    small_value_mass = (
        sum(1 for v in nonzero if v <= thr.small_value_max) / float(len(nonzero))
        if nonzero
        else 0.0
    )

    if len(nonzero) < thr.min_nonzero_obs:
        reasons.append(f"nonzero_obs<{thr.min_nonzero_obs}")

    # Detect best unit based on grid multiple rate.
    unit_scores: list[tuple[float, float]] = []
    for u in thr.candidate_units:
        score = _multiple_rate(nonzero, u)
        unit_scores.append((u, score))

    # Choose best unit with tie-break: prefer larger units only if score is essentially equal.
    best_u, best_score = _pick_best_unit(unit_scores)

    offgrid_mad = _offgrid_mad(nonzero, best_u) if best_u is not None else float("nan")
    offgrid_ratio = (
        (offgrid_mad / best_u)
        if (best_u is not None and best_u > 0 and not isnan(offgrid_mad))
        else float("nan")
    )

    if best_u is None or isnan(best_score):
        return DQCResult(
            dqc_class=DQCClass.UNKNOWN,
            signals=DQCSignals(
                n_obs=n_obs,
                nonzero_obs=nonzero_obs,
                granularity=None,
                multiple_rate=0.0,
                support_size=support_size,
                zero_mass=zero_mass,
                small_value_mass=small_value_mass,
                offgrid_mad=float("nan"),
                candidate_units=thr.candidate_units,
                unit_scores=tuple(unit_scores),
                pack_signature=(),
            ),
            reasons=("no_unit_detected",),
        )

    if best_score >= thr.multiple_rate_quantized:
        reasons.append(f"multiple_rate>= {thr.multiple_rate_quantized:g}")
    if not isnan(offgrid_ratio) and offgrid_ratio <= thr.offgrid_mad_ratio_max:
        reasons.append(f"offgrid_mad_ratio<= {thr.offgrid_mad_ratio_max:g}")

    # Pack signature detection (informative + may upgrade QUANTIZED -> PIECEWISE_PACKED)
    pack_signature = _detect_pack_signature(nonzero, best_u, thr)

    # Classification logic
    quantized_enough = best_score >= thr.multiple_rate_quantized and (
        isnan(offgrid_ratio) or offgrid_ratio <= thr.offgrid_mad_ratio_max
    )

    packed_enough = best_score >= thr.multiple_rate_packed and (
        isnan(offgrid_ratio) or offgrid_ratio <= thr.offgrid_mad_ratio_max
    )

    # If we don't have enough data, be conservative: UNKNOWN unless signal is overwhelming.
    if len(nonzero) < thr.min_nonzero_obs and not packed_enough:
        return DQCResult(
            dqc_class=DQCClass.UNKNOWN,
            signals=DQCSignals(
                n_obs=n_obs,
                nonzero_obs=nonzero_obs,
                granularity=best_u,
                multiple_rate=best_score,
                support_size=support_size,
                zero_mass=zero_mass,
                small_value_mass=small_value_mass,
                offgrid_mad=offgrid_mad,
                candidate_units=thr.candidate_units,
                unit_scores=tuple(unit_scores),
                pack_signature=pack_signature,
            ),
            reasons=tuple(reasons) if reasons else ("insufficient_data",),
        )

    if packed_enough and pack_signature:
        reasons.append("pack_signature_detected")
        return DQCResult(
            dqc_class=DQCClass.PIECEWISE_PACKED,
            signals=DQCSignals(
                n_obs=n_obs,
                nonzero_obs=nonzero_obs,
                granularity=best_u,
                multiple_rate=best_score,
                support_size=support_size,
                zero_mass=zero_mass,
                small_value_mass=small_value_mass,
                offgrid_mad=offgrid_mad,
                candidate_units=thr.candidate_units,
                unit_scores=tuple(unit_scores),
                pack_signature=pack_signature,
            ),
            reasons=tuple(reasons),
        )

    if quantized_enough:
        return DQCResult(
            dqc_class=DQCClass.QUANTIZED,
            signals=DQCSignals(
                n_obs=n_obs,
                nonzero_obs=nonzero_obs,
                granularity=best_u,
                multiple_rate=best_score,
                support_size=support_size,
                zero_mass=zero_mass,
                small_value_mass=small_value_mass,
                offgrid_mad=offgrid_mad,
                candidate_units=thr.candidate_units,
                unit_scores=tuple(unit_scores),
                pack_signature=pack_signature,
            ),
            reasons=tuple(reasons) if reasons else ("grid_structure_detected",),
        )

    return DQCResult(
        dqc_class=DQCClass.CONTINUOUS_LIKE,
        signals=DQCSignals(
            n_obs=n_obs,
            nonzero_obs=nonzero_obs,
            granularity=best_u,
            multiple_rate=best_score,
            support_size=support_size,
            zero_mass=zero_mass,
            small_value_mass=small_value_mass,
            offgrid_mad=offgrid_mad,
            candidate_units=thr.candidate_units,
            unit_scores=tuple(unit_scores),
            pack_signature=pack_signature,
        ),
        reasons=tuple(reasons) if reasons else ("no_strong_grid_structure",),
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _empty_signals(thr: DQCThresholds) -> DQCSignals:
    return DQCSignals(
        n_obs=0,
        nonzero_obs=0,
        granularity=None,
        multiple_rate=0.0,
        support_size=0,
        zero_mass=0.0,
        small_value_mass=0.0,
        offgrid_mad=float("nan"),
        candidate_units=thr.candidate_units,
        unit_scores=(),
        pack_signature=(),
    )


def _clean_nonneg(values: Sequence[float], *, round_decimals: int) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if isnan(fv):
            continue
        if fv < 0:
            continue
        out.append(round(fv, round_decimals))
    return out


def _multiple_rate(values: Sequence[float], unit: float) -> float:
    if not values:
        return float("nan")
    if unit <= 0:
        return float("nan")
    hits = 0
    for v in values:
        # distance to nearest multiple of unit
        k = round(v / unit)
        if fabs(v - (k * unit)) <= 1e-6:
            hits += 1
    return hits / float(len(values))


def _offgrid_mad(values: Sequence[float], unit: float) -> float:
    if not values:
        return float("nan")
    if unit <= 0:
        return float("nan")
    dsum = 0.0
    for v in values:
        k = round(v / unit)
        dsum += fabs(v - (k * unit))
    return dsum / float(len(values))


def _pick_best_unit(unit_scores: Sequence[tuple[float, float]]) -> tuple[float | None, float]:
    if not unit_scores:
        return None, float("nan")

    # Sort by score desc, then by unit desc as a gentle preference for coarser units when tied.
    sorted_scores = sorted(unit_scores, key=lambda p: (p[1], p[0]), reverse=True)
    best_u, best_s = sorted_scores[0]

    # Tie-break: if multiple units are within epsilon, prefer the larger unit (already sorted).
    eps = 1e-6
    top = [p for p in sorted_scores if fabs(p[1] - best_s) <= eps]
    if top:
        return top[0][0], top[0][1]
    return best_u, best_s


def _detect_pack_signature(
    nonzero: Sequence[float], granularity: float, thr: DQCThresholds
) -> tuple[float, ...]:
    """
    Detect a "piecewise packed" signature: multiple distinct pack sizes with meaningful mass.

    Method:
    - Consider a small family of likely pack sizes (multiples of base granularity).
    - Measure the fraction of nonzero observations that land exactly on each pack size.
    - If at least `pack_units_min_count` pack sizes each have >= `pack_unit_mass_min`,
      treat as a pack signature.

    This is intentionally conservative and interpretable.
    """
    if not nonzero:
        return ()
    if granularity <= 0:
        return ()

    # Candidate pack sizes as common multiples of the base unit.
    # This set is deliberately small and operationally motivated.
    multiples = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
    pack_sizes = [granularity * m for m in multiples]

    masses: list[tuple[float, float]] = []
    n = float(len(nonzero))

    for p in pack_sizes:
        ct = 0
        for v in nonzero:
            if fabs(v - p) <= 1e-6:
                ct += 1
        masses.append((p, ct / n))

    strong = [p for p, m in masses if m >= thr.pack_unit_mass_min]

    if len(strong) >= thr.pack_units_min_count:
        return tuple(strong)

    return ()


def dqc_to_dict(result: DQCResult) -> dict[str, object]:
    """Serialize a DQCResult to a JSON-friendly dictionary."""
    s = result.signals
    return {
        "dqc_class": result.dqc_class.value,
        "reasons": list(result.reasons),
        "signals": {
            "n_obs": s.n_obs,
            "nonzero_obs": s.nonzero_obs,
            "granularity": s.granularity,
            "multiple_rate": s.multiple_rate,
            "support_size": s.support_size,
            "zero_mass": s.zero_mass,
            "small_value_mass": s.small_value_mass,
            "offgrid_mad": s.offgrid_mad,
            "candidate_units": list(s.candidate_units),
            "unit_scores": [{"unit": u, "multiple_rate": sc} for u, sc in s.unit_scores],
            "pack_signature": list(s.pack_signature),
        },
    }

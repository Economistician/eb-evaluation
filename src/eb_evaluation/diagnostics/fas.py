"""
Forecast Admissibility Surface (FAS).

FAS is a deterministic, slice-keyed decision surface that partitions a forecast
domain into admissible regions for modeling and control. For a chosen slice
grain (e.g., entity, entity x interval, site x entity x interval), FAS produces a
joinable table whose primary output is a three-way class label:

    ALLOWED | CONDITIONAL | BLOCKED

Mathematically, FAS defines a mask over a slice domain D that partitions D into
A U C U B (allowed, conditional, blocked) using observable signals derived from
realized demand structure and baseline error anatomy (e.g., tail risk and spike
frequency). The resulting surface is designed to be merged back onto forecast
panels at the same grain to constrain downstream training, evaluation, and
readiness-control execution.

This module provides:
- Slice-key resolution helpers (slice_keys)
- Baseline-derived error anatomy aggregation by slice (compute_error_anatomy)
- Threshold-driven classification into admissibility classes (build_fas_surface)

All outputs are deterministic, audit-friendly, and include a stable thresholds
fingerprint and JSON-serialized thresholds used for classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

FASSliceMode = Literal["entity", "entity_interval", "site_entity_interval"]

_FAS_CLASS_ALLOWED: Final[str] = "ALLOWED"
_FAS_CLASS_CONDITIONAL: Final[str] = "CONDITIONAL"
_FAS_CLASS_BLOCKED: Final[str] = "BLOCKED"


@dataclass(frozen=True)
class FASThresholds:
    """Thresholds that define ALLOWED / CONDITIONAL / BLOCKED."""

    # Tail / spike anatomy (baseline-derived, symmetric AE)
    blocked_spike_rate_ge: float = 0.30
    blocked_p95_ae_ge: float = 25.0

    conditional_spike_rate_ge: float = 0.05
    conditional_p95_ae_ge: float = 10.0

    # Optional: shortfall (underbuild) anatomy thresholds.
    # Default to "off" by setting these to None.
    blocked_shortfall_spike_rate_ge: float | None = None
    blocked_p95_shortfall_ge: float | None = None

    conditional_shortfall_spike_rate_ge: float | None = None
    conditional_p95_shortfall_ge: float | None = None

    # Support guards
    min_valid_rows: int = 200  # minimum valid (non-NaN) slice rows required to classify
    min_nonzero_rows: int | None = None  # optional guard on nonzero-demand support


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def slice_keys(
    mode: FASSliceMode,
    *,
    site_col: str = "site_id",
    entity_col: str = "forecast_entity_id",
    interval_col: str = "INTERVAL_30_INDEX",
) -> list[str]:
    if mode == "entity":
        return [entity_col]
    if mode == "entity_interval":
        return [entity_col, interval_col]
    if mode == "site_entity_interval":
        return [site_col, entity_col, interval_col]
    raise ValueError(f"Unknown slice mode: {mode}")


def _safe_nanquantile(x: np.ndarray, q: float) -> float:
    """
    Robust quantile helper:
    - Returns NaN if no finite values exist.
    - Uses np.nanquantile on finite subset to avoid all-NaN edge cases.
    """
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanquantile(finite, q))


def compute_error_anatomy(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    yhat_col: str,
    keys: list[str],
    spike_ge: float = 10.0,
) -> pd.DataFrame:
    """
    Compute baseline-derived error anatomy by slice keys.

    Requires a baseline prediction column (yhat_col).

    Notes:
    - Rows with NaN in y or yhat are dropped before anatomy aggregation.
    - Returns both symmetric absolute-error anatomy and shortfall (underbuild)
      anatomy suitable for production-management contexts.
    - Includes a constant spike_ge column for auditability.
    """
    missing_cols = [c for c in [y_col, yhat_col, *keys] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in df: {missing_cols}")

    # Filter to valid rows for residual anatomy.
    work = df.loc[:, [*keys, y_col, yhat_col]].copy()
    work = work.dropna(subset=[y_col, yhat_col])

    # Residuals
    err = work[yhat_col] - work[y_col]
    abs_error = err.abs()

    # Underbuild / shortfall: y - yhat, floored at 0.
    shortfall = (work[y_col] - work[yhat_col]).clip(lower=0)

    # Nonzero demand indicator (for support diagnostics)
    is_nonzero = work[y_col].ne(0)

    out = (
        work.assign(
            abs_error=abs_error,
            shortfall=shortfall,
            is_nonzero=is_nonzero,
        )
        .groupby(keys, dropna=False)
        .agg(
            # Support
            n_valid=("abs_error", "size"),
            n_nonzero=("is_nonzero", "sum"),
            zero_rate=(y_col, lambda s: float((s == 0).mean())),
            # Symmetric AE anatomy
            spike_rate=("abs_error", lambda s: float((s >= spike_ge).mean())),
            p95_ae=("abs_error", lambda s: _safe_nanquantile(s.to_numpy(dtype=float), 0.95)),
            p90_ae=("abs_error", lambda s: _safe_nanquantile(s.to_numpy(dtype=float), 0.90)),
            mae=("abs_error", lambda s: float(np.nanmean(s.to_numpy(dtype=float)))),
            # Shortfall (underbuild) anatomy
            shortfall_rate=("shortfall", lambda s: float((s > 0).mean())),
            shortfall_spike_rate=("shortfall", lambda s: float((s >= spike_ge).mean())),
            p95_shortfall=("shortfall", lambda s: _safe_nanquantile(s.to_numpy(dtype=float), 0.95)),
            p90_shortfall=("shortfall", lambda s: _safe_nanquantile(s.to_numpy(dtype=float), 0.90)),
            mean_shortfall=("shortfall", lambda s: float(np.nanmean(s.to_numpy(dtype=float)))),
        )
        .reset_index()
    )

    # Record spike threshold used to compute spike rates for auditability.
    out["spike_ge"] = float(spike_ge)
    return out


def build_fas_surface(
    *,
    anatomy: pd.DataFrame,
    keys: list[str],
    thr: FASThresholds = FASThresholds(),
) -> pd.DataFrame:
    """
    Build the FAS surface from an error anatomy table.

    Minimum required columns:
      keys + ['n_valid','zero_rate','spike_rate','p95_ae']

    Additional columns (if present) are carried through to output.
    If 'spike_ge' exists in anatomy, it is included in the thresholds fingerprint
    + JSON payload for full auditability.
    """
    required = set(keys) | {"n_valid", "zero_rate", "spike_rate", "p95_ae"}
    missing = required - set(anatomy.columns)
    if missing:
        raise KeyError(f"Anatomy missing required columns: {sorted(missing)}")

    fas = anatomy.copy()

    # Support guards: conservative -> CONDITIONAL if insufficient.
    n_valid = fas["n_valid"].astype(float)
    support_ok = n_valid.ge(float(thr.min_valid_rows))

    if thr.min_nonzero_rows is not None:
        if "n_nonzero" not in fas.columns:
            raise KeyError(
                "FASThresholds.min_nonzero_rows is set, but anatomy is missing 'n_nonzero'."
            )
        support_ok = support_ok & fas["n_nonzero"].astype(float).ge(float(thr.min_nonzero_rows))

    # Any NaN in gating features should be treated conservatively as not support_ok.
    gate_cols = ["spike_rate", "p95_ae"]
    for c in gate_cols:
        support_ok = support_ok & fas[c].notna()

    # Base gating criteria: symmetric AE anatomy
    spike_rate = fas["spike_rate"].astype(float)
    p95_ae = fas["p95_ae"].astype(float)

    blocked_mask = (spike_rate.ge(thr.blocked_spike_rate_ge)) | (p95_ae.ge(thr.blocked_p95_ae_ge))
    conditional_mask = (spike_rate.ge(thr.conditional_spike_rate_ge)) | (
        p95_ae.ge(thr.conditional_p95_ae_ge)
    )

    # Optional: shortfall-based gating (only if thresholds are configured and columns exist).
    def _maybe_apply_shortfall_rules(
        blocked: pd.Series, conditional: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        if (
            thr.blocked_shortfall_spike_rate_ge is None
            and thr.blocked_p95_shortfall_ge is None
            and thr.conditional_shortfall_spike_rate_ge is None
            and thr.conditional_p95_shortfall_ge is None
        ):
            return blocked, conditional

        needed = {"shortfall_spike_rate", "p95_shortfall"}
        missing_shortfall = needed - set(fas.columns)
        if missing_shortfall:
            raise KeyError(
                "Shortfall thresholds configured in FASThresholds, but anatomy is missing "
                f"columns: {sorted(missing_shortfall)}"
            )

        s_spike = fas["shortfall_spike_rate"].astype(float)
        s_p95 = fas["p95_shortfall"].astype(float)

        # NaNs should behave conservatively when shortfall rules are enabled.
        shortfall_ok = fas["shortfall_spike_rate"].notna() & fas["p95_shortfall"].notna()

        b = blocked.copy()
        c = conditional.copy()

        if thr.blocked_shortfall_spike_rate_ge is not None:
            b = b | (shortfall_ok & s_spike.ge(thr.blocked_shortfall_spike_rate_ge))
        if thr.blocked_p95_shortfall_ge is not None:
            b = b | (shortfall_ok & s_p95.ge(thr.blocked_p95_shortfall_ge))

        if thr.conditional_shortfall_spike_rate_ge is not None:
            c = c | (shortfall_ok & s_spike.ge(thr.conditional_shortfall_spike_rate_ge))
        if thr.conditional_p95_shortfall_ge is not None:
            c = c | (shortfall_ok & s_p95.ge(thr.conditional_p95_shortfall_ge))

        return b, c

    blocked_mask, conditional_mask = _maybe_apply_shortfall_rules(blocked_mask, conditional_mask)

    # Apply support guard: if not support_ok => CONDITIONAL (conservative).
    # Otherwise apply BLOCKED > CONDITIONAL > ALLOWED precedence.
    fas_class = pd.Series(_FAS_CLASS_ALLOWED, index=fas.index, dtype="string")
    fas_class = fas_class.mask(conditional_mask & support_ok, _FAS_CLASS_CONDITIONAL)
    fas_class = fas_class.mask(blocked_mask & support_ok, _FAS_CLASS_BLOCKED)
    fas_class = fas_class.mask(~support_ok, _FAS_CLASS_CONDITIONAL)

    fas["fas_class"] = fas_class

    # Audit payload: thresholds + spike_ge (if present).
    payload: dict[str, Any] = {**thr.__dict__}
    if "spike_ge" in fas.columns:
        # If spike_ge differs across rows, store the sorted unique set.
        uniq = sorted({float(x) for x in fas["spike_ge"].dropna().unique().tolist()})
        payload["spike_ge"] = uniq[0] if len(uniq) == 1 else uniq

    fas["thr_fingerprint"] = _fingerprint_payload(payload)
    fas["thr_json"] = json.dumps(payload, sort_keys=True)

    fas["fas_allowed"] = fas["fas_class"].eq(_FAS_CLASS_ALLOWED)
    fas["fas_conditional"] = fas["fas_class"].eq(_FAS_CLASS_CONDITIONAL)
    fas["fas_blocked"] = fas["fas_class"].eq(_FAS_CLASS_BLOCKED)

    # Output columns: preserve original core + carry useful diagnostics if present.
    base_out_cols = [
        *keys,
        "fas_class",
        "fas_allowed",
        "fas_conditional",
        "fas_blocked",
        "n_valid",
        "zero_rate",
        "spike_rate",
        "p90_ae",
        "p95_ae",
        "mae",
        "thr_fingerprint",
        "thr_json",
    ]

    # Optional columns to include if available
    optional_cols = [
        "spike_ge",
        "n_nonzero",
        "shortfall_rate",
        "shortfall_spike_rate",
        "p90_shortfall",
        "p95_shortfall",
        "mean_shortfall",
    ]
    out_cols = base_out_cols + [c for c in optional_cols if c in fas.columns]

    # Ensure no duplicates and preserve order
    seen: set[str] = set()
    ordered_out_cols: list[str] = []
    for c in out_cols:
        if c not in seen:
            ordered_out_cols.append(c)
            seen.add(c)

    return fas.loc[:, ordered_out_cols].copy()

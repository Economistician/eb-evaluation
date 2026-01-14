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
from typing import Final, Literal

import numpy as np
import pandas as pd

FASSliceMode = Literal["entity", "entity_interval", "site_entity_interval"]

_FAS_CLASS_ALLOWED: Final[str] = "ALLOWED"
_FAS_CLASS_CONDITIONAL: Final[str] = "CONDITIONAL"
_FAS_CLASS_BLOCKED: Final[str] = "BLOCKED"


@dataclass(frozen=True)
class FASThresholds:
    """Thresholds that define ALLOWED / CONDITIONAL / BLOCKED."""

    # Tail / spike anatomy (baseline-derived)
    blocked_spike_rate_ge: float = 0.30
    blocked_p95_ae_ge: float = 25.0

    conditional_spike_rate_ge: float = 0.05
    conditional_p95_ae_ge: float = 10.0

    # Support guards
    min_rows: int = 200  # minimum slice rows required to classify


def _thresholds_fingerprint(thr: FASThresholds) -> str:
    payload = json.dumps(thr.__dict__, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


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
    """
    if yhat_col not in df.columns:
        raise KeyError(f"Missing yhat_col={yhat_col!r} in df.")

    ae = (df[yhat_col] - df[y_col]).abs()
    out = (
        df.assign(abs_error=ae)
        .groupby(keys, dropna=False)
        .agg(
            n=("abs_error", "size"),
            zero_rate=(y_col, lambda s: (s == 0).mean()),
            spike_rate=("abs_error", lambda s: (s >= spike_ge).mean()),
            p95_ae=("abs_error", lambda s: float(np.nanquantile(s.to_numpy(), 0.95))),
            p90_ae=("abs_error", lambda s: float(np.nanquantile(s.to_numpy(), 0.90))),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    return out


def classify_fas_row(row: pd.Series, thr: FASThresholds) -> str:
    """Deterministic rule closure."""
    if float(row["n"]) < float(thr.min_rows):
        # Conservative: if we can't trust the slice, treat as conditional.
        return _FAS_CLASS_CONDITIONAL

    if (float(row["spike_rate"]) >= thr.blocked_spike_rate_ge) or (
        float(row["p95_ae"]) >= thr.blocked_p95_ae_ge
    ):
        return _FAS_CLASS_BLOCKED

    if (float(row["spike_rate"]) >= thr.conditional_spike_rate_ge) or (
        float(row["p95_ae"]) >= thr.conditional_p95_ae_ge
    ):
        return _FAS_CLASS_CONDITIONAL

    return _FAS_CLASS_ALLOWED


def build_fas_surface(
    *,
    anatomy: pd.DataFrame,
    keys: list[str],
    thr: FASThresholds = FASThresholds(),
) -> pd.DataFrame:
    """
    Build the FAS surface from an error anatomy table.

    anatomy must include: keys + ['n','zero_rate','spike_rate','p95_ae'] at minimum.
    """
    required = set(keys) | {"n", "zero_rate", "spike_rate", "p95_ae"}
    missing = required - set(anatomy.columns)
    if missing:
        raise KeyError(f"Anatomy missing required columns: {sorted(missing)}")

    fas = anatomy.copy()

    # Avoid DataFrame.apply(...) ambiguity for type checkers by returning a typed Series.
    fas_class = pd.Series(
        (classify_fas_row(r, thr) for _, r in fas.iterrows()),
        index=fas.index,
        dtype="string",
    )
    fas["fas_class"] = fas_class

    fas["thr_fingerprint"] = _thresholds_fingerprint(thr)
    fas["thr_json"] = json.dumps(thr.__dict__, sort_keys=True)

    fas["fas_allowed"] = fas["fas_class"].eq(_FAS_CLASS_ALLOWED)
    fas["fas_conditional"] = fas["fas_class"].eq(_FAS_CLASS_CONDITIONAL)
    fas["fas_blocked"] = fas["fas_class"].eq(_FAS_CLASS_BLOCKED)

    out_cols = [
        *keys,
        "fas_class",
        "fas_allowed",
        "fas_conditional",
        "fas_blocked",
        "n",
        "zero_rate",
        "spike_rate",
        "p90_ae",
        "p95_ae",
        "mae",
        "thr_fingerprint",
        "thr_json",
    ]
    return fas.loc[:, out_cols].copy()

"""Forecast Admissibility Surface (FAS).

Build a deterministic ALLOWED / CONDITIONAL / BLOCKED surface by slice grain from
demand structure and baseline error anatomy. Helpers: ``slice_keys``,
``compute_error_anatomy``, ``build_fas_surface``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd

FASSliceMode = Literal["entity", "entity_interval", "site_entity_interval"]


class FASClass(StrEnum):
    """Admissibility class for a Forecast Admissibility Surface slice."""

    ALLOWED = "ALLOWED"
    CONDITIONAL = "CONDITIONAL"
    BLOCKED = "BLOCKED"


_FAS_CLASS_ALLOWED: Final[str] = FASClass.ALLOWED.value
_FAS_CLASS_CONDITIONAL: Final[str] = FASClass.CONDITIONAL.value
_FAS_CLASS_BLOCKED: Final[str] = FASClass.BLOCKED.value
_FAS_FINGERPRINT_HEX_LEN: Final[int] = 16
MIXED_FAS_CLASS_TOKEN: Final[str] = "MIXED_FAS_FAIL_CLOSED"


def panel_fas_class_is_mixed(stream_fas: FASClass | str | None) -> bool:
    """Return True when ``resolve_panel_fas_class`` found conflicting tokens."""
    return stream_fas == MIXED_FAS_CLASS_TOKEN


def resolve_panel_fas_class(
    group: pd.DataFrame,
    *,
    fas_class: FASClass | str | pd.Series | None = None,
    fas_class_col: str | None = None,
) -> FASClass | str | None:
    """Resolve one FAS class for a grouped stream.

    Preference: ``fas_class_col`` on ``group``, then a row-aligned ``Series``,
    then a scalar broadcast. Mixed values within a stream return
    ``MIXED_FAS_CLASS_TOKEN`` so the caller can fail-close that stream without
    aborting the panel.
    """
    if fas_class_col is not None:
        if fas_class_col not in group.columns:
            raise ValueError(f"fas_class_col {fas_class_col!r} is missing from the panel.")
        col = group.loc[:, fas_class_col]
        if not isinstance(col, pd.Series):
            raise ValueError(f"fas_class_col {fas_class_col!r} must select a single column.")
        uniq = pd.unique(col.dropna())
        if len(uniq) == 0:
            return None
        if len(uniq) > 1:
            return MIXED_FAS_CLASS_TOKEN
        value = uniq[0]
        return value if isinstance(value, FASClass | str) else str(value)
    if isinstance(fas_class, pd.Series):
        aligned = fas_class.reindex(group.index)
        uniq = pd.unique(aligned.dropna())
        if len(uniq) == 0:
            return None
        if len(uniq) > 1:
            return MIXED_FAS_CLASS_TOKEN
        value = uniq[0]
        return value if isinstance(value, FASClass | str) else str(value)
    return fas_class


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
    return hashlib.sha256(encoded).hexdigest()[:_FAS_FINGERPRINT_HEX_LEN]


def _payload_spike_ge(fas: pd.DataFrame) -> float | list[float] | None:
    """Serialize anatomy ``spike_ge`` into a stable fingerprint field.

    The key is always present in the payload. Missing or all-NaN values become
    ``None``; a single unique value is stored as a float; mixed values are a
    sorted list.
    """
    if "spike_ge" not in fas.columns:
        return None
    uniq = sorted({float(x) for x in fas["spike_ge"].dropna().unique().tolist()})
    if len(uniq) == 0:
        return None
    if len(uniq) == 1:
        return uniq[0]
    return uniq


def _finite_gt_rate(x: np.ndarray, threshold: float) -> float:
    """Share of finite values strictly greater than ``threshold``; NaN if none."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float((finite > threshold).mean())


def _safe_nanmean(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def slice_keys(
    mode: FASSliceMode,
    *,
    site_col: str = "site_id",
    entity_col: str = "forecast_entity_id",
    interval_col: str = "INTERVAL_INDEX",
) -> list[str]:
    if mode == "entity":
        return [entity_col]
    if mode == "entity_interval":
        return [entity_col, interval_col]
    if mode == "site_entity_interval":
        return [site_col, entity_col, interval_col]
    raise ValueError(f"Unknown slice mode: {mode}")


def valid_slice_modes() -> tuple[str, ...]:
    """
    Return the supported slice mode strings.

    This exists to make slice-mode discovery programmatic and stable for callers.
    """
    return ("entity", "entity_interval", "site_entity_interval")


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


def _segmented_finite_gt_rate(
    values: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    threshold: float,
) -> np.ndarray:
    """Share of finite values strictly greater than ``threshold``; NaN if none."""
    out = np.full(n_groups, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    n_finite = np.bincount(labels[finite], minlength=n_groups)
    gt = finite & (values > threshold)
    n_gt = np.bincount(labels[gt], minlength=n_groups)
    ok = n_finite > 0
    out[ok] = n_gt[ok] / n_finite[ok]
    return out


def _segmented_nanmean(values: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean of finite values per group; NaN if a group has none."""
    out = np.full(n_groups, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    n_finite = np.bincount(labels[finite], minlength=n_groups)
    sums = np.zeros(n_groups, dtype=np.float64)
    np.add.at(sums, labels[finite], values[finite])
    ok = n_finite > 0
    out[ok] = sums[ok] / n_finite[ok]
    return out


def _segmented_nanquantile(
    values: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    q: float,
) -> np.ndarray:
    """Per-group ``np.nanquantile`` (linear) on finite values via a sorted layout."""
    out = np.full(n_groups, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    v = values[finite]
    lab = labels[finite]
    order = np.lexsort((v, lab))
    v = np.ascontiguousarray(v[order])
    lab = lab[order]
    counts = np.bincount(lab, minlength=n_groups)
    starts = np.empty(n_groups, dtype=np.intp)
    starts[0] = 0
    if n_groups > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    nonempty = np.flatnonzero(counts)
    for g in nonempty:
        start = int(starts[g])
        stop = start + int(counts[g])
        out[g] = float(np.quantile(v[start:stop], q, method="linear"))
    return out


def _require_unique_columns(df: pd.DataFrame, *, context: str) -> None:
    """
    Guardrail: duplicate column labels can cause df[col] to return a DataFrame,
    which then breaks arithmetic and aggregation in confusing ways.
    """
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{context}: df.columns must be unique; found duplicate columns: {dupes}")


def _require_unique_keys(keys: list[str], *, context: str) -> None:
    """
    Guardrail: duplicate keys in the slice-key list can cause ambiguous groupby
    behavior and subtle join bugs.
    """
    seen: set[str] = set()
    dupes: list[str] = []
    for k in keys:
        if k in seen and k not in dupes:
            dupes.append(k)
        seen.add(k)
    if dupes:
        raise ValueError(f"{context}: keys must be unique; found duplicates: {dupes}")


def _coerce_numeric_series(df: pd.DataFrame, col: str, *, context: str) -> pd.Series:
    """
    Coerce a dataframe column to numeric, raising a clear error if the column
    is not a Series (e.g., due to duplicate column names) or if coercion fails.
    """
    s = df[col]
    if not isinstance(s, pd.Series):
        raise ValueError(
            f"{context}: expected df['{col}'] to be a Series, but got {type(s).__name__}. "
            "This often happens when the dataframe has duplicate column names."
        )
    # pandas typing can return an "array-like" union in stubs; at runtime it's a Series.
    return cast(pd.Series, pd.to_numeric(s, errors="coerce"))


def _coerce_key_series(df: pd.DataFrame, col: str, *, context: str) -> pd.Series:
    """
    Coerce a slice key column into a stable, join-friendly dtype.

    We intentionally coerce keys to pandas' StringDtype to prevent mixed-type
    key columns (e.g., ints + strs) from causing downstream join/sort failures.
    This also makes artifacts more portable across CSV/Parquet/Snowflake paths.
    """
    s = df[col]
    if not isinstance(s, pd.Series):
        raise ValueError(
            f"{context}: expected df['{col}'] to be a Series, but got {type(s).__name__}. "
            "This often happens when the dataframe has duplicate column names."
        )
    return s.astype("string")


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
    - Rows with NaN in y or yhat are excluded from anatomy statistics.
    - Slices that remain with zero valid rows are preserved (n_valid=0).
    - Spike rate uses strict inequality (abs_error > spike_ge).
    - Slice keys are coerced to StringDtype for stability and joinability.
    - Returns both symmetric absolute-error anatomy and shortfall (underbuild)
      anatomy suitable for production-management contexts.
    - Includes a constant spike_ge column for auditability.
    """
    _require_unique_keys(keys, context="compute_error_anatomy")

    missing_cols = [c for c in [y_col, yhat_col, *keys] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in df: {missing_cols}")

    _require_unique_columns(df, context="compute_error_anatomy")

    # Filter to required columns first.
    work = df.loc[:, [*keys, y_col, yhat_col]].copy()

    # Coerce slice keys defensively to avoid mixed-type key bugs downstream.
    for k in keys:
        work[k] = _coerce_key_series(work, k, context="compute_error_anatomy")

    # Coerce numeric defensively; non-numeric becomes NaN. Keep every slice key
    # so zero-valid groups remain in the anatomy table (Domain exhaustiveness).
    work[y_col] = _coerce_numeric_series(work, y_col, context="compute_error_anatomy")
    work[yhat_col] = _coerce_numeric_series(work, yhat_col, context="compute_error_anatomy")

    y = work[y_col].to_numpy(dtype=float, copy=False)
    yhat = work[yhat_col].to_numpy(dtype=float, copy=False)
    is_valid = (~np.isnan(y)) & (~np.isnan(yhat))
    abs_error = np.abs(yhat - y)
    shortfall = np.maximum(y - yhat, 0.0)
    is_nonzero = is_valid & (y != 0)
    is_zero = is_valid & (y == 0)

    grouped = work.groupby(keys, dropna=False, sort=True)
    labels = grouped.ngroup().to_numpy(dtype=np.intp)
    key_frame = grouped.size().reset_index().loc[:, keys]
    n_groups = int(labels.max()) + 1 if labels.size else 0

    if n_groups == 0:
        n_valid = np.array([], dtype=np.int64)
        n_nonzero = np.array([], dtype=np.int64)
        n_zero = np.array([], dtype=np.int64)
        spike_rate = np.array([], dtype=np.float64)
        p95_ae = np.array([], dtype=np.float64)
        p90_ae = np.array([], dtype=np.float64)
        mae = np.array([], dtype=np.float64)
        shortfall_rate = np.array([], dtype=np.float64)
        shortfall_spike_rate = np.array([], dtype=np.float64)
        p95_shortfall = np.array([], dtype=np.float64)
        p90_shortfall = np.array([], dtype=np.float64)
        mean_shortfall = np.array([], dtype=np.float64)
    else:
        n_valid = np.bincount(labels[is_valid], minlength=n_groups).astype(np.int64, copy=False)
        n_nonzero = np.bincount(labels[is_nonzero], minlength=n_groups).astype(np.int64, copy=False)
        n_zero = np.bincount(labels[is_zero], minlength=n_groups).astype(np.int64, copy=False)
        spike_rate = _segmented_finite_gt_rate(abs_error, labels, n_groups, spike_ge)
        p95_ae = _segmented_nanquantile(abs_error, labels, n_groups, 0.95)
        p90_ae = _segmented_nanquantile(abs_error, labels, n_groups, 0.90)
        mae = _segmented_nanmean(abs_error, labels, n_groups)
        shortfall_rate = _segmented_finite_gt_rate(shortfall, labels, n_groups, 0.0)
        shortfall_spike_rate = _segmented_finite_gt_rate(shortfall, labels, n_groups, spike_ge)
        p95_shortfall = _segmented_nanquantile(shortfall, labels, n_groups, 0.95)
        p90_shortfall = _segmented_nanquantile(shortfall, labels, n_groups, 0.90)
        mean_shortfall = _segmented_nanmean(shortfall, labels, n_groups)

    out = key_frame.copy()
    out["n_valid"] = n_valid
    out["n_nonzero"] = n_nonzero
    out["n_zero"] = n_zero
    out["spike_rate"] = spike_rate
    out["p95_ae"] = p95_ae
    out["p90_ae"] = p90_ae
    out["mae"] = mae
    out["shortfall_rate"] = shortfall_rate
    out["shortfall_spike_rate"] = shortfall_spike_rate
    out["p95_shortfall"] = p95_shortfall
    out["p90_shortfall"] = p90_shortfall
    out["mean_shortfall"] = mean_shortfall

    n_valid_f = out["n_valid"].astype(float)
    out["zero_rate"] = out["n_zero"].astype(float).div(n_valid_f).where(n_valid_f.gt(0.0))
    out = out.drop(columns=["n_zero"])

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
    The thresholds fingerprint always includes ``spike_ge`` (``None`` when the
    anatomy column is absent or all-NaN).
    """
    _require_unique_keys(keys, context="build_fas_surface")

    required = set(keys) | {"n_valid", "zero_rate", "spike_rate", "p95_ae"}
    missing = required - set(anatomy.columns)
    if missing:
        raise KeyError(f"Anatomy missing required columns: {sorted(missing)}")

    _require_unique_columns(anatomy, context="build_fas_surface")

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

    # Canonical status column for downstream use (stable, ergonomic).
    fas["fas_status"] = fas["fas_class"]

    # Audit payload: thresholds + spike_ge (always present for a stable schema).
    payload: dict[str, Any] = {**thr.__dict__, "spike_ge": _payload_spike_ge(fas)}

    fas["thr_fingerprint"] = _fingerprint_payload(payload)
    fas["thr_json"] = json.dumps(payload, sort_keys=True)

    fas["fas_allowed"] = fas["fas_class"].eq(_FAS_CLASS_ALLOWED)
    fas["fas_conditional"] = fas["fas_class"].eq(_FAS_CLASS_CONDITIONAL)
    fas["fas_blocked"] = fas["fas_class"].eq(_FAS_CLASS_BLOCKED)

    # Output columns: preserve original core + carry useful diagnostics if present.
    # NOTE: Ensure stable schema even if anatomy omitted some optional diagnostics columns.
    base_out_cols = [
        *keys,
        "fas_class",
        "fas_status",
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

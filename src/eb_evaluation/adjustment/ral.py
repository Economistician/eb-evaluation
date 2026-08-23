"""
Readiness Adjustment Layer (RAL): deterministic fit + apply in eb-evaluation.

This module implements a transparent post-processing step that converts a baseline forecast
into an operationally conservative readiness forecast via a learned uplift.

Responsibilities
---------------
- Fit a simple uplift policy via grid search that minimizes CWSL.
- Apply learned uplift factors to new data (global or segmented).
- Provide before/after diagnostics for auditability.
- Provide a canonical apply_ral(...) utility that joins governance decisions and applies
  governed post-processing (nonneg + snap-to-grid) to prediction columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd

from eb_evaluation.diagnostics.governance import snap_to_grid
from eb_evaluation.diagnostics.presets import preset_policy
from eb_metrics.metrics import cwsl


class ReadinessAdjustmentLayer:
    """Readiness Adjustment Layer (RAL) for operational forecast uplift."""

    def __init__(
        self,
        *,
        cu: float | None = None,
        co: float | None = None,
        uplift_min: float = 1.0,
        uplift_max: float = 1.2,
        grid_step: float = 0.01,
    ) -> None:
        self.cu = None if cu is None else float(cu)
        self.co = None if co is None else float(co)

        self.uplift_min = float(uplift_min)
        self.uplift_max = float(uplift_max)
        self.grid_step = float(grid_step)

        # Learned artifacts (set by fit)
        self.global_uplift_: float | None = None
        self.segment_cols_: list[str] = []
        self.uplift_table_: pd.DataFrame | None = None
        self.diagnostics_: pd.DataFrame = pd.DataFrame()

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _require_costs(self) -> tuple[float, float]:
        if self.cu is None or self.co is None:
            raise TypeError("ReadinessAdjustmentLayer requires cu and co to be set (via __init__).")
        return float(self.cu), float(self.co)

    def _grid(self) -> np.ndarray:
        """Construct an uplift grid that reliably includes uplift_max."""
        if self.grid_step <= 0:
            raise ValueError("grid_step must be > 0.")
        if self.uplift_max < self.uplift_min:
            raise ValueError("uplift_max must be >= uplift_min.")

        # Use arange then explicitly ensure endpoint inclusion.
        grid = np.arange(
            self.uplift_min,
            self.uplift_max + (self.grid_step / 2.0),
            self.grid_step,
            dtype=float,
        )
        grid = np.clip(grid, self.uplift_min, self.uplift_max)

        # Ensure uplift_max is present (avoid floating step drift)
        if not np.isclose(grid[-1], self.uplift_max, rtol=0.0, atol=1e-12):
            grid = np.append(grid, float(self.uplift_max))
        else:
            grid[-1] = float(self.uplift_max)

        # De-dup and sort (append could create duplicates in rare cases)
        grid = np.unique(grid)
        return grid

    def _best_uplift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        cu: float,
        co: float,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Return (best_uplift, cwsl_before, cwsl_after)."""
        y_true = y_true.astype(float, copy=False)
        y_pred = y_pred.astype(float, copy=False)

        before = float(
            cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co, sample_weight=sample_weight)
        )

        grid = self._grid()
        best_u = float(grid[0])
        best_loss = float("inf")

        for u in grid:
            loss = float(
                cwsl(
                    y_true=y_true,
                    y_pred=y_pred * float(u),
                    cu=cu,
                    co=co,
                    sample_weight=sample_weight,
                )
            )
            # Tie-break: prefer the smaller uplift
            if (loss < best_loss) or (abs(loss - best_loss) < 1e-12 and float(u) < best_u):
                best_loss = loss
                best_u = float(u)

        return best_u, before, best_loss

    # ----------------------------
    # Public API
    # ----------------------------
    def fit(
        self,
        df: pd.DataFrame,
        *,
        forecast_col: str,
        actual_col: str,
        segment_cols: Sequence[str] | None = None,
        sample_weight_col: str | None = None,
    ) -> ReadinessAdjustmentLayer:
        cu, co = self._require_costs()

        required = [forecast_col, actual_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for fit(): {missing}")

        if sample_weight_col is not None and sample_weight_col not in df.columns:
            raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in DataFrame.")

        seg_cols = list(segment_cols) if segment_cols is not None else []
        if seg_cols:
            seg_missing = [c for c in seg_cols if c not in df.columns]
            if seg_missing:
                raise KeyError(f"Missing segment columns for fit(): {seg_missing}")

        y_true_all = df[actual_col].to_numpy(dtype=float)
        y_pred_all = df[forecast_col].to_numpy(dtype=float)
        w_all = df[sample_weight_col].to_numpy(dtype=float) if sample_weight_col else None

        # Fit global uplift on full data (used as fallback)
        global_u, g_before, g_after = self._best_uplift(
            y_true_all,
            y_pred_all,
            cu=cu,
            co=co,
            sample_weight=w_all,
        )
        self.global_uplift_ = float(global_u)

        diagnostics_rows: list[dict] = [
            {
                "scope": "global",
                "uplift": float(global_u),
                "cwsl_before": float(g_before),
                "cwsl_after": float(g_after),
            }
        ]

        # Fit per-segment uplifts if requested
        self.segment_cols_ = seg_cols
        self.uplift_table_ = None

        if seg_cols:
            table_rows: list[dict] = []

            grouped = df.groupby(seg_cols, dropna=False, sort=False)
            for key, g in grouped:
                key_vals = (key,) if not isinstance(key, tuple) else key

                y_true = g[actual_col].to_numpy(dtype=float)
                y_pred = g[forecast_col].to_numpy(dtype=float)
                w = g[sample_weight_col].to_numpy(dtype=float) if sample_weight_col else None

                best_u, before, after = self._best_uplift(
                    y_true,
                    y_pred,
                    cu=cu,
                    co=co,
                    sample_weight=w,
                )

                best_u = float(best_u)

                row = dict(zip(seg_cols, key_vals, strict=False))
                row["uplift"] = best_u
                table_rows.append(row)

                diag = dict(zip(seg_cols, key_vals, strict=False))
                diag.update(
                    {
                        "scope": "segment",
                        "uplift": best_u,
                        "cwsl_before": float(before),
                        "cwsl_after": float(after),
                    }
                )
                diagnostics_rows.append(diag)

            self.uplift_table_ = pd.DataFrame(table_rows)

        self.diagnostics_ = pd.DataFrame(diagnostics_rows)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        forecast_col: str,
        output_col: str = "readiness_forecast",
        segment_cols: Sequence[str] | None = None,
        decisions: pd.DataFrame | None = None,
        key_cols: Sequence[str] | None = None,
        apply_mask: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Apply learned uplift factors to produce readiness forecasts.

        This is not a production writer. Callers must pass a joined governance
        ``decisions`` table or an explicit ``apply_mask``. Ungated calls raise
        so ``apply_ral`` remains the sole fail-closed apply path.

        If called before explicit fit(), this can still work for global uplift by
        implicitly fitting on the provided dataframe (requires an actual column),
        but only when costs (cu/co) are set.
        """
        if decisions is None and apply_mask is None:
            raise ValueError(
                "ReadinessAdjustmentLayer.transform requires a joined governance "
                "decisions table or apply_mask; ungated transform is not a "
                "production writer. Use apply_ral as the sole fail-closed apply path."
            )

        if forecast_col not in df.columns:
            raise KeyError(f"Column {forecast_col!r} not found in DataFrame.")

        # If not fit yet, optionally implicitly fit globally (ONLY when cu/co are set).
        if self.global_uplift_ is None or self.diagnostics_.empty:
            if self.cu is None or self.co is None:
                # Required by test_transform_raises_if_not_fit
                raise RuntimeError("ReadinessAdjustmentLayer must be fit() before transform().")

            if "actual" not in df.columns:
                raise RuntimeError(
                    "ReadinessAdjustmentLayer must be fit() before transform(), "
                    "or provide an 'actual' column to allow implicit global fit."
                )

            self.fit(df, forecast_col=forecast_col, actual_col="actual", segment_cols=None)

        # Narrow Optional[float] for type-checkers (runtime behavior unchanged).
        global_uplift = self.global_uplift_
        if global_uplift is None:
            raise RuntimeError("ReadinessAdjustmentLayer must be fit() before transform().")

        seg_cols = list(segment_cols) if segment_cols is not None else list(self.segment_cols_)
        result_df = df.copy()

        if seg_cols and self.uplift_table_ is not None and not self.uplift_table_.empty:
            missing = [c for c in seg_cols if c not in result_df.columns]
            if missing:
                raise KeyError(
                    f"Missing segment columns for transform(): {missing}. "
                    f"Available columns: {list(result_df.columns)}"
                )

            merged = result_df.merge(self.uplift_table_, on=seg_cols, how="left")
            uplift = np.array(merged["uplift"].to_numpy(dtype=float), copy=True)

            mask_nan = ~np.isfinite(uplift)
            if mask_nan.any():
                uplift[mask_nan] = float(global_uplift)
        else:
            uplift = np.full(len(result_df), float(global_uplift), dtype=float)

        baseline = result_df[forecast_col].to_numpy(dtype=float)
        adjusted = baseline * uplift
        authorized = _transform_authorization_mask(
            result_df,
            decisions=decisions,
            key_cols=key_cols,
            apply_mask=apply_mask,
        )
        result_df[output_col] = np.where(authorized, adjusted, baseline)
        return result_df


NonnegPolicy = Literal["allow", "clip_zero"]
SnapMode = Literal["ceil", "round", "floor"]

# apply_ral(nonneg_mode="none") and missing recommendation payloads follow the
# balanced preset, matching run_governance_gate.
_DEFAULT_APPLY_NONNEG: NonnegPolicy = preset_policy("balanced")
_SNAP_DQC_CLASSES = frozenset({"quantized", "piecewise_packed"})
_SNAP_MODES = frozenset({"ceil", "round", "floor"})
_BLOCKED_RAL_POLICIES = ("disallow",)
_BLOCKED_STATUSES = ("red",)
_BLOCKED_FAS_CLASSES = ("blocked",)
_BLOCKED_DQC_CLASSES = ("unknown",)
_RECOMMENDATION_SEP = ","
REQUIRED_DECISION_COLUMNS = (
    "ral_policy",
    "status",
    "fas_class",
    "dqc_class",
    "snap_required",
)


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (ValueError, TypeError):
        return False
    if isinstance(result, bool | np.bool_):
        return bool(result)
    return False


def _tokenize_recommendations(recs: object | None) -> tuple[str, ...]:
    """Split gate recommendation payloads into tokens.

    Panel writers store recommendations as a comma-joined string. Gate objects
    store a sequence of strings. Either form is accepted. Tokens are stripped
    and empty pieces dropped so parsers never iterate raw characters.
    """
    if _is_missing_scalar(recs):
        return ()
    if isinstance(recs, str):
        return tuple(part.strip() for part in recs.split(_RECOMMENDATION_SEP) if part.strip())
    if isinstance(recs, bytes | bytearray):
        return _tokenize_recommendations(recs.decode("utf-8", errors="replace"))
    if isinstance(recs, Sequence) and not isinstance(recs, str | bytes | bytearray):
        tokens: list[str] = []
        for item in recs:
            tokens.extend(_tokenize_recommendations(item))
        return tuple(tokens)
    return _tokenize_recommendations(str(recs))


def _nonneg_from_token(token: str) -> NonnegPolicy | None:
    if token.startswith("forecast_postprocess_nonneg(mode=") and token.endswith(")"):
        mode = token[len("forecast_postprocess_nonneg(mode=") : -1]
    elif token.startswith("nonneg_mode="):
        mode = token.split("=", 1)[1].strip()
    else:
        return None
    if mode == "allow":
        return "allow"
    if mode in ("clip", "clip_zero"):
        return "clip_zero"
    return None


def _snap_mode_from_token(token: str) -> SnapMode | None:
    if token.startswith("snap_forecasts_to_grid(mode=") and token.endswith(")"):
        mode = token[len("snap_forecasts_to_grid(mode=") : -1]
    elif token.startswith("snap_mode="):
        mode = token.split("=", 1)[1].strip()
    else:
        return None
    if mode in _SNAP_MODES:
        return mode  # type: ignore[return-value]
    return None


def _infer_nonneg_policy_from_recommendations(recs: object | None) -> NonnegPolicy:
    """Resolve apply-path nonnegativity from gate recommendation payload.

    Missing or unrecognized payloads fail closed to the balanced preset
    (``clip_zero``). Explicit ``mode=allow`` is honored.
    """
    tokens = _tokenize_recommendations(recs)
    found: NonnegPolicy | None = None
    for token in tokens:
        parsed = _nonneg_from_token(token)
        if parsed == "allow":
            return "allow"
        if parsed is not None:
            found = parsed
    return found if found is not None else _DEFAULT_APPLY_NONNEG


def _infer_snap_mode_from_recommendations(recs: object | None, *, default: SnapMode) -> SnapMode:
    tokens = _tokenize_recommendations(recs)
    for token in tokens:
        parsed = _snap_mode_from_token(token)
        if parsed is not None:
            return parsed
    return default


def _dqc_implies_snap(value: object) -> bool:
    if _is_missing_scalar(value):
        return False
    token = str(value).strip().lower().rsplit(".", 1)[-1]
    return token in _SNAP_DQC_CLASSES


def _enum_token(value: object) -> str:
    """Normalize enum/string policy tokens for fail-closed apply checks."""
    if _is_missing_scalar(value):
        return ""
    raw = getattr(value, "value", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    return str(value).strip().lower().rsplit(".", 1)[-1]


def _adjustment_blocked_mask(work: pd.DataFrame) -> pd.Series:
    """True where RAL must not be applied (DISALLOW, RED, FAS BLOCKED, or DQC UNKNOWN)."""
    blocked = pd.Series(False, index=work.index)
    if "ral_policy" in work.columns:
        tokens = work["ral_policy"].map(_enum_token)
        blocked = blocked | tokens.isin(_BLOCKED_RAL_POLICIES) | (tokens == "")
    if "status" in work.columns:
        tokens = work["status"].map(_enum_token)
        blocked = blocked | tokens.isin(_BLOCKED_STATUSES) | (tokens == "")
    if "fas_class" in work.columns:
        blocked = blocked | work["fas_class"].map(_enum_token).isin(_BLOCKED_FAS_CLASSES)
    if "dqc_class" in work.columns:
        tokens = work["dqc_class"].map(_enum_token)
        blocked = blocked | tokens.isin(_BLOCKED_DQC_CLASSES) | (tokens == "")
    return blocked.astype(bool)


def _require_decisions_table(decisions: pd.DataFrame | None) -> pd.DataFrame:
    """Refuse ungated apply_ral; callers must run the governance gate first."""
    if decisions is None:
        raise ValueError(
            "apply_ral requires a non-null governance decisions table with "
            "columns ral_policy, status, fas_class, dqc_class, and snap_required. "
            "Run evaluate_governance_panel_df or run_governance_gate first."
        )
    missing = [c for c in REQUIRED_DECISION_COLUMNS if c not in decisions.columns]
    if missing:
        raise ValueError(
            "apply_ral decisions are missing required governance columns: "
            f"{missing}. Run evaluate_governance_panel_df or run_governance_gate first."
        )
    return decisions


def _transform_authorization_mask(
    frame: pd.DataFrame,
    *,
    decisions: pd.DataFrame | None,
    key_cols: Sequence[str] | None,
    apply_mask: pd.Series | None,
) -> np.ndarray:
    """Authorize RAL.transform rows from a decisions join and/or explicit mask."""
    authorized = np.ones(len(frame), dtype=bool)
    if decisions is not None:
        keys = list(key_cols) if key_cols is not None else []
        if len(keys) == 0:
            raise ValueError("`key_cols` is required when `decisions` is provided.")
        missing_frame = [c for c in keys if c not in frame.columns]
        missing_decisions = [c for c in keys if c not in decisions.columns]
        if missing_frame or missing_decisions:
            raise ValueError(
                "Missing join keys for transform decisions: "
                f"frame={missing_frame}, decisions={missing_decisions}."
            )
        joined = frame.loc[:, keys].merge(decisions, on=keys, how="left")
        joined.index = frame.index
        blocked = _adjustment_blocked_mask(joined)
        policy_obj = joined["ral_policy"] if "ral_policy" in joined.columns else None
        if policy_obj is None:
            missing_join = pd.Series(True, index=joined.index)
        elif isinstance(policy_obj, pd.Series):
            missing_join = policy_obj.isna()
        else:
            missing_join = pd.Series(pd.isna(policy_obj), index=joined.index)
        authorized = authorized & ~(blocked | missing_join).to_numpy(dtype=bool)
    if apply_mask is not None:
        aligned = apply_mask.reindex(frame.index)
        if bool(aligned.isna().any()):
            raise ValueError("apply_mask must cover every transform row.")
        authorized = authorized & aligned.to_numpy(dtype=bool)
    return authorized


def _series_from_column(work: pd.DataFrame, col: str) -> pd.Series:
    obj = work[col]
    if isinstance(obj, pd.Series):
        return obj
    return pd.Series(obj, index=work.index, name=col)


def _apply_nonneg_policy(values: np.ndarray, *, policy: NonnegPolicy) -> np.ndarray:
    if policy == "allow":
        return values
    return np.maximum(values, 0.0)


def _apply_snap_policy_series(
    values: pd.Series,
    units: pd.Series,
    *,
    mode: SnapMode,
) -> pd.Series:
    """
    Apply snap-to-grid with per-row (possibly varying) units.

    If the unit is constant (typical), we snap in one pass. If it varies, we
    snap per-unit group to avoid row-wise Python loops.
    """
    v = values.to_numpy(dtype=float, copy=False)
    u = units.to_numpy(dtype=float, copy=False)

    mask = np.isfinite(v) & np.isfinite(u) & (u > 0.0)
    if not mask.any():
        return values

    uniq_units = np.unique(u[mask])
    if uniq_units.size == 1:
        unit = float(uniq_units[0])
        snapped = np.asarray(snap_to_grid(v.tolist(), unit, mode=mode), dtype=float)
        out = v.copy()
        out[:] = snapped
        return pd.Series(out, index=values.index, name=values.name)

    out = v.copy()
    for unit in uniq_units.tolist():
        unit_f = float(unit)
        idx = mask & (u == unit_f)
        snapped_sub = np.asarray(snap_to_grid(v[idx].tolist(), unit_f, mode=mode), dtype=float)
        out[idx] = snapped_sub
    return pd.Series(out, index=values.index, name=values.name)


def apply_ral(
    df: pd.DataFrame,
    *,
    # --- legacy alias kwargs (kept for compatibility with older tests/callers) ---
    join_keys: Sequence[str] | None = None,
    pred_col: str | None = None,
    output_col: str | None = None,
    nonneg_mode: str | None = None,
    # --- canonical inputs ---
    decisions: pd.DataFrame | None = None,
    key_cols: Sequence[str] = ("forecast_entity_id",),
    yhat_base_col: str = "yhat_base",
    yhat_ral_col: str | None = "yhat_ral",
    uplift_col: str | None = None,
    snap_required_col: str = "snap_required",
    snap_unit_col: str = "snap_unit",
    recommendations_col: str = "recommendations",
    snap_mode: SnapMode = "ceil",
    nonneg_policy: NonnegPolicy | None = None,
    infer_policy_from_recommendations: bool = True,
    out_base_col: str = "yhat_base_governed",
    out_ral_col: str = "yhat_ral_governed",
    out_audit_prefix: str = "ral_apply_",
) -> pd.DataFrame:
    """
    Canonical RAL application utility:
    - joins a required governance decisions table onto a panel,
    - produces a raw RAL prediction column (from yhat_ral_col or uplift_col),
    - applies governed nonnegativity + snap-to-grid policies,
    - emits governed prediction columns + audit columns.

    Notes
    -----
    - This function does *not* fit RAL. It applies already-produced predictions.
    - ``decisions`` is required and must include ``ral_policy``, ``status``,
      ``fas_class``, ``dqc_class``, and ``snap_required``. Ungated calls raise.
    - If yhat_ral_col is None, uplift_col must be provided and we compute:
        yhat_ral_raw = yhat_base * uplift
    - If infer_policy_from_recommendations is True and recommendations exist,
      snap and nonneg policies are inferred **per row** from recommendation
      strings emitted by run.py (comma-joined strings or sequences). Mixed
      entity policies are not collapsed to the first row.
    - When joined decisions mark ``ral_policy=disallow``, ``status=red``,
      ``fas_class=BLOCKED``, or ``dqc_class=UNKNOWN``, ``out_ral_col`` is copied
      from the governed baseline so readiness adjustment is not applied.
    """
    decisions = _require_decisions_table(decisions)

    # ---- apply legacy aliases (if provided) ----
    if join_keys is not None:
        key_cols = join_keys
    if pred_col is not None:
        # Legacy callers/tests pass the *prediction to govern* (not a baseline).
        yhat_ral_col = pred_col
        # If a baseline column isn't present, reuse the same prediction column so the
        # function can still emit out_base_col deterministically.
        if yhat_base_col not in df.columns:
            yhat_base_col = pred_col
    if output_col is not None:
        out_ral_col = output_col
    if nonneg_mode is not None and nonneg_policy is None:
        # Map run.py-style names onto the stable policy vocabulary.
        # "none" follows the balanced preset (clip_zero), matching the gate.
        if nonneg_mode == "none":
            nonneg_policy = _DEFAULT_APPLY_NONNEG
        elif nonneg_mode == "allow":
            nonneg_policy = "allow"
        elif nonneg_mode in ("clip", "clip_zero"):
            nonneg_policy = "clip_zero"
        else:
            raise ValueError(
                "apply_ral: nonneg_mode must be one of {'none','allow','clip','clip_zero'} "
                "(or use nonneg_policy={'allow','clip_zero'})."
            )

    if yhat_base_col not in df.columns:
        raise KeyError(f"apply_ral: missing required column {yhat_base_col!r} in df.")

    keys = list(key_cols)
    for k in keys:
        if k not in df.columns:
            raise KeyError(f"apply_ral: missing key column {k!r} in df.")

    work = df.copy()

    for k in keys:
        if k not in decisions.columns:
            raise KeyError(f"apply_ral: missing key column {k!r} in decisions.")

    overlap = [c for c in decisions.columns if c in work.columns and c not in keys]
    if overlap:
        work = work.drop(columns=overlap)
    merged = work.merge(decisions, on=keys, how="left", indicator=True)
    missing = merged.loc[merged["_merge"] != "both", keys]
    if not missing.empty:
        # Fail loudly: prevent silent "policy missing" behavior.
        missing_keys = missing.drop_duplicates().to_dict(orient="records")[
            :10
        ]  # cap for readability
        raise ValueError(
            "apply_ral: missing governance decision rows for some join keys. "
            f"Examples: {missing_keys}"
        )
    work = merged.drop(columns=["_merge"])

    # Build raw RAL prediction stream.
    if yhat_ral_col is not None:
        if yhat_ral_col not in work.columns:
            raise KeyError(f"apply_ral: missing column {yhat_ral_col!r} in df.")
        yhat_ral_raw = work[yhat_ral_col].to_numpy(dtype=float)
    else:
        if uplift_col is None:
            raise ValueError("apply_ral: yhat_ral_col is None, so uplift_col must be provided.")
        if uplift_col not in work.columns:
            raise KeyError(f"apply_ral: missing uplift column {uplift_col!r} in df.")
        yhat_ral_raw = work[yhat_base_col].to_numpy(dtype=float) * work[uplift_col].to_numpy(
            dtype=float
        )

    yhat_base_raw = work[yhat_base_col].to_numpy(dtype=float)

    infer_from_recs = infer_policy_from_recommendations and recommendations_col in work.columns
    recs_col = _series_from_column(work, recommendations_col) if infer_from_recs else None

    if nonneg_policy is not None:
        nonneg_s = pd.Series(nonneg_policy, index=work.index, dtype=object)
    elif recs_col is not None:
        nonneg_s = recs_col.map(_infer_nonneg_policy_from_recommendations)
    else:
        nonneg_s = pd.Series(_DEFAULT_APPLY_NONNEG, index=work.index, dtype=object)

    if recs_col is not None:
        snap_s = recs_col.map(
            lambda recs: _infer_snap_mode_from_recommendations(recs, default=snap_mode)
        )
    else:
        snap_s = pd.Series(snap_mode, index=work.index, dtype=object)

    yhat_base_g = yhat_base_raw.copy()
    yhat_ral_g = yhat_ral_raw.copy()
    clip_mask = nonneg_s.to_numpy() == "clip_zero"
    if bool(np.any(clip_mask)):
        yhat_base_g[clip_mask] = _apply_nonneg_policy(yhat_base_g[clip_mask], policy="clip_zero")
        yhat_ral_g[clip_mask] = _apply_nonneg_policy(yhat_ral_g[clip_mask], policy="clip_zero")

    work[out_base_col] = yhat_base_g
    work[out_ral_col] = yhat_ral_g

    flagged = pd.Series(False, index=work.index)
    if snap_required_col in work.columns:
        snap_required_s = work[snap_required_col].astype("boolean")
        flagged = snap_required_s.fillna(False).astype(bool)

    dqc_implies = pd.Series(False, index=work.index)
    if "dqc_class" in work.columns:
        dqc_implies = work["dqc_class"].map(_dqc_implies_snap).fillna(False).astype(bool)

    required = flagged | dqc_implies
    if bool(required.any()):
        if snap_unit_col not in work.columns:
            raise ValueError(
                "apply_ral: snapping is required (snap_required or quantized/packed DQC) "
                "but snap_unit column is missing; refusing fail-open unsnapped forecasts."
            )
        snap_unit_obj = pd.to_numeric(work[snap_unit_col], errors="coerce")
        snap_unit_s = (
            snap_unit_obj
            if isinstance(snap_unit_obj, pd.Series)
            else pd.Series(snap_unit_obj, index=work.index, name=snap_unit_col)
        )
        unit_vals = snap_unit_s.to_numpy(dtype=float)
        unit_ok = pd.Series(
            np.isfinite(unit_vals) & (unit_vals > 0.0),
            index=work.index,
        )
        invalid = required & ~unit_ok
        if bool(invalid.any()):
            raise ValueError(
                "apply_ral: snapping is required but snap_unit is missing or not "
                "finite and > 0; refusing fail-open unsnapped forecasts."
            )

        for mode_val in ("ceil", "round", "floor"):
            mode_mask = required & (snap_s == mode_val)
            if not bool(mode_mask.any()):
                continue
            mode_snap = cast(SnapMode, mode_val)
            work.loc[mode_mask, out_base_col] = _apply_snap_policy_series(
                work.loc[mode_mask, out_base_col],
                snap_unit_s.loc[mode_mask],
                mode=mode_snap,
            )
            work.loc[mode_mask, out_ral_col] = _apply_snap_policy_series(
                work.loc[mode_mask, out_ral_col],
                snap_unit_s.loc[mode_mask],
                mode=mode_snap,
            )

        clip_required = required & (nonneg_s == "clip_zero")
        if bool(clip_required.any()):
            work.loc[clip_required, out_base_col] = _apply_nonneg_policy(
                work.loc[clip_required, out_base_col].to_numpy(dtype=float),
                policy="clip_zero",
            )
            work.loc[clip_required, out_ral_col] = _apply_nonneg_policy(
                work.loc[clip_required, out_ral_col].to_numpy(dtype=float),
                policy="clip_zero",
            )

    blocked = _adjustment_blocked_mask(work)
    if bool(blocked.any()):
        work.loc[blocked, out_ral_col] = work.loc[blocked, out_base_col].to_numpy()

    work[f"{out_audit_prefix}nonneg_policy"] = nonneg_s
    work[f"{out_audit_prefix}snap_mode"] = snap_s
    work[f"{out_audit_prefix}ral_applied"] = ~blocked

    return work

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
from typing import Literal

import numpy as np
import pandas as pd

from eb_evaluation.diagnostics.governance import snap_to_grid
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
    ) -> pd.DataFrame:
        """Apply learned uplift factors to produce readiness forecasts.

        Test expectation:
        - If called before explicit fit(), this should still work for global uplift by
          implicitly fitting on the provided dataframe (requires an actual column), but only
          when costs (cu/co) are set.
        """
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
            uplift = merged["uplift"].to_numpy(dtype=float)

            mask_nan = ~np.isfinite(uplift)
            if mask_nan.any():
                uplift[mask_nan] = float(global_uplift)
        else:
            uplift = np.full(len(result_df), float(global_uplift), dtype=float)

        result_df[output_col] = result_df[forecast_col].to_numpy(dtype=float) * uplift
        return result_df


NonnegPolicy = Literal["allow", "clip_zero"]
SnapMode = Literal["ceil", "round", "floor"]


def _infer_nonneg_policy_from_recommendations(recs: Sequence[str] | None) -> NonnegPolicy:
    if not recs:
        return "allow"
    for r in recs:
        if r == "forecast_postprocess_nonneg(mode=clip_zero)":
            return "clip_zero"
    return "allow"


def _infer_snap_mode_from_recommendations(
    recs: Sequence[str] | None, *, default: SnapMode
) -> SnapMode:
    if not recs:
        return default
    prefix = "snap_forecasts_to_grid(mode="
    for r in recs:
        if r.startswith(prefix) and r.endswith(")"):
            mode = r[len(prefix) : -1]
            if mode in ("ceil", "round", "floor"):
                return mode  # type: ignore[return-value]
    return default


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
    - optionally joins governance decisions onto a panel,
    - produces a raw RAL prediction column (from yhat_ral_col or uplift_col),
    - applies governed nonnegativity + snap-to-grid policies,
    - emits governed prediction columns + audit columns.

    Notes
    -----
    - This function does *not* fit RAL. It applies already-produced predictions.
    - If yhat_ral_col is None, uplift_col must be provided and we compute:
        yhat_ral_raw = yhat_base * uplift
    - If infer_policy_from_recommendations is True and recommendations exist,
      policy is inferred from stable recommendation strings emitted by run.py.
    """

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
        # Map legacy run.py-style names -> policy
        if nonneg_mode in ("none", "allow"):
            nonneg_policy = "allow"
        elif nonneg_mode in ("clip", "clip_zero"):
            nonneg_policy = "clip_zero"
        else:
            raise ValueError(
                "apply_ral: nonneg_mode must be one of {'none','clip'} "
                "(or use nonneg_policy={'allow','clip_zero'})."
            )

    if yhat_base_col not in df.columns:
        raise KeyError(f"apply_ral: missing required column {yhat_base_col!r} in df.")

    keys = list(key_cols)
    for k in keys:
        if k not in df.columns:
            raise KeyError(f"apply_ral: missing key column {k!r} in df.")

    work = df.copy()

    if decisions is not None:
        for k in keys:
            if k not in decisions.columns:
                raise KeyError(f"apply_ral: missing key column {k!r} in decisions.")

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

    # Determine effective policy:
    # - If nonneg_policy explicitly provided, use it.
    # - Else, optionally infer from recommendations (first-row), otherwise allow.
    if recommendations_col in work.columns and infer_policy_from_recommendations:
        recs_first = work[recommendations_col].iloc[0] if len(work) else None
    else:
        recs_first = None

    nonneg_eff: NonnegPolicy
    if nonneg_policy is not None:
        nonneg_eff = nonneg_policy
    else:
        nonneg_eff = _infer_nonneg_policy_from_recommendations(recs_first)

    snap_mode_eff: SnapMode = snap_mode
    if infer_policy_from_recommendations and len(work):
        snap_mode_eff = _infer_snap_mode_from_recommendations(recs_first, default=snap_mode)

    # Apply nonnegativity first (matches run.py behavior: constrain before diagnostics/snapping).
    yhat_base_g = _apply_nonneg_policy(yhat_base_raw.copy(), policy=nonneg_eff)
    yhat_ral_g = _apply_nonneg_policy(yhat_ral_raw.copy(), policy=nonneg_eff)

    work[out_base_col] = yhat_base_g
    work[out_ral_col] = yhat_ral_g

    # Apply snapping if required and unit available.
    if snap_required_col in work.columns and snap_unit_col in work.columns:
        snap_required_s = work[snap_required_col].astype(bool)
        snap_unit_obj = pd.to_numeric(work[snap_unit_col], errors="coerce")
        snap_unit_s = (
            snap_unit_obj
            if isinstance(snap_unit_obj, pd.Series)
            else pd.Series(snap_unit_obj, index=work.index, name=snap_unit_col)
        )

        # Only snap rows where required + a finite positive unit exists.
        need = snap_required_s & snap_unit_s.notna() & (snap_unit_s > 0.0)
        if bool(need.any()):
            base_series = work.loc[need, out_base_col]
            ral_series = work.loc[need, out_ral_col]
            unit_series = snap_unit_s.loc[need]

            work.loc[need, out_base_col] = _apply_snap_policy_series(
                base_series, unit_series, mode=snap_mode_eff
            )
            work.loc[need, out_ral_col] = _apply_snap_policy_series(
                ral_series, unit_series, mode=snap_mode_eff
            )

            # If snapping can reintroduce negatives (e.g., round on negatives), re-apply nonneg.
            if nonneg_eff != "allow":
                idx = work.loc[need, out_base_col].index
                work.loc[need, out_base_col] = pd.Series(
                    _apply_nonneg_policy(
                        work.loc[need, out_base_col].to_numpy(dtype=float),
                        policy=nonneg_eff,
                    ),
                    index=idx,
                    name=out_base_col,
                )

                idx2 = work.loc[need, out_ral_col].index
                work.loc[need, out_ral_col] = pd.Series(
                    _apply_nonneg_policy(
                        work.loc[need, out_ral_col].to_numpy(dtype=float),
                        policy=nonneg_eff,
                    ),
                    index=idx2,
                    name=out_ral_col,
                )

    # Audit columns (stable, lightweight)
    work[f"{out_audit_prefix}nonneg_policy"] = nonneg_eff
    work[f"{out_audit_prefix}snap_mode"] = snap_mode_eff

    return work

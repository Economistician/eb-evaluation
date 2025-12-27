"""
Readiness Adjustment Layer (RAL): deterministic fit + apply in eb-evaluation.

This module implements a transparent post-processing step that converts a baseline forecast
into an operationally conservative readiness forecast via a learned uplift.

Responsibilities
---------------
- Fit a simple uplift policy via grid search that minimizes CWSL.
- Apply learned uplift factors to new data (global or segmented).
- Provide before/after diagnostics for auditability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

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
            raise TypeError(
                "ReadinessAdjustmentLayer requires cu and co to be set (via __init__)."
            )
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
            cwsl(
                y_true=y_true, y_pred=y_pred, cu=cu, co=co, sample_weight=sample_weight
            )
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
            if (loss < best_loss) or (
                abs(loss - best_loss) < 1e-12 and float(u) < best_u
            ):
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
            raise KeyError(
                f"sample_weight_col {sample_weight_col!r} not found in DataFrame."
            )

        seg_cols = list(segment_cols) if segment_cols is not None else []
        if seg_cols:
            seg_missing = [c for c in seg_cols if c not in df.columns]
            if seg_missing:
                raise KeyError(f"Missing segment columns for fit(): {seg_missing}")

        y_true_all = df[actual_col].to_numpy(dtype=float)
        y_pred_all = df[forecast_col].to_numpy(dtype=float)
        w_all = (
            df[sample_weight_col].to_numpy(dtype=float) if sample_weight_col else None
        )

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
                w = (
                    g[sample_weight_col].to_numpy(dtype=float)
                    if sample_weight_col
                    else None
                )

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
                raise RuntimeError(
                    "ReadinessAdjustmentLayer must be fit() before transform()."
                )

            if "actual" not in df.columns:
                raise RuntimeError(
                    "ReadinessAdjustmentLayer must be fit() before transform(), "
                    "or provide an 'actual' column to allow implicit global fit."
                )

            self.fit(
                df, forecast_col=forecast_col, actual_col="actual", segment_cols=None
            )

        seg_cols = (
            list(segment_cols) if segment_cols is not None else list(self.segment_cols_)
        )
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
                uplift[mask_nan] = float(self.global_uplift_)
        else:
            uplift = np.full(len(result_df), float(self.global_uplift_), dtype=float)

        result_df[output_col] = result_df[forecast_col].to_numpy(dtype=float) * uplift
        return result_df

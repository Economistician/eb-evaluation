from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ebmetrics.metrics import cwsl, frs, nsl


def _underbuild_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    Fraction of intervals where the forecast underbuilds demand (y_pred < y_true).
    """
    y_true_f = np.asarray(y_true, dtype=float)
    y_pred_f = np.asarray(y_pred, dtype=float)

    mask = y_pred_f < y_true_f
    if sample_weight is None:
        return float(mask.mean())

    w = np.asarray(sample_weight, dtype=float)
    if w.shape != y_true_f.shape:
        raise ValueError("sample_weight must have same shape as y_true.")
    return float((w * mask) / w.sum())


def _make_grid(u_min: float, u_max: float, step: float) -> np.ndarray:
    """
    Build an uplift grid INCLUDING the upper bound, robust to floating point drift.
    """
    if step <= 0:
        raise ValueError("grid_step must be strictly positive.")
    if u_min <= 0:
        raise ValueError("uplift_min must be strictly positive.")
    if u_max < u_min:
        raise ValueError("uplift_max must be >= uplift_min.")

    span = u_max - u_min
    # Number of steps from u_min to u_max in increments of `step`
    n_steps = int(round(span / step))

    grid = u_min + step * np.arange(n_steps + 1)

    # Clip and de-duplicate for safety
    grid = np.clip(grid, u_min, u_max)
    grid = np.unique(np.round(grid, 10))

    return grid


@dataclass
class ReadinessAdjustmentResult:
    """Container for per-segment uplift diagnostics."""

    uplift: float
    cwsl_before: float
    cwsl_after: float
    frs_before: float
    frs_after: float
    ub_rate_before: float
    ub_rate_after: float

    @property
    def cwsl_delta(self) -> float:
        # Negative is better (cost reduction)
        return self.cwsl_after - self.cwsl_before

    @property
    def frs_delta(self) -> float:
        # Positive is better (readiness gain)
        return self.frs_after - self.frs_before

    @property
    def ub_rate_delta(self) -> float:
        # Negative is better (fewer underbuild intervals)
        return self.ub_rate_after - self.ub_rate_before


class ReadinessAdjustmentLayer:
    """
    Readiness Adjustment Layer (RAL) for operational forecast uplift.

    This layer learns multiplicative uplift factors that adjust statistical
    forecasts into *readiness forecasts* targeted to minimize CWSL and
    optionally improve FRS / underbuild behavior.

    Typical usage
    -------------
    1. Fit uplift factors on historical data:

        ral = ReadinessAdjustmentLayer(cu=2.0, co=1.0)
        ral.fit(
            df,
            forecast_col="forecast",
            actual_col="actual",
            segment_cols=["store_cluster", "daypart"],
        )

    2. Apply uplift factors to new forecasts:

        df_future = ral.transform(
            df_future,
            forecast_col="forecast",
            output_col="readiness_forecast",
        )

    Segmentation
    ------------
    - If `segment_cols` is None → a single global uplift factor is learned.
    - If `segment_cols` is provided → one factor per segment combination.
    - At transform time, unseen segment combos fall back to the global uplift.
    """

    def __init__(
        self,
        cu: float = 2.0,
        co: float = 1.0,
        uplift_min: float = 1.0,
        uplift_max: float = 1.15,
        grid_step: float = 0.01,
        default_segment_cols: Optional[Sequence[str]] = None,
    ) -> None:
        if cu <= 0.0 or co <= 0.0:
            raise ValueError("cu and co must be strictly positive.")
        self.cu = float(cu)
        self.co = float(co)

        self.uplift_min = float(uplift_min)
        self.uplift_max = float(uplift_max)
        self.grid_step = float(grid_step)

        self.default_segment_cols: List[str] = list(default_segment_cols or [])

        # Learned attributes after fit()
        self.segment_cols_: List[str] = []
        self.global_uplift_: float = 1.0
        self.uplift_table_: pd.DataFrame = pd.DataFrame()
        self.diagnostics_: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        *,
        forecast_col: str,
        actual_col: str,
        segment_cols: Optional[Sequence[str]] = None,
        sample_weight_col: Optional[str] = None,
    ) -> "ReadinessAdjustmentLayer":
        """
        Learn uplift factors to minimize CWSL, optionally segmented.

        Parameters
        ----------
        df : pandas.DataFrame
            Historical data with forecast and actual columns.

        forecast_col : str
            Column containing baseline statistical forecasts.

        actual_col : str
            Column containing realized demand / actuals.

        segment_cols : sequence of str, optional
            Columns defining segmentation (e.g., ["store_cluster", "daypart"]).
            If None, falls back to `default_segment_cols` from __init__.
            If both are empty, a single global uplift factor is learned.

        sample_weight_col : str, optional
            Optional column with per-row weights used in CWSL / diagnostics.

        Returns
        -------
        self : ReadinessAdjustmentLayer
        """
        for col in [forecast_col, actual_col]:
            if col not in df.columns:
                raise KeyError(f"Column {col!r} not found in DataFrame.")

        if (df[actual_col] < 0).any():
            raise ValueError("Negative values found in actual_col; expected >= 0.")

        seg_cols = list(segment_cols) if segment_cols is not None else list(
            self.default_segment_cols
        )
        self.segment_cols_ = seg_cols

        if sample_weight_col is not None and sample_weight_col not in df.columns:
            raise KeyError(f"sample_weight_col {sample_weight_col!r} not found.")

        y_true_all = df[actual_col].to_numpy(dtype=float)
        y_pred_all = df[forecast_col].to_numpy(dtype=float)
        sw_all = (
            df[sample_weight_col].to_numpy(dtype=float)
            if sample_weight_col is not None
            else None
        )

        # ------------------------------------------------------------------
        # Global uplift (fallback + global diagnostics)
        # ------------------------------------------------------------------
        self.global_uplift_, global_result = self._fit_segment(
            y_true=y_true_all,
            y_pred=y_pred_all,
            sample_weight=sw_all,
        )

        records: List[Dict[str, Any]] = []

        # Add global row (no segment values)
        global_row: Dict[str, Any] = {
            "scope": "global",
            "uplift": global_result.uplift,
            "cwsl_before": global_result.cwsl_before,
            "cwsl_after": global_result.cwsl_after,
            "cwsl_delta": global_result.cwsl_delta,
            "frs_before": global_result.frs_before,
            "frs_after": global_result.frs_after,
            "frs_delta": global_result.frs_delta,
            "ub_rate_before": global_result.ub_rate_before,
            "ub_rate_after": global_result.ub_rate_after,
            "ub_rate_delta": global_result.ub_rate_delta,
        }
        records.append(global_row)

        # ------------------------------------------------------------------
        # Segmented uplifts (if any segment columns are provided)
        # ------------------------------------------------------------------
        if seg_cols:
            grouped = df.groupby(seg_cols, dropna=False, sort=False)

            seg_records: List[Dict[str, Any]] = []
            for key, g in grouped:
                # key is a scalar if 1 column, tuple otherwise
                if not isinstance(key, tuple):
                    key = (key,)

                y_true = g[actual_col].to_numpy(dtype=float)
                y_pred = g[forecast_col].to_numpy(dtype=float)
                sw = (
                    g[sample_weight_col].to_numpy(dtype=float)
                    if sample_weight_col is not None
                    else None
                )

                uplift, result = self._fit_segment(
                    y_true=y_true,
                    y_pred=y_pred,
                    sample_weight=sw,
                )

                row: Dict[str, Any] = {
                    "scope": "segment",
                    "uplift": uplift,
                    "cwsl_before": result.cwsl_before,
                    "cwsl_after": result.cwsl_after,
                    "cwsl_delta": result.cwsl_delta,
                    "frs_before": result.frs_before,
                    "frs_after": result.frs_after,
                    "frs_delta": result.frs_delta,
                    "ub_rate_before": result.ub_rate_before,
                    "ub_rate_after": result.ub_rate_after,
                    "ub_rate_delta": result.ub_rate_delta,
                }
                for col_name, value in zip(seg_cols, key):
                    row[col_name] = value
                seg_records.append(row)

            records.extend(seg_records)

            # Build per-segment uplift table (used during transform)
            uplift_table = (
                pd.DataFrame(seg_records)
                .set_index(seg_cols)
                .sort_index()
                .reset_index()
            )
        else:
            uplift_table = pd.DataFrame(
                columns=[*seg_cols, "uplift"],
                data=[],
            )

        self.uplift_table_ = uplift_table
        self.diagnostics_ = pd.DataFrame(records)

        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        forecast_col: str,
        output_col: str = "readiness_forecast",
        segment_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply learned uplift factors to new forecasts.

        Parameters
        ----------
        df : pandas.DataFrame
            New data containing at least `forecast_col` and (optionally)
            the segmentation columns used during fit().

        forecast_col : str
            Column with baseline forecast to be uplifted.

        output_col : str, default "readiness_forecast"
            Name of the column to hold the uplifted readiness forecast.

        segment_cols : sequence of str, optional
            Segment columns in `df`. If None, uses the columns stored from fit().
            If columns are missing or not found in the uplift table, the global
            uplift factor is used as fallback.

        Returns
        -------
        pandas.DataFrame
            Copy of input with an added `output_col`.
        """
        if forecast_col not in df.columns:
            raise KeyError(f"Column {forecast_col!r} not found in DataFrame.")

        if self.uplift_table_ is None or self.diagnostics_.empty:
            raise RuntimeError("ReadinessAdjustmentLayer must be fit() before transform().")

        seg_cols = list(segment_cols) if segment_cols is not None else self.segment_cols_

        result_df = df.copy()

        if seg_cols:
            # Ensure requested segment columns exist
            missing = [c for c in seg_cols if c not in result_df.columns]
            if missing:
                raise KeyError(
                    f"Segment columns {missing!r} not found in DataFrame during transform()."
                )

            # Merge per-segment uplift factors
            uplift_df = self.uplift_table_[seg_cols + ["uplift"]]
            merged = result_df.merge(
                uplift_df,
                on=seg_cols,
                how="left",
                suffixes=("", "_uplift"),
            )
            uplift = merged["uplift"].to_numpy(dtype=float)
            # Fallback to global uplift where no segment-specific uplift is found
            mask_nan = ~np.isfinite(uplift)
            if mask_nan.any():
                uplift[mask_nan] = self.global_uplift_
        else:
            # Pure global uplift
            uplift = np.full(len(result_df), self.global_uplift_, dtype=float)

        readiness_forecast = result_df[forecast_col].to_numpy(dtype=float) * uplift
        result_df[output_col] = readiness_forecast

        return result_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fit_segment(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[float, ReadinessAdjustmentResult]:
        """
        Grid-search uplift factor for a single segment to minimize CWSL.
        """
        y_true_f = np.asarray(y_true, dtype=float)
        y_pred_f = np.asarray(y_pred, dtype=float)

        if y_true_f.shape != y_pred_f.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if sw.shape != y_true_f.shape:
                raise ValueError("sample_weight must match shape of y_true.")
        else:
            sw = None

        grid = _make_grid(self.uplift_min, self.uplift_max, self.grid_step)

        cwsl_before = cwsl(
            y_true=y_true_f,
            y_pred=y_pred_f,
            cu=self.cu,
            co=self.co,
            sample_weight=sw,
        )

        frs_before = frs(
            y_true=y_true_f,
            y_pred=y_pred_f,
            cu=self.cu,
            co=self.co,
            sample_weight=sw,
        )

        nsl_before = nsl(
            y_true=y_true_f,
            y_pred=y_pred_f,
            sample_weight=sw,
        )
        ub_rate_before = 1.0 - nsl_before

        best_uplift = float(grid[0])
        best_cwsl = np.inf

        for u in grid:
            y_adj = y_pred_f * u
            val = cwsl(
                y_true=y_true_f,
                y_pred=y_adj,
                cu=self.cu,
                co=self.co,
                sample_weight=sw,
            )
            if val < best_cwsl:
                best_cwsl = float(val)
                best_uplift = float(u)

        # Diagnostics using best uplift
        y_best = y_pred_f * best_uplift
        cwsl_after = best_cwsl
        frs_after = frs(
            y_true=y_true_f,
            y_pred=y_best,
            cu=self.cu,
            co=self.co,
            sample_weight=sw,
        )
        nsl_after = nsl(
            y_true=y_true_f,
            y_pred=y_best,
            sample_weight=sw,
        )
        ub_rate_after = 1.0 - nsl_after

        result = ReadinessAdjustmentResult(
            uplift=best_uplift,
            cwsl_before=float(cwsl_before),
            cwsl_after=float(cwsl_after),
            frs_before=float(frs_before),
            frs_after=float(frs_after),
            ub_rate_before=float(ub_rate_before),
            ub_rate_after=float(ub_rate_after),
        )
        return best_uplift, result
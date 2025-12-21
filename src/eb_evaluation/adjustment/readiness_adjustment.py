from __future__ import annotations

r"""
Readiness Adjustment Layer (RAL).

This module implements the **Readiness Adjustment Layer (RAL)**: a lightweight, transparent
post-processing step that converts a baseline statistical forecast into an operationally
conservative *readiness forecast* via a learned multiplicative uplift.

RAL learns uplift factors by searching over a user-defined grid of multipliers and selecting
the value that minimizes **Cost-Weighted Service Loss (CWSL)** on historical data. It also
tracks secondary diagnostics such as **Forecast Readiness Score (FRS)** and an
underbuild-oriented service metric via `eb_metrics.metrics.nsl`.

Design notes
------------
- This module lives in **eb-evaluation** because it is an evaluation/selection utility.
  It *consumes* metric definitions from `eb_metrics.metrics` and does not re-define them.
- The learned uplift can be global or segmented by one or more categorical columns.
  Segment-level uplifts fall back to the global uplift for unseen segment combinations.

The RAL is intentionally simple and transparent: it is easy to explain, audit, and deploy
in operational environments where asymmetric costs (underbuild vs overbuild) matter.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eb_metrics.metrics import cwsl, frs, nsl


def _underbuild_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    r"""Compute the (optionally weighted) underbuild rate.

    The *underbuild rate* is the fraction of intervals where the forecast is strictly below
    realized demand:

    $$
    \mathrm{UB}(\hat{y}, y) = \frac{\sum_i w_i \,\mathbb{1}[\hat{y}_i < y_i]}{\sum_i w_i}
    $$

    where $w_i$ are optional non-negative sample weights.

    Parameters
    ----------
    y_true
        Array of realized demand / actuals.
    y_pred
        Array of baseline forecasts.
    sample_weight
        Optional non-negative weights with the same shape as `y_true`. If `None`,
        each interval is weighted equally.

    Returns
    -------
    float
        Underbuild rate in `[0, 1]`.

    Raises
    ------
    ValueError
        If `sample_weight` is provided with a shape mismatch or with non-positive total weight.

    Notes
    -----
    This helper is primarily a reporting utility. In the RAL implementation, underbuild
    behavior is also tracked via `eb_metrics.metrics.nsl` and is commonly reported
    as `ub_rate = 1 - nsl(...)` for consistency with EB service metrics.
    """
    y_true_f = np.asarray(y_true, dtype=float)
    y_pred_f = np.asarray(y_pred, dtype=float)

    mask = y_pred_f < y_true_f
    if sample_weight is None:
        return float(mask.mean())

    w = np.asarray(sample_weight, dtype=float)
    if w.shape != y_true_f.shape:
        raise ValueError("sample_weight must have same shape as y_true.")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("sample_weight must have positive total weight.")
    return float((w * mask).sum() / total)


def _make_grid(u_min: float, u_max: float, step: float) -> np.ndarray:
    """Create a grid of multiplicative uplift candidates.

    The returned grid:

    - starts at `u_min`
    - increments by `step`
    - includes `u_max` (to the extent permitted by floating point arithmetic)
    - is clipped and de-duplicated for numerical robustness

    Parameters
    ----------
    u_min
        Lower bound for the uplift grid. Must be strictly positive.
    u_max
        Upper bound for the uplift grid. Must be greater than or equal to `u_min`.
    step
        Step size between candidates. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        1D array of candidate uplift multipliers.

    Raises
    ------
    ValueError
        If `step` is not strictly positive, or if bounds are invalid.
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
    """Diagnostics for a fitted uplift (global or per-segment).

    This dataclass captures before/after metrics for a single uplift factor. It is returned
    by `ReadinessAdjustmentLayer._fit_segment` and is summarized into
    `ReadinessAdjustmentLayer.diagnostics_`.

    Attributes
    ----------
    uplift
        Multiplicative uplift applied to the baseline forecast.
    cwsl_before, cwsl_after
        Cost-Weighted Service Loss before/after applying the uplift. Lower is better.
    frs_before, frs_after
        Forecast Readiness Score before/after applying the uplift. Higher is better.
    ub_rate_before, ub_rate_after
        Underbuild-oriented rate before/after applying the uplift. Lower is better.
        In this implementation it is computed as `1 - nsl(...)` for alignment with EB service metrics.
    """

    uplift: float
    cwsl_before: float
    cwsl_after: float
    frs_before: float
    frs_after: float
    ub_rate_before: float
    ub_rate_after: float

    @property
    def cwsl_delta(self) -> float:
        """Change in CWSL after applying the uplift.

        Returns
        -------
        float
            `cwsl_after - cwsl_before`. Negative values indicate improvement (cost reduction).
        """
        return self.cwsl_after - self.cwsl_before

    @property
    def frs_delta(self) -> float:
        """Change in FRS after applying the uplift.

        Returns
        -------
        float
            `frs_after - frs_before`. Positive values indicate improvement.
        """
        return self.frs_after - self.frs_before

    @property
    def ub_rate_delta(self) -> float:
        """Change in underbuild-oriented rate after applying the uplift.

        Returns
        -------
        float
            `ub_rate_after - ub_rate_before`. Negative values indicate fewer underbuild intervals.
        """
        return self.ub_rate_after - self.ub_rate_before


class ReadinessAdjustmentLayer:
    r"""Readiness Adjustment Layer (RAL) for operational forecast uplift.

    The Readiness Adjustment Layer (RAL) is a transparent post-processing step that learns a
    multiplicative uplift $u$ and produces a *readiness forecast*:

    $$
    \hat{y}^{(r)} = u \cdot \hat{y}
    $$

    where $\hat{y}$ is a baseline statistical forecast and $\hat{y}^{(r)}$ is the uplifted
    readiness forecast.

    The uplift is selected via a grid search to minimize **Cost-Weighted Service Loss (CWSL)**:

    $$
    u^* = \arg\min_{u \in \mathcal{U}} \mathrm{CWSL}(u \cdot \hat{y}, y)
    $$

    Optionally, uplift can be learned per segment (e.g., store cluster × daypart). Segment-level
    uplifts fall back to a global uplift when an unseen segment combination appears at transform time.

    Parameters
    ----------
    cu
        Underbuild cost coefficient passed to `eb_metrics.metrics.cwsl` and
        `eb_metrics.metrics.frs`. Must be strictly positive.
    co
        Overbuild cost coefficient passed to `eb_metrics.metrics.cwsl` and
        `eb_metrics.metrics.frs`. Must be strictly positive.
    uplift_min
        Minimum candidate uplift multiplier (inclusive). Must be strictly positive.
    uplift_max
        Maximum candidate uplift multiplier (inclusive). Must be greater than or equal to `uplift_min`.
    grid_step
        Step size for candidate uplifts. Must be strictly positive.
    default_segment_cols
        Optional default segmentation columns used when `segment_cols` is not provided
        to `fit()` / `transform()`.

    Attributes
    ----------
    global_uplift_
        Learned global uplift factor (fit-time fallback).
    segment_cols_
        Segmentation columns used during `fit()`.
    uplift_table_
        DataFrame of per-segment uplift factors with a final `uplift` column. Empty when fit globally.
    diagnostics_
        DataFrame of global and per-segment diagnostics including before/after metrics and deltas.

    Examples
    --------
    >>> ral = ReadinessAdjustmentLayer(cu=2.0, co=1.0, uplift_max=1.15)
    >>> ral.fit(df, forecast_col="forecast", actual_col="actual", segment_cols=["cluster", "daypart"])
    ReadinessAdjustmentLayer(...)
    >>> out = ral.transform(df_future, forecast_col="forecast", output_col="readiness_forecast")

    Notes
    -----
    - RAL does not generate forecasts; it adjusts an existing forecast for operational readiness.
    - The uplift search is intentionally discrete and bounded to prioritize interpretability and
      deployability over continuous optimization.
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

        self.default_segment_cols: List[str] = (
            list(default_segment_cols) if default_segment_cols else []
        )

        # Learned artifacts (set during fit)
        self.global_uplift_: float = 1.0
        self.segment_cols_: List[str] = []
        self.uplift_table_: Optional[pd.DataFrame] = None
        self.diagnostics_: pd.DataFrame = pd.DataFrame()

        self._grid = _make_grid(self.uplift_min, self.uplift_max, self.grid_step)

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
        """Fit uplift factors from historical data.

        This method learns:

        1. A **global uplift** (always), used both as a diagnostic baseline and as a fallback.
        2. Optional **segment-level** uplifts when `segment_cols` is provided.

        For each scope, the uplift is chosen by grid-searching candidates in
        `[uplift_min, uplift_max]` with increment `grid_step` and selecting the candidate
        that minimizes `eb_metrics.metrics.cwsl`.

        Parameters
        ----------
        df
            Historical dataset containing forecasts, actuals, and optional segment and weight columns.
        forecast_col
            Column containing the baseline statistical forecast.
        actual_col
            Column containing realized demand / actual values.
        segment_cols
            Optional segmentation columns. If `None`, only a global uplift is learned.
            If provided, one uplift is learned per unique segment combination.
        sample_weight_col
            Optional column containing non-negative sample weights. Weights are passed through to the
            EB metric functions (CWSL/FRS/NSL) to support weighted evaluation (e.g., volume-weighted
            operational cost).

        Returns
        -------
        ReadinessAdjustmentLayer
            The fitted instance (`self`).

        Raises
        ------
        KeyError
            If required columns are missing from `df`.
        ValueError
            If `df` is empty after validation or if invalid hyperparameters are supplied.

        See Also
        --------
        transform
            Apply the learned uplift factors to new forecasts.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        if forecast_col not in df.columns:
            raise KeyError(f"forecast_col {forecast_col!r} not found.")
        if actual_col not in df.columns:
            raise KeyError(f"actual_col {actual_col!r} not found.")

        seg_cols = (
            list(segment_cols)
            if segment_cols is not None
            else list(self.default_segment_cols)
        )
        self.segment_cols_ = seg_cols

        for c in seg_cols:
            if c not in df.columns:
                raise KeyError(f"segment_col {c!r} not found.")

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
        # Segment-level uplift(s)
        # ------------------------------------------------------------------
        seg_records: List[Dict[str, Any]] = []
        if seg_cols:
            grouped = df.groupby(seg_cols, dropna=False)
            for key, g in grouped:
                # Normalize key to tuple (pandas gives scalar if one grouping col)
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

            # Uplift lookup table for transform()
            uplift_table = pd.DataFrame(seg_records)
            uplift_table = uplift_table[[*seg_cols, "uplift"]].copy()
        else:
            uplift_table = pd.DataFrame(
                columns=[*seg_cols, "uplift"],
                data=[],
            )

        self.uplift_table_ = uplift_table
        self.diagnostics_ = pd.DataFrame(records + seg_records)

        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        forecast_col: str,
        output_col: str = "readiness_forecast",
        segment_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Apply learned uplift factors to produce readiness forecasts.

        Parameters
        ----------
        df
            New data containing `forecast_col` and, when segmented, the segmentation columns.
        forecast_col
            Name of the baseline forecast column to uplift.
        output_col
            Name of the output column that will contain the readiness forecast.
        segment_cols
            Optional segmentation columns. If not provided, the segmentation used during `fit()`
            is used. If the layer was fit globally, this parameter is ignored.

        Returns
        -------
        pandas.DataFrame
            A copy of `df` with `output_col` added.

        Raises
        ------
        KeyError
            If required columns are missing.
        RuntimeError
            If the layer has not been fit prior to calling `transform()`.

        Notes
        -----
        When segmented, rows whose segment combination was not seen during `fit()` will use
        `global_uplift_` as a fallback.
        """
        if forecast_col not in df.columns:
            raise KeyError(f"Column {forecast_col!r} not found in DataFrame.")

        if self.uplift_table_ is None or self.diagnostics_.empty:
            raise RuntimeError(
                "ReadinessAdjustmentLayer must be fit() before transform()."
            )

        seg_cols = list(segment_cols) if segment_cols is not None else self.segment_cols_

        result_df = df.copy()

        if seg_cols:
            # Ensure requested segment columns exist
            missing = [c for c in seg_cols if c not in result_df.columns]
            if missing:
                raise KeyError(
                    f"Missing segment columns for transform(): {missing}. "
                    f"Available columns: {list(result_df.columns)}"
                )

            # Left-join uplifts onto rows
            merged = result_df.merge(self.uplift_table_, on=seg_cols, how="left")
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
        """Fit the best uplift for a single evaluation scope.

        Parameters
        ----------
        y_true
            Actual demand array for the scope.
        y_pred
            Baseline forecast array for the scope.
        sample_weight
            Optional sample weights aligned to `y_true` / `y_pred`.

        Returns
        -------
        tuple[float, ReadinessAdjustmentResult]
            The chosen uplift and a diagnostic bundle with before/after metrics.

        Notes
        -----
        The optimization objective is CWSL. FRS and the underbuild-oriented rate are tracked
        as secondary diagnostics.
        """
        y_true_f = np.asarray(y_true, dtype=float)
        y_pred_f = np.asarray(y_pred, dtype=float)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

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

        best_uplift = float(self._grid[0])
        best_cwsl = float("inf")
        y_best = y_pred_f

        for uplift in self._grid:
            y_adj = y_pred_f * float(uplift)
            score = cwsl(
                y_true=y_true_f,
                y_pred=y_adj,
                cu=self.cu,
                co=self.co,
                sample_weight=sw,
            )
            if score < best_cwsl:
                best_cwsl = float(score)
                best_uplift = float(uplift)
                y_best = y_adj

        cwsl_after = cwsl(
            y_true=y_true_f,
            y_pred=y_best,
            cu=self.cu,
            co=self.co,
            sample_weight=sw,
        )
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

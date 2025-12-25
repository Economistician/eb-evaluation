from __future__ import annotations

"""
Readiness Adjustment Layer (RAL).

This module implements the deterministic application of the Readiness Adjustment
Layer (RAL): a transparent post-processing step that converts a baseline forecast
into an operationally conservative readiness forecast via a fixed policy.

Responsibilities:
- Apply learned RAL policies to new data
- Provide before/after evaluation diagnostics
- Enforce fallback behavior for unseen segments

Non-responsibilities:
- Learning or tuning RAL parameters
- Searching over uplift candidates
- Owning optimization logic

Design notes:
- RAL policies are learned offline in `eb-optimization`
- This module remains deterministic and production-safe
- Optimization dependencies are intentionally excluded (except an optional fit wrapper)
"""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


class ReadinessAdjustmentLayer:
    r"""Readiness Adjustment Layer (RAL) for operational forecast uplift.

    The Readiness Adjustment Layer (RAL) is a transparent post-processing step that applies
    a multiplicative uplift `u` to a baseline statistical forecast to produce a readiness
    forecast:

    $$
    \hat{y}^{(r)} = u \cdot \hat{y}
    $$

    In this refactored architecture:

    - **Tuning / learning** uplift parameters is performed offline in `eb-optimization`.
    - This class is responsible for **deterministic application** of learned uplift factors
      and for storing audit diagnostics produced at tuning time.

    Parameters
    ----------
    cu
        Underbuild cost coefficient used by RAL tuning objectives (passed through to
        `eb-optimization` when calling `fit`). Must be strictly positive.
    co
        Overbuild cost coefficient used by RAL tuning objectives (passed through to
        `eb-optimization` when calling `fit`). Must be strictly positive.
    uplift_min
        Minimum candidate uplift multiplier (inclusive) used during tuning.
        Must be strictly positive.
    uplift_max
        Maximum candidate uplift multiplier (inclusive) used during tuning.
        Must be >= `uplift_min`.
    grid_step
        Step size for candidate uplifts used during tuning. Must be strictly positive.
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

    Notes
    -----
    - `transform()` is deterministic and does not require `eb-optimization`.
    - `fit()` is an optional convenience wrapper; it requires `eb-optimization` to be installed.
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
        if uplift_min <= 0.0:
            raise ValueError("uplift_min must be strictly positive.")
        if uplift_max < uplift_min:
            raise ValueError("uplift_max must be >= uplift_min.")
        if grid_step <= 0.0:
            raise ValueError("grid_step must be strictly positive.")

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
        """Fit uplift factors from historical data via `eb-optimization`.

        This method is a convenience wrapper around
        :func:`eb_optimization.tuning.ral.tune_ral_policy`.

        It learns:
        1. A **global uplift** (always), used as a fallback and baseline readiness adjustment.
        2. Optional **segment-level** uplifts when `segment_cols` is provided.

        The optimization objective is **Cost-Weighted Service Loss (CWSL)**. Secondary
        diagnostics are tracked for audit and interpretability.

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
            Optional column containing non-negative sample weights passed through to EB metrics.

        Returns
        -------
        ReadinessAdjustmentLayer
            The fitted instance (`self`).

        Raises
        ------
        ImportError
            If `eb-optimization` is not installed.
        ValueError
            If `df` is empty.
        KeyError
            If required columns are missing.

        Notes
        -----
        Once fit, `transform()` is deterministic and does not require `eb-optimization`.
        """
        try:
            from eb_optimization.tuning.ral import tune_ral_policy
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ReadinessAdjustmentLayer.fit requires eb-optimization. "
                "Install eb-optimization to tune RAL policies."
            ) from e

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

        for c in seg_cols:
            if c not in df.columns:
                raise KeyError(f"segment_col {c!r} not found.")

        if sample_weight_col is not None and sample_weight_col not in df.columns:
            raise KeyError(f"sample_weight_col {sample_weight_col!r} not found.")

        policy, diagnostics = tune_ral_policy(
            df,
            forecast_col=forecast_col,
            actual_col=actual_col,
            cu=self.cu,
            co=self.co,
            uplift_min=self.uplift_min,
            uplift_max=self.uplift_max,
            grid_step=self.grid_step,
            segment_cols=seg_cols if seg_cols else None,
            sample_weight_col=sample_weight_col,
        )

        self.global_uplift_ = float(policy.global_uplift)
        self.segment_cols_ = list(policy.segment_cols)

        if policy.uplift_table is None:
            self.uplift_table_ = pd.DataFrame(
                columns=[*self.segment_cols_, "uplift"],
                data=[],
            )
        else:
            self.uplift_table_ = policy.uplift_table.copy()

        self.diagnostics_ = diagnostics
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
            raise RuntimeError("ReadinessAdjustmentLayer must be fit() before transform().")

        seg_cols = list(segment_cols) if segment_cols is not None else self.segment_cols_

        result_df = df.copy()

        if seg_cols:
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
                uplift[mask_nan] = self.global_uplift_
        else:
            uplift = np.full(len(result_df), self.global_uplift_, dtype=float)

        result_df[output_col] = result_df[forecast_col].to_numpy(dtype=float) * uplift
        return result_df
from __future__ import annotations

"""
Readiness Adjustment Layer (RAL) - Apply-Only in `eb-evaluation`.

This module implements the deterministic application of the Readiness Adjustment
Layer (RAL): a transparent post-processing step that converts a baseline forecast
into an operationally conservative readiness forecast via a fixed policy.

Responsibilities:
- Apply learned RAL policies to new data
- Provide before/after evaluation diagnostics
"""

from typing import Optional, Sequence
import pandas as pd
import numpy as np

class ReadinessAdjustmentLayer:
    """Readiness Adjustment Layer (RAL) for operational forecast uplift.

    This is a **deterministic** post-processing step to apply RAL policies
    to new forecast data.
    """

    def __init__(self):
        self.global_uplift_ = 1.0
        self.segment_cols_ = []
        self.uplift_table_ = None
        self.diagnostics_ = pd.DataFrame()

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
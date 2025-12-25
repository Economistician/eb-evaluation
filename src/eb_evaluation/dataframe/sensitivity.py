from __future__ import annotations

"""
CWSL cost-ratio sensitivity (DataFrame utilities).

This module is a compatibility wrapper. The source-of-truth implementation
lives in `eb-optimization`:

- `eb_optimization.tuning.sensitivity.compute_cwsl_sensitivity_df`

Why?
- Sensitivity curves are hyperparameter-grid diagnostics and belong to the tuning layer.
- `eb-evaluation` remains focused on deterministic evaluation workflows.

If `eb-optimization` is not installed, calling this function raises ImportError.
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["compute_cwsl_sensitivity_df"]


def compute_cwsl_sensitivity_df(
    df: pd.DataFrame,
    *,
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    R_list: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: Union[float, str] = 1.0,
    group_cols: Optional[list[str]] = None,
    sample_weight_col: Optional[str] = None,
    # Backwards-compatible aliases
    y_true_col: Optional[str] = None,
    y_pred_col: Optional[str] = None,
) -> pd.DataFrame:
    r"""
    Compute CWSL sensitivity curves from a DataFrame.

    This function delegates to
    :func:`eb_optimization.tuning.sensitivity.compute_cwsl_sensitivity_df`.

    Parameters
    ----------
    df
        Input data.
    actual_col, forecast_col
        Column names for actuals and forecasts.
    R_list
        Candidate cost ratios to evaluate.
    co
        Overbuild cost specification (float or column name).
    group_cols
        Optional grouping columns.
    sample_weight_col
        Optional sample weight column.
    y_true_col, y_pred_col
        Backwards-compatible aliases for ``actual_col`` and ``forecast_col``.

    Returns
    -------
    pandas.DataFrame
        Long-form sensitivity results.

    Raises
    ------
    ImportError
        If `eb-optimization` is not installed.
    """
    # Resolve backwards-compatible aliases
    if y_true_col is not None:
        actual_col = y_true_col
    if y_pred_col is not None:
        forecast_col = y_pred_col

    try:
        from eb_optimization.tuning.sensitivity import compute_cwsl_sensitivity_df as _impl
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "compute_cwsl_sensitivity_df requires eb-optimization. "
            "Install eb-optimization to compute CWSL sensitivity curves."
        ) from e

    # Delegate (normalize group_cols=None -> None/[] handled by impl)
    return _impl(
        df=df,
        actual_col=actual_col,
        forecast_col=forecast_col,
        R_list=np.asarray(list(R_list), dtype=float),
        co=co,
        group_cols=group_cols,
        sample_weight_col=sample_weight_col,
    )
from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from ebmetrics.metrics import cwsl


def compute_cwsl_df(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    cu: Union[float, str],
    co: Union[float, str],
    sample_weight_col: Optional[str] = None,
) -> float:
    """
    Compute CWSL from a pandas DataFrame.

    This is a convenience wrapper around the core ``cwsl(...)`` function.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least the actual and forecast columns.

    y_true_col : str
        Name of the column in ``df`` containing actual demand.

    y_pred_col : str
        Name of the column in ``df`` containing forecasted demand.

    cu : float or str
        Either:
        * A scalar cost per unit of underbuild (applied to all rows), OR
        * The name of a column in ``df`` containing per-row underbuild costs.

    co : float or str
        Either:
        * A scalar cost per unit of overbuild (applied to all rows), OR
        * The name of a column in ``df`` containing per-row overbuild costs.

    sample_weight_col : str, optional
        If provided, the name of a column in ``df`` containing non-negative
        sample weights. If None, all rows are weighted equally.

    Returns
    -------
    float
        Cost-weighted service loss, demand-normalized.
    """
    # Extract core series as numpy arrays
    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)

    # Handle cu: scalar vs column name
    if isinstance(cu, str):
        cu_value = df[cu].to_numpy(dtype=float)
    else:
        cu_value = cu

    # Handle co: scalar vs column name
    if isinstance(co, str):
        co_value = df[co].to_numpy(dtype=float)
    else:
        co_value = co

    # Handle optional sample_weight column
    if sample_weight_col is not None:
        sample_weight = df[sample_weight_col].to_numpy(dtype=float)
    else:
        sample_weight = None

    # Delegate to the core implementation
    return cwsl(
        y_true=y_true,
        y_pred=y_pred,
        cu=cu_value,
        co=co_value,
        sample_weight=sample_weight,
    )
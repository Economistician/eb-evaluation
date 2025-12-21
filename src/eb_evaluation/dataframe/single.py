from __future__ import annotations

"""
Single-slice CWSL evaluation (DataFrame utilities).

This module provides a lightweight DataFrame wrapper around
``ebmetrics.metrics.cwsl`` for computing Cost-Weighted Service Loss (CWSL) on a single
slice of data (i.e., the entire DataFrame provided).

It supports both scalar costs and per-row cost columns for asymmetric cost evaluation.
"""

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
    r"""
    Compute CWSL from a DataFrame.

    This is a convenience wrapper around ``ebmetrics.metrics.cwsl`` that accepts a
    pandas DataFrame and column names.

    Costs can be specified either as scalars or as per-row columns. The cost ratio is:

    $$
    R = \frac{c_u}{c_o}
    $$

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least the actual and forecast columns, and optionally
        cost/weight columns.
    y_true_col : str
        Name of the column containing actual demand values.
    y_pred_col : str
        Name of the column containing forecast values.
    cu : float | str
        Underbuild (shortfall) cost coefficient.

        - If ``float``: scalar cost applied uniformly across all rows.
        - If ``str``: name of a column in ``df`` containing per-row underbuild costs.
    co : float | str
        Overbuild (excess) cost coefficient.

        - If ``float``: scalar cost applied uniformly across all rows.
        - If ``str``: name of a column in ``df`` containing per-row overbuild costs.
    sample_weight_col : str | None, default=None
        Optional column name containing non-negative sample weights per row. If ``None``,
        all rows are weighted equally.

    Returns
    -------
    float
        The Cost-Weighted Service Loss (CWSL) value for the provided DataFrame slice.

    Raises
    ------
    KeyError
        If any required columns are missing.
    ValueError
        If the underlying ``ebmetrics.metrics.cwsl`` raises due to invalid values.

    Notes
    -----
    This function performs minimal validation and delegates metric validation to
    ``ebmetrics.metrics.cwsl``.
    """
    required_cols = [y_true_col, y_pred_col]
    if isinstance(cu, str):
        required_cols.append(cu)
    if isinstance(co, str):
        required_cols.append(co)
    if sample_weight_col is not None:
        required_cols.append(sample_weight_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)

    cu_value = df[cu].to_numpy(dtype=float) if isinstance(cu, str) else cu
    co_value = df[co].to_numpy(dtype=float) if isinstance(co, str) else co

    sample_weight = (
        df[sample_weight_col].to_numpy(dtype=float)
        if sample_weight_col is not None
        else None
    )

    return cwsl(
        y_true=y_true,
        y_pred=y_pred,
        cu=cu_value,
        co=co_value,
        sample_weight=sample_weight,
    )
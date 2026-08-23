"""Single-slice CWSL helpers over pandas columns.

Wraps ``eb_metrics.metrics.cwsl`` for one DataFrame slice with scalar or per-row costs.
"""

from __future__ import annotations

import pandas as pd

from eb_metrics.metrics import cwsl


def compute_cwsl_df(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    cu: float | str,
    co: float | str,
    sample_weight_col: str | None = None,
) -> float:
    """Compute CWSL for one DataFrame slice.

    ``cu`` / ``co`` may be floats or column names; optional ``sample_weight_col``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with actual and forecast columns, and optional cost/weight columns.
    y_true_col : str
        Column of actual demand values.
    y_pred_col : str
        Column of forecast values.
    cu : float | str
        Underbuild cost; scalar or column name.
    co : float | str
        Overbuild cost; scalar or column name.
    sample_weight_col : str | None, default=None
        Optional non-negative per-row weights. If ``None``, rows are weighted equally.

    Returns
    -------
    float
        Cost-Weighted Service Loss for the slice.

    Raises
    ------
    KeyError
        If any required columns are missing.
    ValueError
        If ``eb_metrics.metrics.cwsl`` rejects the inputs.
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
        df[sample_weight_col].to_numpy(dtype=float) if sample_weight_col is not None else None
    )

    return cwsl(
        y_true=y_true,
        y_pred=y_pred,
        cu=cu_value,
        co=co_value,
        sample_weight=sample_weight,
    )

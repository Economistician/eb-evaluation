from __future__ import annotations

from typing import Union

import pandas as pd

from ebmetrics.metrics import (
    cwsl,
    nsl,
    ud,
    wmape,
    hr_at_tau,
    frs,
    mae,
    rmse,
    mape,
)


def evaluate_groups_df(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    cu: Union[float, str] = 2.0,
    co: Union[float, str] = 1.0,
    tau: float = 2.0,
    sample_weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute all core CWSL metrics per group from a pandas DataFrame.

    For each group defined by ``group_cols``, this helper computes:
        - CWSL
        - NSL
        - UD
        - wMAPE
        - HR@tau
        - FRS
        - MAE
        - RMSE
        - MAPE

    It supports both global and per-row asymmetric costs:

        - If ``cu`` / ``co`` are floats, they are treated as global costs.
        - If ``cu`` / ``co`` are column names (str), those columns are read
          per row and passed through to the underlying metric functions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing actuals, forecasts, and grouping columns.

    group_cols : list of str
        Column names to group by (e.g. ``["store_id", "item_id"]``).

    actual_col : str, default "actual_qty"
        Name of the column containing actual demand.

    forecast_col : str, default "forecast_qty"
        Name of the column containing forecasted demand.

    cu : float or str, default 2.0
        Underbuild (shortfall) cost per unit. Either:
        - scalar, applied uniformly across all rows/groups, OR
        - name of a column in ``df`` containing per-row underbuild costs.

    co : float or str, default 1.0
        Overbuild (excess) cost per unit. Either:
        - scalar, applied uniformly across all rows/groups, OR
        - name of a column in ``df`` containing per-row overbuild costs.

    tau : float, default 2.0
        Tolerance parameter for HR@tau (absolute units).

    sample_weight_col : str or None, default None
        Optional column name containing non-negative sample weights
        per row. If provided, all metrics that accept ``sample_weight``
        will use this column.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per group and columns:
        ``group_cols + ["CWSL", "NSL", "UD", "wMAPE", "HR@tau",
                        "FRS", "MAE", "RMSE", "MAPE"]``.

        If a metric is undefined for a particular group (e.g. due to
        invalid data), the corresponding value is NaN rather than
        raising an error for the entire evaluation.
    """
    # Basic column validation
    missing = [c for c in group_cols + [actual_col, forecast_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    if isinstance(cu, str) and cu not in df.columns:
        raise KeyError(f"cu column '{cu}' not found in df")
    if isinstance(co, str) and co not in df.columns:
        raise KeyError(f"co column '{co}' not found in df")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col '{sample_weight_col}' not found in df")

    results: list[dict] = []

    # Helper to safely compute a metric per group (return NaN on ValueError)
    def _safe_metric(fn) -> float:
        try:
            return float(fn())
        except ValueError:
            return float("nan")

    grouped = df.groupby(group_cols, sort=False)

    for key, g in grouped:
        # Normalize group key into a tuple
        if not isinstance(key, tuple):
            key = (key,)

        row: dict = {col: val for col, val in zip(group_cols, key)}

        y_true = g[actual_col].to_numpy(dtype=float)
        y_pred = g[forecast_col].to_numpy(dtype=float)

        if sample_weight_col is not None:
            sample_weight = g[sample_weight_col].to_numpy(dtype=float)
        else:
            sample_weight = None

        # Handle cu: scalar vs column
        if isinstance(cu, str):
            cu_value = g[cu].to_numpy(dtype=float)
        else:
            cu_value = cu

        # Handle co: scalar vs column
        if isinstance(co, str):
            co_value = g[co].to_numpy(dtype=float)
        else:
            co_value = co

        # Core + diagnostics
        row["CWSL"] = _safe_metric(
            lambda: cwsl(
                y_true,
                y_pred,
                cu=cu_value,
                co=co_value,
                sample_weight=sample_weight,
            )
        )
        row["NSL"] = _safe_metric(lambda: nsl(y_true, y_pred, sample_weight=sample_weight))
        row["UD"] = _safe_metric(lambda: ud(y_true, y_pred, sample_weight=sample_weight))

        # wMAPE in ebmetrics does not take sample_weight, so call unweighted.
        row["wMAPE"] = _safe_metric(
            lambda: wmape(y_true, y_pred)
        )

        row["HR@tau"] = _safe_metric(
            lambda: hr_at_tau(
                y_true,
                y_pred,
                tau=tau,
                sample_weight=sample_weight,
            )
        )
        row["FRS"] = _safe_metric(
            lambda: frs(
                y_true,
                y_pred,
                cu=cu_value,
                co=co_value,
                sample_weight=sample_weight,
            )
        )

        # Baseline symmetric metrics: current ebmetrics versions do not
        # accept sample_weight for these, so we call them unweighted.
        row["MAE"] = _safe_metric(lambda: mae(y_true, y_pred))
        row["RMSE"] = _safe_metric(lambda: rmse(y_true, y_pred))
        row["MAPE"] = _safe_metric(lambda: mape(y_true, y_pred))

        results.append(row)

    return pd.DataFrame(results)
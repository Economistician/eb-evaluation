from __future__ import annotations

"""
Group-level evaluation (DataFrame utilities).

This module provides helpers for evaluating forecasts on grouped subsets of a DataFrame
(e.g., by store, item, daypart, region). It orchestrates grouping, parameter handling,
and tabular output, while delegating metric *definitions* to :mod:`ebmetrics.metrics`.

The primary entry point is :func:`evaluate_groups_df`, which computes the Electric Barometer
metric suite (CWSL, NSL, UD, HR@τ, FRS) plus common symmetric diagnostics (wMAPE, MAE, RMSE, MAPE)
for each group.
"""

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
    r"""
    Evaluate core EB metrics per group from a DataFrame.

    For each group defined by ``group_cols``, this helper computes:

    - CWSL
    - NSL
    - UD
    - wMAPE
    - HR@τ
    - FRS
    - MAE
    - RMSE
    - MAPE

    Cost parameters can be provided either globally (scalar) or per-row (column name).

    If ``cu`` and ``co`` are scalars, they are treated as global costs:

    $$
        c_u \in \mathbb{R}_{>0}, \quad c_o \in \mathbb{R}_{>0}
    $$

    If ``cu`` and/or ``co`` are strings, they are interpreted as column names and the
    corresponding per-row arrays are passed through to the underlying EB metric functions:

    $$
        c_u = c_u(i), \quad c_o = c_o(i)
    $$

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing actuals, forecasts, and grouping columns.
    group_cols : list[str]
        Column names used to define groups (e.g., ``["store_id", "item_id"]``).
    actual_col : str, default="actual_qty"
        Name of the column containing actual demand values.
    forecast_col : str, default="forecast_qty"
        Name of the column containing forecast values.
    cu : float | str, default=2.0
        Underbuild (shortfall) cost coefficient.

        - If ``float``: scalar cost applied uniformly across all rows/groups.
        - If ``str``: name of a column in ``df`` containing per-row underbuild costs.
    co : float | str, default=1.0
        Overbuild (excess) cost coefficient.

        - If ``float``: scalar cost applied uniformly across all rows/groups.
        - If ``str``: name of a column in ``df`` containing per-row overbuild costs.
    tau : float, default=2.0
        Absolute-error tolerance parameter for the hit-rate metric HR@τ.
    sample_weight_col : str | None, default=None
        Optional column name containing non-negative sample weights per row. If provided,
        weights are passed into metrics that accept a ``sample_weight`` argument.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per group and columns:

        ``group_cols + ["CWSL", "NSL", "UD", "wMAPE", "HR@tau", "FRS", "MAE", "RMSE", "MAPE"]``.

        If a metric is undefined for a particular group (e.g., invalid values for that group),
        the corresponding value is returned as ``NaN`` rather than raising an error for the
        entire evaluation.

    Raises
    ------
    KeyError
        If required columns are missing from ``df``.
    ValueError
        If ``df`` is empty, or if ``group_cols`` is empty.

    Notes
    -----
    - ``wmape`` in :mod:`ebmetrics.metrics` does not take a ``sample_weight`` argument, so it is
      computed unweighted here (consistent with the ebmetrics API).
    - Symmetric diagnostics (MAE, RMSE, MAPE) are computed unweighted to match the current
      ebmetrics signatures.
    - Metrics are evaluated group-by-group; a failure in one group does not prevent evaluation
      of other groups.

    """
    if df.empty:
        raise ValueError("df is empty.")
    if not group_cols:
        raise ValueError("group_cols must be a non-empty list of column names.")

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

    def _safe_metric(fn) -> float:
        """Compute a metric for a single group; return NaN if the metric raises ValueError."""
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

        sample_weight = (
            g[sample_weight_col].to_numpy(dtype=float) if sample_weight_col is not None else None
        )

        # Handle cu: scalar vs column
        cu_value = g[cu].to_numpy(dtype=float) if isinstance(cu, str) else cu

        # Handle co: scalar vs column
        co_value = g[co].to_numpy(dtype=float) if isinstance(co, str) else co

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
        row["wMAPE"] = _safe_metric(lambda: wmape(y_true, y_pred))

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

        # Baseline symmetric metrics: current ebmetrics versions do not accept sample weights.
        row["MAE"] = _safe_metric(lambda: mae(y_true, y_pred))
        row["RMSE"] = _safe_metric(lambda: rmse(y_true, y_pred))
        row["MAPE"] = _safe_metric(lambda: mape(y_true, y_pred))

        results.append(row)

    return pd.DataFrame(results)
"""
Group-level evaluation (DataFrame utilities).

This module provides helpers for evaluating forecasts on grouped subsets of a DataFrame
(e.g., by store, item, daypart, region). It orchestrates grouping, parameter handling, and
tabular output while delegating metric definitions to ``eb_metrics.metrics``.

The primary entry point is ``evaluate_groups_df``, which computes the Electric Barometer
metric suite (CWSL, NSL, UD, HR@tau, FRS) plus common symmetric diagnostics (wMAPE, MAE,
RMSE, MAPE) for each group.
"""

from __future__ import annotations

import pandas as pd

from eb_metrics.metrics import cwsl, frs, hr_at_tau, mae, mape, nsl, rmse, ud, wmape


def evaluate_groups_df(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    cu: float | str = 2.0,
    co: float | str = 1.0,
    tau: float = 2.0,
    sample_weight_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate core EB metrics per group from a DataFrame.

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

    Cost parameters can be provided either globally (scalar) or per-row (column name).

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
        Absolute-error tolerance parameter for the hit-rate metric HR@tau.
    sample_weight_col : str | None, default=None
        Optional column name containing non-negative sample weights per row. If provided,
        weights are passed into metrics that accept a ``sample_weight`` argument.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per group and columns::

            group_cols + ["CWSL", "NSL", "UD", "wMAPE", "HR@tau", "FRS", "MAE", "RMSE", "MAPE"].

        If a metric is undefined for a particular group (e.g., invalid values for that
        group), the corresponding value is returned as NaN rather than raising an error
        for the entire evaluation.

    Raises
    ------
    KeyError
        If required columns are missing from ``df``.
    ValueError
        If ``df`` is empty, or if ``group_cols`` is empty.

    Notes
    -----
    - ``wmape`` in ``eb_metrics.metrics`` does not take ``sample_weight``, so it is
      computed unweighted here.
    - Symmetric diagnostics (MAE, RMSE, MAPE) are computed unweighted to match the
      current ``eb_metrics`` signatures.
    - Metrics are evaluated group-by-group; a failure in one group does not prevent
      evaluation of other groups.
    """
    if df.empty:
        raise ValueError("df is empty.")
    if not group_cols:
        raise ValueError("group_cols must be a non-empty list of column names.")

    missing = [
        c for c in [*group_cols, actual_col, forecast_col] if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    if isinstance(cu, str) and cu not in df.columns:
        raise KeyError(f"cu column {cu!r} not found in df")
    if isinstance(co, str) and co not in df.columns:
        raise KeyError(f"co column {co!r} not found in df")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in df")

    results: list[dict] = []

    def _safe_metric(fn) -> float:
        """Compute a metric for a single group, returning NaN on ValueError."""
        try:
            return float(fn())
        except ValueError:
            return float("nan")

    grouped = df.groupby(group_cols, sort=False)

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row: dict = dict(zip(group_cols, key, strict=False))

        y_true = g[actual_col].to_numpy(dtype=float)
        y_pred = g[forecast_col].to_numpy(dtype=float)

        sample_weight = (
            g[sample_weight_col].to_numpy(dtype=float)
            if sample_weight_col is not None
            else None
        )

        cu_value = g[cu].to_numpy(dtype=float) if isinstance(cu, str) else cu
        co_value = g[co].to_numpy(dtype=float) if isinstance(co, str) else co

        row["CWSL"] = _safe_metric(
            lambda y_true=y_true,
            y_pred=y_pred,
            cu_value=cu_value,
            co_value=co_value,
            sample_weight=sample_weight: cwsl(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu_value,
                co=co_value,
                sample_weight=sample_weight,
            )
        )
        row["NSL"] = _safe_metric(
            lambda y_true=y_true, y_pred=y_pred, sample_weight=sample_weight: nsl(
                y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
            )
        )
        row["UD"] = _safe_metric(
            lambda y_true=y_true, y_pred=y_pred, sample_weight=sample_weight: ud(
                y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
            )
        )

        # wMAPE in eb_metrics does not take sample_weight, so call unweighted.
        row["wMAPE"] = _safe_metric(
            lambda y_true=y_true, y_pred=y_pred: wmape(y_true=y_true, y_pred=y_pred)
        )

        row["HR@tau"] = _safe_metric(
            lambda y_true=y_true,
            y_pred=y_pred,
            tau=tau,
            sample_weight=sample_weight: hr_at_tau(
                y_true=y_true,
                y_pred=y_pred,
                tau=tau,
                sample_weight=sample_weight,
            )
        )
        row["FRS"] = _safe_metric(
            lambda y_true=y_true,
            y_pred=y_pred,
            cu_value=cu_value,
            co_value=co_value,
            sample_weight=sample_weight: frs(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu_value,
                co=co_value,
                sample_weight=sample_weight,
            )
        )

        row["MAE"] = _safe_metric(
            lambda y_true=y_true, y_pred=y_pred: mae(y_true=y_true, y_pred=y_pred)
        )
        row["RMSE"] = _safe_metric(
            lambda y_true=y_true, y_pred=y_pred: rmse(y_true=y_true, y_pred=y_pred)
        )
        row["MAPE"] = _safe_metric(
            lambda y_true=y_true, y_pred=y_pred: mape(y_true=y_true, y_pred=y_pred)
        )

        results.append(row)

    return pd.DataFrame(results)

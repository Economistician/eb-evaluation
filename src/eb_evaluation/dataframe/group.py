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

import numpy as np
import pandas as pd


def _compose_frs(nsl_val: float, cwsl_val: float, cwsl_max: float) -> float:
    """FRS = NSL - min(1, CWSL / CWSL_max) from already-computed components."""
    cwsl_max_val = float(cwsl_max)
    if not np.isfinite(cwsl_max_val) or cwsl_max_val <= 0.0:
        raise ValueError("cwsl_max must be finite and strictly greater than 0.")
    return float(nsl_val - min(1.0, cwsl_val / cwsl_max_val))


def _safe_compose_frs(nsl_val: float, cwsl_val: float, cwsl_max: float) -> float:
    try:
        return _compose_frs(nsl_val, cwsl_val, cwsl_max)
    except ValueError:
        return float("nan")


def evaluate_groups_df(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    cu: float | str,
    co: float | str,
    tau: float = 2.0,
    sample_weight_col: str | None = None,
    cwsl_max: float,
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

    Interval-level shortfall, overbuild, and coverage masks are computed once at
    the panel level; groups are reduced with a single ``groupby.sum()``. Groups
    that would make a metric raise ``ValueError`` (zero demand with positive
    cost, non-finite or negative demand/forecast, zero total weight, etc.)
    receive NaN for that metric only.

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
    cu : float | str
        Underbuild (shortfall) cost coefficient. Required; there is no default.

        - If ``float``: scalar cost applied uniformly across all rows/groups.
        - If ``str``: name of a column in ``df`` containing per-row underbuild costs.
    co : float | str
        Overbuild (excess) cost coefficient. Required; there is no default.

        - If ``float``: scalar cost applied uniformly across all rows/groups.
        - If ``str``: name of a column in ``df`` containing per-row overbuild costs.
    tau : float, default=2.0
        Absolute-error tolerance parameter for the hit-rate metric HR@tau.
    sample_weight_col : str | None, default=None
        Optional column name containing non-negative sample weights per row. If provided,
        weights are passed into metrics that accept a ``sample_weight`` argument.
    cwsl_max : float
        Required upper bound for FRS scaling: the largest economically meaningful CWSL
        for the application. Must be finite and strictly greater than 0. There is no
        default.

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
    - FRS is composed from the group's NSL and CWSL: ``NSL - min(1, CWSL / CWSL_max)``.
    """
    if df.empty:
        raise ValueError("df is empty.")
    if not group_cols:
        raise ValueError("group_cols must be a non-empty list of column names.")

    missing = [c for c in [*group_cols, actual_col, forecast_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    if isinstance(cu, str) and cu not in df.columns:
        raise KeyError(f"cu column {cu!r} not found in df")
    if isinstance(co, str) and co not in df.columns:
        raise KeyError(f"co column {co!r} not found in df")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in df")

    y_true = df[actual_col].to_numpy(dtype=float)
    y_pred = df[forecast_col].to_numpy(dtype=float)

    if sample_weight_col is not None:
        w = df[sample_weight_col].to_numpy(dtype=float)
    else:
        w = None

    if isinstance(cu, str):
        cu_arr = df[cu].to_numpy(dtype=float)
        cu_scalar = None
    else:
        cu_arr = None
        cu_as = np.asarray(cu, dtype=float)
        if cu_as.ndim == 0:
            cu_scalar = float(cu_as)
        else:
            raise ValueError("cu must be a scalar or a column name.")

    if isinstance(co, str):
        co_arr = df[co].to_numpy(dtype=float)
        co_scalar = None
    else:
        co_arr = None
        co_as = np.asarray(co, dtype=float)
        if co_as.ndim == 0:
            co_scalar = float(co_as)
        else:
            raise ValueError("co must be a scalar or a column name.")

    delta = y_true - y_pred
    shortfall = np.maximum(delta, 0.0)
    overbuild = shortfall - delta
    abs_err = np.abs(delta)

    if cu_scalar is not None and co_scalar is not None:
        cost = cu_scalar * shortfall + co_scalar * overbuild
    elif cu_scalar is not None:
        cost = cu_scalar * shortfall + co_arr * overbuild
    elif co_scalar is not None:
        cost = cu_arr * shortfall + co_scalar * overbuild
    else:
        cost = cu_arr * shortfall + co_arr * overbuild

    nsl_hit = (y_pred >= y_true).astype(np.float64)
    hr_hit = (abs_err <= float(tau)).astype(np.float64)
    sf_flag = (delta > 0).astype(np.float64)
    nonzero_y = y_true != 0
    mape_term = np.zeros_like(y_true)
    mape_term[nonzero_y] = abs_err[nonzero_y] / np.abs(y_true[nonzero_y])

    svc_bad = ~np.isfinite(y_true) | ~np.isfinite(y_pred) | (y_true < 0) | (y_pred < 0)
    if w is not None:
        w_bad = ~np.isfinite(w) | (w < 0)
    else:
        w_bad = np.zeros(len(df), dtype=bool)

    cost_bad = np.zeros(len(df), dtype=bool)
    if cu_scalar is not None:
        if (not np.isfinite(cu_scalar)) or cu_scalar < 0:
            cost_bad[:] = True
    else:
        cost_bad |= ~np.isfinite(cu_arr) | (cu_arr < 0)
    if co_scalar is not None:
        if (not np.isfinite(co_scalar)) or co_scalar < 0:
            cost_bad[:] = True
    else:
        cost_bad |= ~np.isfinite(co_arr) | (co_arr < 0)

    tau_bad = (not np.isfinite(float(tau))) or float(tau) < 0

    if w is None:
        w_cost = cost
        w_y = y_true
        w_nsl = nsl_hit
        w_hr = hr_hit
        w_sf = sf_flag
        w_short = shortfall
        w_mass = None
    else:
        w_cost = w * cost
        w_y = w * y_true
        w_nsl = w * nsl_hit
        w_hr = w * hr_hit
        w_sf = w * sf_flag
        w_short = w * shortfall
        w_mass = w

    feat_cols: dict[str, np.ndarray] = {
        **{c: df[c].to_numpy() for c in group_cols},
        "w_cost": w_cost,
        "w_y": w_y,
        "w_nsl": w_nsl,
        "w_hr": w_hr,
        "w_sf": w_sf,
        "w_short": w_short,
    }
    if w_mass is not None:
        feat_cols["w_mass"] = w_mass
    feat_cols.update(
        {
            "abs_err": abs_err,
            "sq_err": delta * delta,
            "abs_y": np.abs(y_true),
            "mape_term": mape_term,
            "mape_n": nonzero_y.astype(np.float64),
            "svc_bad": svc_bad.astype(np.float64),
            "w_bad": w_bad.astype(np.float64),
            "cost_bad": cost_bad.astype(np.float64),
        }
    )
    feat = pd.DataFrame(feat_cols)

    grouped = feat.groupby(group_cols, sort=False)
    agg = grouped.sum(numeric_only=True)
    n_g = grouped.size().astype(np.float64)

    demand = agg["w_y"].to_numpy(dtype=float)
    tot_cost = agg["w_cost"].to_numpy(dtype=float)
    n_arr = n_g.to_numpy(dtype=float)
    w_sum = n_arr if w_mass is None else agg["w_mass"].to_numpy(dtype=float)
    w_sf_sum = agg["w_sf"].to_numpy(dtype=float)

    svc_invalid = agg["svc_bad"].to_numpy(dtype=float) > 0
    w_invalid = agg["w_bad"].to_numpy(dtype=float) > 0
    cost_invalid = agg["cost_bad"].to_numpy(dtype=float) > 0
    service_blocked = svc_invalid | w_invalid

    cwsl_out = np.where(demand > 0, tot_cost / demand, np.where(tot_cost == 0.0, 0.0, np.nan))
    cwsl_out = np.where(service_blocked | cost_invalid, np.nan, cwsl_out)

    nsl_out = np.where((w_sum > 0) & ~service_blocked, agg["w_nsl"].to_numpy(dtype=float) / w_sum, np.nan)

    ud_raw = np.divide(
        agg["w_short"].to_numpy(dtype=float),
        w_sf_sum,
        out=np.zeros(w_sf_sum.shape, dtype=float),
        where=w_sf_sum > 0,
    )
    ud_out = np.where((w_sum > 0) & ~service_blocked, ud_raw, np.nan)

    hr_ok = (w_sum > 0) & ~service_blocked
    if tau_bad:
        hr_out = np.full(len(agg), np.nan)
    else:
        hr_out = np.where(hr_ok, agg["w_hr"].to_numpy(dtype=float) / w_sum, np.nan)

    cwsl_max_val = float(cwsl_max)
    if not np.isfinite(cwsl_max_val) or cwsl_max_val <= 0.0:
        frs_out = np.full(len(agg), np.nan)
    else:
        frs_out = nsl_out - np.minimum(1.0, cwsl_out / cwsl_max_val)

    mae_out = agg["abs_err"].to_numpy(dtype=float) / n_arr
    rmse_out = np.sqrt(agg["sq_err"].to_numpy(dtype=float) / n_arr)
    abs_y = agg["abs_y"].to_numpy(dtype=float)
    wmape_out = np.where(abs_y == 0.0, np.nan, 100.0 * agg["abs_err"].to_numpy(dtype=float) / abs_y)
    mape_n = agg["mape_n"].to_numpy(dtype=float)
    mape_out = np.where(mape_n == 0.0, np.nan, 100.0 * agg["mape_term"].to_numpy(dtype=float) / mape_n)

    out = agg.reset_index()
    result = out.loc[:, group_cols].copy()
    result["CWSL"] = cwsl_out
    result["NSL"] = nsl_out
    result["UD"] = ud_out
    result["wMAPE"] = wmape_out
    result["HR@tau"] = hr_out
    result["FRS"] = frs_out
    result["MAE"] = mae_out
    result["RMSE"] = rmse_out
    result["MAPE"] = mape_out
    return result

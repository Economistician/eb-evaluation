"""Group-level forecast evaluation over a DataFrame.

Computes the Electric Barometer metric suite plus common symmetric diagnostics
per group, delegating primitives to ``eb_metrics.metrics``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
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

    Per-group CWSL, NSL, UD, HR@τ, FRS, wMAPE, MAE, RMSE, and MAPE. Costs may be
    scalar or per-row columns. Invalid groups yield NaN for the failing metric only.

    Interval-level shortfall, overbuild, and coverage masks are computed once at
    the panel level; groups are reduced with a single ``groupby.sum()``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing actuals, forecasts, and grouping columns.
    group_cols : list[str]
        Column names used to define groups (e.g., ``["store_id", "item_id"]``).
    actual_col : str, default="actual_qty"
        Column of actual demand values.
    forecast_col : str, default="forecast_qty"
        Column of forecast values.
    cu : float | str
        Underbuild cost; scalar or column name.
    co : float | str
        Overbuild cost; scalar or column name.
    tau : float, default=2.0
        Absolute-error tolerance for HR@tau.
    sample_weight_col : str | None, default=None
        Optional non-negative per-row weights for metrics that accept ``sample_weight``.
    cwsl_max : float
        Finite upper bound for FRS scaling; must be ``> 0``.

    Returns
    -------
    pandas.DataFrame
        One row per group with columns::

            group_cols + ["CWSL", "NSL", "UD", "wMAPE", "HR@tau", "FRS", "MAE", "RMSE", "MAPE"].

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

    w: NDArray[np.float64] | None = (
        df[sample_weight_col].to_numpy(dtype=float) if sample_weight_col is not None else None
    )

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
    w_bad = ~np.isfinite(w) | (w < 0) if w is not None else np.zeros(len(df), dtype=bool)

    cost_bad = np.zeros(len(df), dtype=bool)
    if cu_scalar is not None:
        if (not np.isfinite(cu_scalar)) or cu_scalar < 0:
            cost_bad[:] = True
    elif cu_arr is not None:
        cost_bad |= ~np.isfinite(cu_arr) | (cu_arr < 0)
    if co_scalar is not None:
        if (not np.isfinite(co_scalar)) or co_scalar < 0:
            cost_bad[:] = True
    elif co_arr is not None:
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
    agg = pd.DataFrame(grouped.sum(numeric_only=True))
    n_g = pd.Series(grouped.size(), dtype=np.float64)

    demand = agg["w_y"].to_numpy(dtype=float)
    tot_cost = agg["w_cost"].to_numpy(dtype=float)
    n_arr = n_g.to_numpy(dtype=float)
    w_sum = n_arr if w_mass is None else agg["w_mass"].to_numpy(dtype=float)

    svc_invalid = agg["svc_bad"].to_numpy(dtype=float) > 0
    w_invalid = agg["w_bad"].to_numpy(dtype=float) > 0
    cost_invalid = agg["cost_bad"].to_numpy(dtype=float) > 0
    service_blocked = svc_invalid | w_invalid

    cwsl_out = np.where(demand > 0, tot_cost / demand, np.where(tot_cost == 0.0, 0.0, np.nan))
    cwsl_out = np.where(service_blocked | cost_invalid, np.nan, cwsl_out)

    nsl_out = np.where(
        (w_sum > 0) & ~service_blocked, agg["w_nsl"].to_numpy(dtype=float) / w_sum, np.nan
    )

    # UD is conditional on shortfall intervals, matching eb_metrics.ud:
    # mean shortfall magnitude over T^SF = {i : y_i > yhat_i}. No shortfalls => 0.0.
    sf_weight = agg["w_sf"].to_numpy(dtype=float)
    short_sum = agg["w_short"].to_numpy(dtype=float)
    ud_ratio = np.zeros_like(short_sum)
    has_sf = sf_weight > 0
    ud_ratio[has_sf] = short_sum[has_sf] / sf_weight[has_sf]
    ud_out = np.where(service_blocked | (w_sum <= 0), np.nan, ud_ratio)

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
    mape_out = np.where(
        mape_n == 0.0, np.nan, 100.0 * agg["mape_term"].to_numpy(dtype=float) / mape_n
    )

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

"""
Data-driven tolerance (τ) selection utilities for HR@τ.

This module provides deterministic, residual-only methods for selecting the tolerance
parameter τ used by the hit-rate metric HR@τ (hit rate within an absolute-error band).

The hit-rate metric is:

$$
    \mathrm{HR}@\tau = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left(|y_i-\hat{y}_i|\le \tau\right)
$$

Here, τ defines an *acceptability band*: the maximum absolute error considered operationally
acceptable.

Design notes
------------
- τ is estimated from historical residuals only (no exogenous data, no model assumptions).
- The module supports global τ estimation and entity-level τ estimation.
- Optional governance controls allow capping entity τ values by a global cap to prevent
  tolerance inflation.

Methods
-------
The global estimator supports three selection modes:

1. ``"target_hit_rate"``:
   choose τ such that a target fraction of residual magnitudes is covered:

   $$
       \tau = Q_h\left(|e|\right), \quad e_i = y_i-\hat{y}_i
   $$

   where $Q_h(\cdot)$ is the quantile function at level $h$.

2. ``"knee"``:
   select τ at a diminishing-returns point on the monotone curve $\mathrm{HR}@\tau$.

3. ``"utility"``:
   maximize a simple tradeoff between coverage and tolerance width:

   $$
       \tau^\* = \arg\max_{\tau \in \mathcal{T}}
       \left[\mathrm{HR}@\tau - \lambda\left(\frac{\tau}{\tau_{\max}}\right)\right]
   $$

The entity-level estimator runs the same procedure per entity, optionally capping each
entity τ by a global cap derived from the full residual distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Single source of truth for the HR@τ metric math
from ebmetrics.metrics.service import hr_at_tau as _hr_at_tau_core

TauMethod = Literal["target_hit_rate", "knee", "utility"]


def _to_1d_float_array(x: Union[pd.Series, np.ndarray, Iterable[float]]) -> np.ndarray:
    """Convert input to a 1D float NumPy array."""
    return np.asarray(x, dtype=float).reshape(-1)


def _nan_safe_abs_errors(
    y: Union[pd.Series, np.ndarray, Iterable[float]],
    yhat: Union[pd.Series, np.ndarray, Iterable[float]],
) -> np.ndarray:
    """
    Compute absolute errors with NaN/inf filtering.

    Non-finite (y, yhat) pairs are dropped. The returned array contains:

    $$
        |e_i| = |y_i - \hat{y}_i|
    $$

    for finite pairs only.
    """
    y_arr = _to_1d_float_array(y)
    yhat_arr = _to_1d_float_array(yhat)
    if y_arr.shape[0] != yhat_arr.shape[0]:
        raise ValueError(
            f"y and yhat must have the same length. Got {len(y_arr)} vs {len(yhat_arr)}."
        )
    mask = np.isfinite(y_arr) & np.isfinite(yhat_arr)
    return np.abs(y_arr[mask] - yhat_arr[mask])


def _validate_tau(tau: float) -> float:
    """Validate τ as finite and non-negative."""
    if not np.isfinite(tau):
        raise ValueError(f"tau must be finite. Got {tau}.")
    if tau < 0:
        raise ValueError(f"tau must be >= 0. Got {tau}.")
    return float(tau)


def _quantile(x: np.ndarray, q: float) -> float:
    """Compute a quantile with basic guards; returns NaN for empty input."""
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile q must be in [0, 1]. Got {q}.")
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def _make_tau_grid(
    abs_errors: np.ndarray,
    grid: Optional[Union[np.ndarray, Iterable[float]]] = None,
    grid_size: int = 101,
    grid_quantiles: Tuple[float, float] = (0.0, 0.99),
) -> np.ndarray:
    """
    Construct a non-negative τ grid.

    If ``grid`` is provided, it is filtered to finite, unique, non-negative values.
    Otherwise, a linear grid is constructed between quantiles of the absolute error
    distribution.
    """
    if abs_errors.size == 0:
        return np.array([], dtype=float)

    if grid is not None:
        g = _to_1d_float_array(grid)
        g = g[np.isfinite(g)]
        g = np.unique(g)
        g = g[g >= 0]
        return g

    q_lo, q_hi = grid_quantiles
    if not (0 <= q_lo <= q_hi <= 1):
        raise ValueError(
            f"grid_quantiles must satisfy 0 <= q_lo <= q_hi <= 1. Got {grid_quantiles}."
        )

    lo = _quantile(abs_errors, q_lo)
    hi = _quantile(abs_errors, q_hi)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.array([], dtype=float)

    if hi < lo:
        hi = lo

    if grid_size < 2:
        return np.array([lo], dtype=float)

    return np.linspace(lo, hi, grid_size, dtype=float)


def hr_at_tau(
    y: Union[pd.Series, np.ndarray, Iterable[float]],
    yhat: Union[pd.Series, np.ndarray, Iterable[float]],
    tau: float,
) -> float:
    r"""
    Compute HR@τ: fraction of observations whose absolute error is within τ.

    HR@τ is defined as:

    $$
        \mathrm{HR}@\tau = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left(|y_i-\hat{y}_i|\le \tau\right)
    $$

    This is an evaluation-friendly wrapper around the core implementation in
    :func:`ebmetrics.metrics.service.hr_at_tau`. Non-finite (y, yhat) pairs are dropped
    prior to delegating. If no finite pairs remain, returns ``np.nan``.

    Parameters
    ----------
    y : array-like
        Actual values $y_i$.
    yhat : array-like
        Forecast values $\hat{y}_i$.
    tau : float
        Non-negative tolerance threshold τ.

    Returns
    -------
    float
        Hit rate within τ, or ``np.nan`` if no finite pairs remain.

    Raises
    ------
    ValueError
        If input lengths do not match or if ``tau`` is invalid.

    """
    tau = _validate_tau(tau)

    y_arr = _to_1d_float_array(y)
    yhat_arr = _to_1d_float_array(yhat)
    if y_arr.shape[0] != yhat_arr.shape[0]:
        raise ValueError(
            f"y and yhat must have the same length. Got {len(y_arr)} vs {len(yhat_arr)}."
        )

    mask = np.isfinite(y_arr) & np.isfinite(yhat_arr)
    if not np.any(mask):
        return np.nan

    # Delegate metric math to ebmetrics (single source of truth)
    return float(_hr_at_tau_core(y_true=y_arr[mask], y_pred=yhat_arr[mask], tau=tau))


@dataclass(frozen=True)
class TauEstimate:
    """
    Result container for τ estimation.

    Attributes
    ----------
    tau : float
        Estimated tolerance τ (may be NaN if estimation failed).
    method : str
        Method identifier used to produce the estimate.
    n : int
        Number of finite (y, yhat) pairs used.
    diagnostics : dict[str, Any]
        Method-specific diagnostics intended for reporting and governance.

    """
    tau: float
    method: str
    n: int
    diagnostics: Dict[str, Any]


def estimate_tau(
    y: Union[pd.Series, np.ndarray, Iterable[float]],
    yhat: Union[pd.Series, np.ndarray, Iterable[float]],
    method: TauMethod = "target_hit_rate",
    *,
    # target_hit_rate params
    target_hit_rate: float = 0.90,
    # knee params
    grid: Optional[Union[np.ndarray, Iterable[float]]] = None,
    grid_size: int = 101,
    grid_quantiles: Tuple[float, float] = (0.0, 0.99),
    knee_rule: Literal["slope_threshold", "max_distance"] = "slope_threshold",
    slope_threshold: float = 0.0025,
    # utility params
    lambda_: float = 0.10,
    tau_max: Optional[float] = None,
    # general guards
    tau_floor: float = 0.0,
    tau_cap: Optional[float] = None,
) -> TauEstimate:
    r"""
    Estimate a global tolerance τ from residuals.

    Residuals are defined as:

    $$
        e_i = y_i - \hat{y}_i
    $$

    and the estimator uses the empirical distribution of $|e_i|$ after filtering
    non-finite pairs.

    Parameters
    ----------
    y : array-like
        Actual values $y_i$.
    yhat : array-like
        Forecast values $\hat{y}_i$.
    method : {"target_hit_rate", "knee", "utility"}, default="target_hit_rate"
        Method used to select τ.

        - ``"target_hit_rate"``: choose τ as a quantile of $|e|$ such that
          $\mathrm{HR}@\tau \approx h$ on the calibration residuals.
        - ``"knee"``: choose τ at a diminishing-returns point of the curve
          $\mathrm{HR}@\tau$ evaluated on a τ grid.
        - ``"utility"``: choose τ to maximize
          $\mathrm{HR}@\tau - \lambda(\tau/\tau_{\max})$ on a τ grid.
    target_hit_rate : float, default=0.90
        Target coverage level $h$ used by the ``"target_hit_rate"`` method.
        Must satisfy $0 < h \le 1$.
    grid : array-like | None, default=None
        Optional explicit τ grid for ``"knee"`` and ``"utility"``. If provided,
        values are filtered to finite, unique, non-negative values.
    grid_size : int, default=101
        Grid size used when constructing an implicit τ grid.
    grid_quantiles : tuple[float, float], default=(0.0, 0.99)
        Quantile range used to construct an implicit τ grid from $|e|$.
    knee_rule : {"slope_threshold", "max_distance"}, default="slope_threshold"
        Knee detection rule used when ``method="knee"``.
    slope_threshold : float, default=0.0025
        Threshold used by the ``"slope_threshold"`` knee rule. Smaller values pick
        earlier knees.
    lambda_ : float, default=0.10
        Tradeoff weight used by the ``"utility"`` method. Must be non-negative.
    tau_max : float | None, default=None
        Reference tolerance width used by the ``"utility"`` method. If ``None``,
        it defaults to the 0.99 quantile of $|e|$.
    tau_floor : float, default=0.0
        Minimum allowed τ (must be non-negative).
    tau_cap : float | None, default=None
        Optional maximum allowed τ (must be non-negative if provided).

    Returns
    -------
    TauEstimate
        Estimated τ plus diagnostics describing the calibration behavior.

    Raises
    ------
    ValueError
        If parameters are invalid (e.g., negative τ bounds, invalid quantiles, etc.).

    """
    abs_errors = _nan_safe_abs_errors(y, yhat)
    n = int(abs_errors.size)

    if n == 0:
        return TauEstimate(
            tau=np.nan,
            method=str(method),
            n=0,
            diagnostics={"reason": "no_finite_pairs"},
        )

    tau_floor = _validate_tau(tau_floor)
    if tau_cap is not None:
        tau_cap = _validate_tau(tau_cap)

    if method == "target_hit_rate":
        if not (0.0 < target_hit_rate <= 1.0):
            raise ValueError(
                f"target_hit_rate must be in (0, 1]. Got {target_hit_rate}."
            )

        tau = _quantile(abs_errors, target_hit_rate)

        if np.isfinite(tau):
            tau = max(tau, tau_floor)
            if tau_cap is not None:
                tau = min(tau, tau_cap)

        diag = {
            "target_hit_rate": float(target_hit_rate),
            "achieved_hr_calibration": float(np.mean(abs_errors <= tau))
            if np.isfinite(tau)
            else np.nan,
            "abs_error_quantile_used": float(target_hit_rate),
            "tau_floor": float(tau_floor),
            "tau_cap": float(tau_cap) if tau_cap is not None else None,
        }
        return TauEstimate(tau=float(tau), method="target_hit_rate", n=n, diagnostics=diag)

    tau_grid = _make_tau_grid(
        abs_errors,
        grid=grid,
        grid_size=grid_size,
        grid_quantiles=grid_quantiles,
    )
    if tau_grid.size == 0:
        return TauEstimate(
            tau=np.nan,
            method=str(method),
            n=n,
            diagnostics={"reason": "empty_tau_grid"},
        )

    # HR curve on grid (monotone non-decreasing)
    e_sorted = np.sort(abs_errors)
    idx = np.searchsorted(e_sorted, tau_grid, side="right")
    hr_curve = idx / float(n)

    if method == "knee":
        if knee_rule == "slope_threshold":
            d_tau = np.diff(tau_grid)
            d_hr = np.diff(hr_curve)
            slope = np.where(d_tau > 0, d_hr / d_tau, np.inf)

            candidates = np.where(slope < slope_threshold)[0]
            pick_i = int(candidates[0] + 1) if candidates.size > 0 else int(len(tau_grid) - 1)

            tau = float(tau_grid[pick_i])
            hr_pick = float(hr_curve[pick_i])

            diag = {
                "knee_rule": knee_rule,
                "slope_threshold": float(slope_threshold),
                "picked_index": pick_i,
                "picked_hr_calibration": hr_pick,
                "grid_size": int(tau_grid.size),
                "tau_grid_min": float(tau_grid.min()),
                "tau_grid_max": float(tau_grid.max()),
                "tau_floor": float(tau_floor),
                "tau_cap": float(tau_cap) if tau_cap is not None else None,
            }

        elif knee_rule == "max_distance":
            t0, t1 = float(tau_grid[0]), float(tau_grid[-1])
            t_norm = (tau_grid - t0) / (t1 - t0) if t1 > t0 else np.zeros_like(tau_grid)

            x = t_norm
            yv = hr_curve
            x0, y0 = 0.0, float(hr_curve[0])
            x1, y1 = 1.0, float(hr_curve[-1])

            denom = np.hypot(x1 - x0, y1 - y0)
            if denom == 0:
                pick_i = int(len(tau_grid) // 2)
            else:
                dist = np.abs((y1 - y0) * x - (x1 - x0) * yv + x1 * y0 - y1 * x0) / denom
                pick_i = int(np.argmax(dist))

            tau = float(tau_grid[pick_i])
            hr_pick = float(hr_curve[pick_i])

            diag = {
                "knee_rule": knee_rule,
                "picked_index": pick_i,
                "picked_hr_calibration": hr_pick,
                "grid_size": int(tau_grid.size),
                "tau_grid_min": float(tau_grid.min()),
                "tau_grid_max": float(tau_grid.max()),
                "tau_floor": float(tau_floor),
                "tau_cap": float(tau_cap) if tau_cap is not None else None,
            }
        else:
            raise ValueError(f"Unknown knee_rule: {knee_rule}")

        tau = max(tau, tau_floor)
        if tau_cap is not None:
            tau = min(tau, tau_cap)

        return TauEstimate(tau=float(tau), method="knee", n=n, diagnostics=diag)

    if method == "utility":
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be >= 0. Got {lambda_}.")

        if tau_max is None:
            tau_max_val = _quantile(abs_errors, 0.99)
        else:
            tau_max_val = float(tau_max)

        if not np.isfinite(tau_max_val) or tau_max_val <= 0:
            tau_max_val = (
                float(tau_grid[-1])
                if np.isfinite(tau_grid[-1]) and tau_grid[-1] > 0
                else 1.0
            )

        utility = hr_curve - float(lambda_) * (tau_grid / tau_max_val)
        pick_i = int(np.argmax(utility))

        tau = float(tau_grid[pick_i])
        hr_pick = float(hr_curve[pick_i])
        u_pick = float(utility[pick_i])

        tau = max(tau, tau_floor)
        if tau_cap is not None:
            tau = min(tau, tau_cap)

        diag = {
            "lambda_": float(lambda_),
            "tau_max": float(tau_max_val),
            "picked_index": pick_i,
            "picked_hr_calibration": hr_pick,
            "picked_utility": u_pick,
            "grid_size": int(tau_grid.size),
            "tau_grid_min": float(tau_grid.min()),
            "tau_grid_max": float(tau_grid.max()),
            "tau_floor": float(tau_floor),
            "tau_cap": float(tau_cap) if tau_cap is not None else None,
        }
        return TauEstimate(tau=float(tau), method="utility", n=n, diagnostics=diag)

    raise ValueError(f"Unknown method: {method}")


def estimate_entity_tau(
    df: pd.DataFrame,
    *,
    entity_col: str,
    y_col: str,
    yhat_col: str,
    method: TauMethod = "target_hit_rate",
    min_n: int = 30,
    estimate_kwargs: Optional[Mapping[str, Any]] = None,
    cap_with_global: bool = False,
    global_cap_quantile: float = 0.99,
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    r"""
    Estimate τ per entity from residuals.

    For each entity, residual magnitudes are:

    $$
        |e_i| = |y_i - \hat{y}_i|
    $$

    and τ is estimated using the same method as :func:`estimate_tau`, provided the entity has
    at least ``min_n`` finite residual pairs.

    If ``cap_with_global=True``, each entity τ is capped by a global cap derived from the full
    residual distribution:

    $$
        \tau_{\mathrm{cap}} = Q_q\left(|e|\right)
    $$

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing entity identifiers and (y, yhat) columns.
    entity_col : str
        Entity identifier column.
    y_col : str
        Actual values column.
    yhat_col : str
        Forecast values column.
    method : {"target_hit_rate", "knee", "utility"}, default="target_hit_rate"
        τ estimation method.
    min_n : int, default=30
        Minimum number of finite (y, yhat) pairs required to produce an entity τ.
        Entities below this threshold return τ = NaN.
    estimate_kwargs : Mapping[str, Any] | None, default=None
        Additional keyword arguments forwarded to :func:`estimate_tau`.
    cap_with_global : bool, default=False
        If True, cap each entity τ by a global cap computed using ``global_cap_quantile`` over
        the entire DataFrame residual distribution.
    global_cap_quantile : float, default=0.99
        Quantile used to compute the global τ cap. Must be in [0, 1].
    include_diagnostics : bool, default=True
        If True, include diagnostic fields derived from :class:`TauEstimate.diagnostics`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per entity including:

        - ``entity_col``
        - ``tau`` : estimated entity tolerance
        - ``n`` : number of finite residual pairs used
        - ``method`` : method identifier

        If diagnostics are enabled, additional columns are appended.

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If ``min_n < 1``.

    """
    if estimate_kwargs is None:
        estimate_kwargs = {}

    required = {entity_col, y_col, yhat_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if min_n < 1:
        raise ValueError(f"min_n must be >= 1. Got {min_n}.")

    global_cap = None
    if cap_with_global:
        abs_errors_all = _nan_safe_abs_errors(df[y_col], df[yhat_col])
        global_cap = _quantile(abs_errors_all, global_cap_quantile)
        if not np.isfinite(global_cap):
            global_cap = None

    rows: list[dict[str, Any]] = []

    for ent, g in df.groupby(entity_col, dropna=False):
        abs_errors = _nan_safe_abs_errors(g[y_col], g[yhat_col])
        n = int(abs_errors.size)

        if n < min_n:
            rows.append(
                {
                    entity_col: ent,
                    "tau": np.nan,
                    "n": n,
                    "method": method,
                    "reason": f"min_n_not_met(<{min_n})",
                }
            )
            continue

        est = estimate_tau(
            y=g[y_col],
            yhat=g[yhat_col],
            method=method,
            **dict(estimate_kwargs),
        )

        tau_val = est.tau
        if global_cap is not None and np.isfinite(tau_val):
            tau_val = float(min(tau_val, global_cap))

        row: dict[str, Any] = {
            entity_col: ent,
            "tau": tau_val,
            "n": est.n,
            "method": est.method,
        }

        if include_diagnostics:
            diag = dict(est.diagnostics or {})
            row["diagnostics"] = diag
            row["achieved_hr_calibration"] = diag.get(
                "achieved_hr_calibration", diag.get("picked_hr_calibration")
            )
            row["tau_floor"] = diag.get("tau_floor")
            row["tau_cap"] = diag.get("tau_cap")
            if method == "utility":
                row["lambda_"] = diag.get("lambda_")
                row["tau_max"] = diag.get("tau_max")
                row["picked_utility"] = diag.get("picked_utility")
            if method == "knee":
                row["knee_rule"] = diag.get("knee_rule")

        if global_cap is not None:
            row["global_cap_tau"] = global_cap
            row["global_cap_quantile"] = float(global_cap_quantile)

        rows.append(row)

    out = pd.DataFrame(rows)

    base_cols = [entity_col, "tau", "n", "method"]
    extra_cols = [c for c in out.columns if c not in base_cols]
    out = out[base_cols + extra_cols]

    return out


def hr_auto_tau(
    y: Union[pd.Series, np.ndarray, Iterable[float]],
    yhat: Union[pd.Series, np.ndarray, Iterable[float]],
    method: TauMethod = "target_hit_rate",
    **estimate_kwargs: Any,
) -> Tuple[float, float, Dict[str, Any]]:
    r"""
    Estimate τ from residuals, then compute HR@τ.

    This is a convenience wrapper that performs:

    1. τ estimation via :func:`estimate_tau`, then
    2. HR@τ evaluation via :func:`hr_at_tau`.

    Returns
    -------
    tuple[float, float, dict[str, Any]]
        Tuple ``(hr, tau, diagnostics)`` where:

        - ``hr`` is the computed HR@τ
        - ``tau`` is the estimated τ
        - ``diagnostics`` are method-specific details from the τ estimator

        If τ cannot be estimated (e.g., no finite pairs), returns ``(nan, nan, diagnostics)``.

    """
    est = estimate_tau(y=y, yhat=yhat, method=method, **estimate_kwargs)
    if not np.isfinite(est.tau):
        return (np.nan, np.nan, dict(est.diagnostics or {}))

    hr = hr_at_tau(y, yhat, est.tau)
    return (float(hr), float(est.tau), dict(est.diagnostics or {}))
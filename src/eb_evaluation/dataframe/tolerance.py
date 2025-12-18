"""
eb_evaluation.dataframe.tolerance

Data-driven tolerance (tau, τ) selection utilities for HR@τ (hit-rate within threshold).

Theme:
- Like cost_ratio.py estimates R = cu/co by balancing under- vs over-cost,
  this module estimates τ from *historical residuals only* (no exogenous data).
- Supports global τ and entity-level τ, with optional caps/guards.

Core idea:
HR(τ) = mean( |y - yhat| <= τ )

We can choose τ automatically from residuals via:
- target_hit_rate (quantile of abs errors)
- knee (diminishing returns point on HR curve)
- utility (maximize HR(τ) - λ * τ/τ_max)

All functions are deterministic given inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


TauMethod = Literal["target_hit_rate", "knee", "utility"]


# ----------------------------
# Small helpers
# ----------------------------
def _to_1d_float_array(x: Union[pd.Series, np.ndarray, Iterable[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def _nan_safe_abs_errors(y: Union[pd.Series, np.ndarray, Iterable[float]],
                         yhat: Union[pd.Series, np.ndarray, Iterable[float]]) -> np.ndarray:
    y_arr = _to_1d_float_array(y)
    yhat_arr = _to_1d_float_array(yhat)
    if y_arr.shape[0] != yhat_arr.shape[0]:
        raise ValueError(f"y and yhat must have the same length. Got {len(y_arr)} vs {len(yhat_arr)}.")
    mask = np.isfinite(y_arr) & np.isfinite(yhat_arr)
    return np.abs(y_arr[mask] - yhat_arr[mask])


def _validate_tau(tau: float) -> float:
    if not np.isfinite(tau):
        raise ValueError(f"tau must be finite. Got {tau}.")
    if tau < 0:
        raise ValueError(f"tau must be >= 0. Got {tau}.")
    return float(tau)


def _quantile(x: np.ndarray, q: float) -> float:
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
        raise ValueError(f"grid_quantiles must satisfy 0 <= q_lo <= q_hi <= 1. Got {grid_quantiles}.")

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
    """
    Compute HR@τ: fraction of observations whose absolute error is within τ.
    """
    tau = _validate_tau(tau)
    abs_errors = _nan_safe_abs_errors(y, yhat)
    if abs_errors.size == 0:
        return np.nan
    return float(np.mean(abs_errors <= tau))


# ----------------------------
# τ estimation: global
# ----------------------------
@dataclass(frozen=True)
class TauEstimate:
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
    """
    Estimate a global tolerance τ from residuals (no exogenous data).

    Parameters
    ----------
    method:
        - "target_hit_rate": τ = quantile(|e|, target_hit_rate)
        - "knee": choose τ at diminishing returns point on HR(τ)
        - "utility": maximize HR(τ) - λ * τ/τ_max

    tau_floor:
        Minimum allowed τ (>=0).
    tau_cap:
        Optional maximum allowed τ.
    """
    abs_errors = _nan_safe_abs_errors(y, yhat)
    n = int(abs_errors.size)

    if n == 0:
        return TauEstimate(tau=np.nan, method=str(method), n=0, diagnostics={"reason": "no_finite_pairs"})

    tau_floor = _validate_tau(tau_floor)
    if tau_cap is not None:
        tau_cap = _validate_tau(tau_cap)

    if method == "target_hit_rate":
        if not (0.0 < target_hit_rate <= 1.0):
            raise ValueError(f"target_hit_rate must be in (0, 1]. Got {target_hit_rate}.")
        tau = _quantile(abs_errors, target_hit_rate)

        if np.isfinite(tau):
            tau = max(tau, tau_floor)
            if tau_cap is not None:
                tau = min(tau, tau_cap)

        diag = {
            "target_hit_rate": float(target_hit_rate),
            "achieved_hr_calibration": float(np.mean(abs_errors <= tau)) if np.isfinite(tau) else np.nan,
            "abs_error_quantile_used": float(target_hit_rate),
            "tau_floor": float(tau_floor),
            "tau_cap": float(tau_cap) if tau_cap is not None else None,
        }
        return TauEstimate(tau=float(tau), method="target_hit_rate", n=n, diagnostics=diag)

    # For knee / utility, we need a grid
    tau_grid = _make_tau_grid(abs_errors, grid=grid, grid_size=grid_size, grid_quantiles=grid_quantiles)
    if tau_grid.size == 0:
        return TauEstimate(tau=np.nan, method=str(method), n=n, diagnostics={"reason": "empty_tau_grid"})

    # HR curve on grid (monotone non-decreasing)
    # We compute HR efficiently by sorting errors and using searchsorted.
    e_sorted = np.sort(abs_errors)
    idx = np.searchsorted(e_sorted, tau_grid, side="right")
    hr_curve = idx / float(n)

    if method == "knee":
        if knee_rule == "slope_threshold":
            # Choose smallest tau where marginal gain per tau drops below threshold.
            # Use finite differences on the HR curve.
            d_tau = np.diff(tau_grid)
            d_hr = np.diff(hr_curve)
            # Avoid division by zero; where d_tau==0, set slope to +inf (won't trigger threshold)
            slope = np.where(d_tau > 0, d_hr / d_tau, np.inf)

            # Find first index where slope < threshold, pick corresponding tau
            # We pick i+1 because slope is between i and i+1.
            candidates = np.where(slope < slope_threshold)[0]
            if candidates.size > 0:
                pick_i = int(candidates[0] + 1)
            else:
                # If it never drops below threshold, pick the largest tau (most coverage)
                pick_i = int(len(tau_grid) - 1)

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
            # Classic "knee" heuristic: maximize distance to the diagonal between endpoints.
            # Normalize tau to [0,1] so tau scale doesn't dominate.
            t0, t1 = float(tau_grid[0]), float(tau_grid[-1])
            if t1 > t0:
                t_norm = (tau_grid - t0) / (t1 - t0)
            else:
                t_norm = np.zeros_like(tau_grid)

            # Line from (0, hr(0)) to (1, hr(1)) in normalized coordinates.
            # Distance from each point to the line; maximize.
            x = t_norm
            yv = hr_curve
            x0, y0 = 0.0, float(hr_curve[0])
            x1, y1 = 1.0, float(hr_curve[-1])

            # If endpoints are identical, no knee
            denom = np.hypot(x1 - x0, y1 - y0)
            if denom == 0:
                pick_i = int(len(tau_grid) // 2)
            else:
                # Perpendicular distance from point to line
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

        # Apply guards
        tau = max(tau, tau_floor)
        if tau_cap is not None:
            tau = min(tau, tau_cap)

        return TauEstimate(tau=float(tau), method="knee", n=n, diagnostics=diag)

    if method == "utility":
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be >= 0. Got {lambda_}.")

        if tau_max is None:
            # Robust default: use high quantile of abs errors (avoids max outlier)
            tau_max_val = _quantile(abs_errors, 0.99)
        else:
            tau_max_val = float(tau_max)

        if not np.isfinite(tau_max_val) or tau_max_val <= 0:
            # Fall back: last grid point (should be finite and >=0)
            tau_max_val = float(tau_grid[-1]) if np.isfinite(tau_grid[-1]) and tau_grid[-1] > 0 else 1.0

        utility = hr_curve - float(lambda_) * (tau_grid / tau_max_val)
        pick_i = int(np.argmax(utility))

        tau = float(tau_grid[pick_i])
        hr_pick = float(hr_curve[pick_i])
        u_pick = float(utility[pick_i])

        # Apply guards
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


# ----------------------------
# τ estimation: entity-level
# ----------------------------
def estimate_entity_tau(
    df: pd.DataFrame,
    *,
    entity_col: str,
    y_col: str,
    yhat_col: str,
    method: TauMethod = "target_hit_rate",
    min_n: int = 30,
    # pass-through kwargs for estimate_tau
    estimate_kwargs: Optional[Mapping[str, Any]] = None,
    # optional global cap derived from all entities
    cap_with_global: bool = False,
    global_cap_quantile: float = 0.99,
    # include useful diagnostics
    include_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Estimate τ per entity from residuals.

    Parameters
    ----------
    min_n:
        Minimum number of finite (y,yhat) pairs required to estimate entity τ.
        Entities below min_n get tau = NaN.
    cap_with_global:
        If True, cap each entity τ by a global τ cap computed as
        quantile(|e|, global_cap_quantile) over the entire dataframe.
    """
    if estimate_kwargs is None:
        estimate_kwargs = {}

    required = {entity_col, y_col, yhat_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if min_n < 1:
        raise ValueError(f"min_n must be >= 1. Got {min_n}.")

    # Global cap, if requested
    global_cap = None
    if cap_with_global:
        abs_errors_all = _nan_safe_abs_errors(df[y_col], df[yhat_col])
        global_cap = _quantile(abs_errors_all, global_cap_quantile)
        if not np.isfinite(global_cap):
            global_cap = None

    rows: list[dict[str, Any]] = []

    # Group and estimate
    for ent, g in df.groupby(entity_col, dropna=False):
        abs_errors = _nan_safe_abs_errors(g[y_col], g[yhat_col])
        n = int(abs_errors.size)

        if n < min_n:
            row = {
                entity_col: ent,
                "tau": np.nan,
                "n": n,
                "method": method,
                "reason": f"min_n_not_met(<{min_n})",
            }
            rows.append(row)
            continue

        # Estimate using the shared estimator (by passing vectors)
        est = estimate_tau(
            y=g[y_col],
            yhat=g[yhat_col],
            method=method,
            **dict(estimate_kwargs),
        )

        tau_val = est.tau
        if global_cap is not None and np.isfinite(tau_val):
            tau_val = float(min(tau_val, global_cap))

        row = {
            entity_col: ent,
            "tau": tau_val,
            "n": est.n,
            "method": est.method,
        }

        if include_diagnostics:
            # flatten a few useful diagnostics; keep full dict too
            diag = dict(est.diagnostics or {})
            row["diagnostics"] = diag
            row["achieved_hr_calibration"] = diag.get("achieved_hr_calibration", diag.get("picked_hr_calibration"))
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

    # Consistent ordering
    base_cols = [entity_col, "tau", "n", "method"]
    extra_cols = [c for c in out.columns if c not in base_cols]
    out = out[base_cols + extra_cols]

    return out


# ----------------------------
# Convenience: compute HR@τ with auto τ
# ----------------------------
def hr_auto_tau(
    y: Union[pd.Series, np.ndarray, Iterable[float]],
    yhat: Union[pd.Series, np.ndarray, Iterable[float]],
    method: TauMethod = "target_hit_rate",
    **estimate_kwargs: Any,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper: estimate τ from residuals, then compute HR@τ.

    Returns (hr, tau, diagnostics)
    """
    est = estimate_tau(y=y, yhat=yhat, method=method, **estimate_kwargs)
    if not np.isfinite(est.tau):
        return (np.nan, np.nan, dict(est.diagnostics or {}))
    hr = hr_at_tau(y, yhat, est.tau)
    return (float(hr), float(est.tau), dict(est.diagnostics or {}))
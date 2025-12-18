"""
eb_evaluation.dataframe.tolerance

Data-driven tolerance (tau, τ) selection utilities for HR@τ
(hit-rate within threshold).

Theme:
- Like cost_ratio.py estimates R = cu/co by balancing under- vs over-cost,
  this module estimates τ from *historical residuals only*
  (no exogenous data, no model assumptions).
- Supports global τ and entity-level τ, with optional caps and guards
  to prevent tolerance inflation.

Core idea:
    HR(τ) = mean( |y - yhat| <= τ )

τ represents an *acceptability band*:
the maximum error magnitude considered operationally acceptable.

τ can be selected automatically from residuals via:
- target_hit_rate:
    τ = quantile(|e|, h), enforcing a readiness standard
- knee:
    τ at the point of diminishing returns in HR(τ)
- utility:
    τ maximizing HR(τ) − λ · (τ / τ_max), trading coverage against tolerance width

Governance notes:
- τ should be calibrated on a calibration window and reported on holdout data.
- Entity-level τ should be capped (optionally) by a global tolerance to
  avoid masking poor performance.

All functions are deterministic given inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Single source of truth for the HR@τ metric math
from ebmetrics.metrics.service import hr_at_tau as _hr_at_tau_core

TauMethod = Literal["target_hit_rate", "knee", "utility"]


# ----------------------------
# Small helpers
# ----------------------------
def _to_1d_float_array(x: Union[pd.Series, np.ndarray, Iterable[float]]) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _nan_safe_abs_errors(
    y: Union[pd.Series, np.ndarray, Iterable[float]],
    yhat: Union[pd.Series, np.ndarray, Iterable[float]],
) -> np.ndarray:
    y_arr = _to_1d_float_array(y)
    yhat_arr = _to_1d_float_array(yhat)
    if y_arr.shape[0] != yhat_arr.shape[0]:
        raise ValueError(
            f"y and yhat must have the same length. Got {len(y_arr)} vs {len(yhat_arr)}."
        )
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
    """
    Compute HR@τ: fraction of observations whose absolute error is within τ.

    Notes
    -----
    This is an evaluation-friendly wrapper around the core metric
    implementation in ebmetrics.metrics.service.hr_at_tau.

    We drop non-finite (y, yhat) pairs prior to delegating. If no finite
    pairs remain, returns np.nan.
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

    # For knee / utility, we need a grid
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
            if candidates.size > 0:
                pick_i = int(candidates[0] + 1)
            else:
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
            t0, t1 = float(tau_grid[0]), float(tau_grid[-1])
            if t1 > t0:
                t_norm = (tau_grid - t0) / (t1 - t0)
            else:
                t_norm = np.zeros_like(tau_grid)

            x = t_norm
            yv = hr_curve
            x0, y0 = 0.0, float(hr_curve[0])
            x1, y1 = 1.0, float(hr_curve[-1])

            denom = np.hypot(x1 - x0, y1 - y0)
            if denom == 0:
                pick_i = int(len(tau_grid) // 2)
            else:
                dist = (
                    np.abs((y1 - y0) * x - (x1 - x0) * yv + x1 * y0 - y1 * x0) / denom
                )
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
    estimate_kwargs: Optional[Mapping[str, Any]] = None,
    cap_with_global: bool = False,
    global_cap_quantile: float = 0.99,
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
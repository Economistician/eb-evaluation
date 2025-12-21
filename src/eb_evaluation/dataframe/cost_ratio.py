from __future__ import annotations

"""
Entity-level cost ratio estimation (DataFrame utilities).

This module provides DataFrame-oriented utilities for estimating an entity-specific
underbuild-to-overbuild cost ratio from historical forecast error patterns.

The primary use case is to calibrate a per-entity ratio

$$
    R_e = \frac{c_{u,e}}{c_o}
$$

that can later be used in entity-aware evaluation workflows (e.g., computing cost-weighted
metrics with entity-specific asymmetry parameters).
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd


__all__ = ["estimate_entity_R_from_balance"]


def estimate_entity_R_from_balance(
    df: pd.DataFrame,
    entity_col: str,
    y_true_col: str,
    y_pred_col: str,
    ratios: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: float = 1.0,
    sample_weight_col: Optional[str] = None,
) -> pd.DataFrame:
    r"""
    Estimate an entity-level cost ratio via a cost-balance grid search.

    This function estimates a per-entity underbuild-to-overbuild cost ratio

    $$
        R_e = \frac{c_{u,e}}{c_o}
    $$

    by searching over a user-provided grid of candidate ratios. For each entity, define
    shortfall and overbuild per interval $i$:

    $$
        s_i = \max(0, y_i - \hat{y}_i)
    $$

    $$
        o_i = \max(0, \hat{y}_i - y_i)
    $$

    For each candidate ratio $R \in \mathcal{R}$, set the implied underbuild cost coefficient:

    $$
        c_u(R) = R \cdot c_o
    $$

    Then compute weighted total underbuild and overbuild costs:

    $$
        C_u(R) = \sum_i w_i \; c_u(R) \; s_i
    $$

    $$
        C_o = \sum_i w_i \; c_o \; o_i
    $$

    The selected ratio is the value that minimizes the absolute imbalance between these totals:

    $$
        R_e = \arg\min_{R \in \mathcal{R}} \left| C_u(R) - C_o \right|
    $$

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing an entity identifier, actuals, and forecasts (and optionally weights).
    entity_col : str
        Column in ``df`` identifying the entity (e.g., ``"item"``, ``"sku"``, ``"location"``).
    y_true_col : str
        Column containing realized demand values $y_i$. Must be non-negative.
    y_pred_col : str
        Column containing baseline forecast values $\hat{y}_i$. Must be non-negative.
    ratios : Sequence[float], default=(0.5, 1.0, 2.0, 3.0)
        Candidate ratio grid $\mathcal{R}$. Values must be strictly positive.
    co : float, default=1.0
        Overbuild (excess) cost coefficient $c_o$. Must be strictly positive.
    sample_weight_col : str | None, default=None
        Optional column containing non-negative sample weights $w_i$. If ``None``, all rows are
        equally weighted within each entity.

    Returns
    -------
    pandas.DataFrame
        One row per entity with columns:

        - ``entity_col`` : entity identifier
        - ``R`` : chosen ratio $R_e$
        - ``cu`` : implied underbuild cost $c_{u,e} = R_e \cdot c_o$
        - ``co`` : provided overbuild cost coefficient $c_o$
        - ``under_cost`` : $C_u(R_e)$
        - ``over_cost`` : $C_o$
        - ``diff`` : $\left|C_u(R_e) - C_o\right|$

    Raises
    ------
    KeyError
        If required columns are missing from ``df``.
    ValueError
        If ``ratios`` is empty or contains non-positive values, if ``co <= 0``,
        if any entity contains negative values, or if sample weights are negative.

    Notes
    -----
    - This is a calibration helper for cost-ratio tuning. It does not infer economics directly
      (margin, food cost, labor, etc.); it uses the observed error pattern under an assumed $c_o$.
    - If an entity has zero error across all intervals (no shortfall and no overbuild), the ratio
      in ``ratios`` closest to 1.0 is selected and under/over costs are returned as zero.
    - Entities with strongly skewed error patterns will often select ratios at the edges of the
      provided grid; widen the grid if needed.

    """
    required = {entity_col, y_true_col, y_pred_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df: {sorted(missing)}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col '{sample_weight_col}' not found in df")

    ratios_arr = np.asarray(list(ratios), dtype=float)
    if ratios_arr.ndim != 1 or ratios_arr.size == 0 or np.any(ratios_arr <= 0):
        raise ValueError("ratios must be a non-empty 1D sequence of positive floats.")

    if co <= 0:
        raise ValueError("co must be strictly positive.")

    results: list[dict] = []

    grouped = df.groupby(entity_col, sort=False)

    for entity_id, g in grouped:
        y_true = g[y_true_col].to_numpy(dtype=float)
        y_pred = g[y_pred_col].to_numpy(dtype=float)

        if sample_weight_col is not None:
            w = g[sample_weight_col].to_numpy(dtype=float)
        else:
            w = np.ones_like(y_true, dtype=float)

        # Basic validation per entity
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"For entity {entity_id!r}, y_true and y_pred have different shapes: "
                f"{y_true.shape} vs {y_pred.shape}"
            )
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError(
                f"For entity {entity_id!r}, y_true and y_pred must be non-negative."
            )
        if np.any(w < 0):
            raise ValueError(
                f"For entity {entity_id!r}, sample weights must be non-negative."
            )

        shortfall = np.maximum(0.0, y_true - y_pred)
        overbuild = np.maximum(0.0, y_pred - y_true)

        # Degenerate case: no error at all for this entity
        if np.all(shortfall == 0.0) and np.all(overbuild == 0.0):
            idx = int(np.argmin(np.abs(ratios_arr - 1.0)))
            R_e = float(ratios_arr[idx])
            cu_e = R_e * co
            results.append(
                {
                    entity_col: entity_id,
                    "R": R_e,
                    "cu": cu_e,
                    "co": co,
                    "under_cost": 0.0,
                    "over_cost": 0.0,
                    "diff": 0.0,
                }
            )
            continue

        best_R: float | None = None
        best_cu: float | None = None
        best_under_cost: float | None = None
        best_over_cost: float | None = None
        best_diff: float | None = None

        for R in ratios_arr:
            cu_val = float(R * co)

            under_cost = float(np.sum(w * cu_val * shortfall))
            over_cost = float(np.sum(w * co * overbuild))
            diff = abs(under_cost - over_cost)

            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_R = float(R)
                best_cu = cu_val
                best_under_cost = under_cost
                best_over_cost = over_cost

        results.append(
            {
                entity_col: entity_id,
                "R": best_R,
                "cu": best_cu,
                "co": co,
                "under_cost": best_under_cost,
                "over_cost": best_over_cost,
                "diff": best_diff,
            }
        )

    return pd.DataFrame(results)

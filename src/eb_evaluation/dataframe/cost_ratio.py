from __future__ import annotations

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
    """
    Estimate an entity-level cost ratio R_e = cu_e / co for each entity,
    using a simple "cost balance" method over a grid of candidate ratios.

    For each entity:
        1. Take all historical intervals for that entity.
        2. For each candidate R in `ratios`:
            - set cu = R * co
            - compute total underbuild cost and overbuild cost:
                  under_cost(R) = sum(w_i * cu * shortfall_i)
                  over_cost(R)  = sum(w_i * co * overbuild_i)
            - measure the imbalance:
                  diff(R) = |under_cost(R) - over_cost(R)|
        3. Choose the R that MINIMIZES diff(R).
        4. Report R_e, cu_e = R_e * co, and the under/over costs at that R_e.

    This is a *data-driven helper* for cost-ratio tuning. It does NOT try to
    infer economics (margin, food cost, etc.) directly. Instead, it uses the
    historical error pattern for each entity under an assumed overbuild cost
    profile `co`.

    Typical usage:
        - Use a small grid like (0.5, 1.0, 2.0, 3.0) for exploratory work.
        - Use the resulting R_e as input into `evaluate_panel_with_entity_R`
          to compute entity-aware CWSL / FRS.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing an entity identifier, actuals, and forecasts.

    entity_col : str
        Column in `df` identifying the entity (e.g., "item", "sku",
        "product", "line", "location").

    y_true_col : str
        Column containing actual demand. Must be non-negative.

    y_pred_col : str
        Column containing forecasted demand. Must be non-negative.

    ratios : sequence of float, default (0.5, 1.0, 2.0, 3.0)
        Candidate R values to search over for each entity. These must be
        strictly positive. A typical starting grid is something like
        (0.5, 1.0, 2.0, 3.0).

    co : float, default 1.0
        Overbuild (excess) cost per unit, assumed scalar and common
        across entities for this estimation method. Must be > 0.

    sample_weight_col : str or None, default None
        Optional column of non-negative sample weights per row. If None,
        all rows are equally weighted within each entity.

    Returns
    -------
    pandas.DataFrame
        One row per entity with columns:

            - entity_col   (entity identifier)
            - R            (chosen cost ratio)
            - cu           (shortfall cost = R * co)
            - co           (overbuild cost, scalar input)
            - under_cost   (total underbuild cost at chosen R)
            - over_cost    (total overbuild cost at chosen R)
            - diff         (|under_cost - over_cost| at chosen R)

    Notes
    -----
    - If an entity has literally zero error (no shortfall and no overbuild)
      across all its intervals, we pick the R in `ratios` that is closest to 1.0,
      and under_cost = over_cost = diff = 0.
    - Entities with strongly skewed error patterns (mostly short or mostly long)
      will tend to land on the edges of the `ratios` grid.
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
            # Pick the R in the grid closest to 1.0
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
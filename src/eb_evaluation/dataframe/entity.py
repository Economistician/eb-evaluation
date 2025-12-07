from __future__ import annotations

from typing import Dict

import numpy as np
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


def evaluate_panel_with_entity_R(
    df: pd.DataFrame,
    entity_R: pd.DataFrame,
    *,
    entity_col: str = "entity",
    y_true_col: str = "actual_qty",
    y_pred_col: str = "forecast_qty",
    R_col: str = "R",
    co_col: str = "co",
    tau: float = 2.0,
    sample_weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate a panel of entity–interval data using *entity-level* cost ratios R_e.

    This helper is designed to work with the output of
    ``estimate_entity_R_from_balance(...)``, which returns one row per entity with
    columns like:

        - entity_col
        - R      (cost ratio, cu = R * co)
        - cu     (optional, shortfall cost)
        - co     (overbuild cost)
        - under_cost, over_cost, diff (diagnostics)

    For each entity, this function:

        1. Joins the entity-level R and co onto the interval-level panel.
        2. Uses cu_i = R_e * co_e (constant per entity) and co_i = co_e for all
           rows belonging to that entity.
        3. Computes the full CWSL metric suite (CWSL, NSL, UD, wMAPE, HR@tau,
           FRS, MAE, RMSE, MAPE) using those entity-specific cu/co values.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel of interval-level data with at least [entity_col, y_true_col,
        y_pred_col] and optionally sample_weight_col.

    entity_R : pandas.DataFrame
        Table with one row per entity containing at least [entity_col, R_col,
        co_col]. Typically the output of ``estimate_entity_R_from_balance``.

    entity_col : str, default "entity"
        Column identifying the entity (e.g. "item", "sku", "store", etc.).

    y_true_col : str, default "actual_qty"
        Column containing actual demand.

    y_pred_col : str, default "forecast_qty"
        Column containing forecasted demand.

    R_col : str, default "R"
        Column in ``entity_R`` containing the cost ratio R_e.

    co_col : str, default "co"
        Column in ``entity_R`` containing the overbuild cost per unit.

    tau : float, default 2.0
        Absolute-error tolerance for HR@tau.

    sample_weight_col : str or None, default None
        Optional column of non-negative sample weights per row in ``df``.

    Returns
    -------
    pandas.DataFrame
        One row per entity with columns:

            [entity_col, "R", "cu", "co",
             "CWSL", "NSL", "UD", "wMAPE", "HR@tau",
             "FRS", "MAE", "RMSE", "MAPE"]

        If a metric is undefined for a given entity slice, the corresponding
        value is NaN (rather than raising for the entire panel).
    """
    # Validate required columns in df
    required_df = {entity_col, y_true_col, y_pred_col}
    missing_df = required_df - set(df.columns)
    if missing_df:
        raise KeyError(f"Missing required columns in df: {sorted(missing_df)}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col '{sample_weight_col}' not found in df")

    # Validate required columns in entity_R
    required_R = {entity_col, R_col, co_col}
    missing_R = required_R - set(entity_R.columns)
    if missing_R:
        raise KeyError(
            f"Missing required columns in entity_R: {sorted(missing_R)}"
        )

    # Keep only the join + cost columns we care about
    entity_costs = entity_R[[entity_col, R_col, co_col]].copy()

    # Merge entity-level R, co onto panel
    merged = df.merge(entity_costs, on=entity_col, how="inner", validate="m:1")
    if merged.empty:
        raise ValueError("After merging, no rows remain. Check entity identifiers.")

    results: list[dict] = []

    def _safe_metric(fn) -> float:
        try:
            return float(fn())
        except ValueError:
            return float("nan")

    grouped = merged.groupby(entity_col, sort=False)

    for entity_id, g in grouped:
        y_true = g[y_true_col].to_numpy(dtype=float)
        y_pred = g[y_pred_col].to_numpy(dtype=float)

        # Per-entity R and co should be constant; take the first
        R_e = float(g[R_col].iloc[0])
        co_e = float(g[co_col].iloc[0])

        if sample_weight_col is not None:
            sample_weight = g[sample_weight_col].to_numpy(dtype=float)
        else:
            sample_weight = None

        # Build per-row cu/co arrays (constant within entity)
        cu_arr = np.full_like(y_true, fill_value=R_e * co_e, dtype=float)
        co_arr = np.full_like(y_true, fill_value=co_e, dtype=float)

        row: Dict[str, float] = {
            entity_col: entity_id,
            "R": R_e,
            "cu": R_e * co_e,
            "co": co_e,
        }

        row["CWSL"] = _safe_metric(
            lambda: cwsl(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu_arr,
                co=co_arr,
                sample_weight=sample_weight,
            )
        )
        row["NSL"] = _safe_metric(
            lambda: nsl(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        )
        row["UD"] = _safe_metric(
            lambda: ud(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        )
        # wMAPE: ebmetrics.wmape has no sample_weight parameter, so call unweighted.
        row["wMAPE"] = _safe_metric(
            lambda: wmape(
                y_true=y_true,
                y_pred=y_pred,
            )
        )
        row["HR@tau"] = _safe_metric(
            lambda: hr_at_tau(
                y_true=y_true,
                y_pred=y_pred,
                tau=tau,
                sample_weight=sample_weight,
            )
        )
        row["FRS"] = _safe_metric(
            lambda: frs(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu_arr,
                co=co_arr,
                sample_weight=sample_weight,
            )
        )
        # Symmetric metrics: call unweighted to match ebmetrics signatures.
        row["MAE"] = _safe_metric(
            lambda: mae(y_true=y_true, y_pred=y_pred)
        )
        row["RMSE"] = _safe_metric(
            lambda: rmse(y_true=y_true, y_pred=y_pred)
        )
        row["MAPE"] = _safe_metric(
            lambda: mape(y_true=y_true, y_pred=y_pred)
        )

        results.append(row)

    return pd.DataFrame(results)
"""
Entity-level evaluation utilities (DataFrame helpers).

This module contains DataFrame-oriented evaluation helpers that operate on panel data:
entity-by-interval observations containing actuals and forecasts.

The primary helper evaluates each entity using entity-specific cost asymmetry parameters
(cost ratios) that are typically estimated upstream (for example, via a balance-based
estimator). The result is one row per entity containing cost-weighted and service-oriented
Electric Barometer metrics, plus familiar symmetric error metrics.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from eb_metrics.metrics import cwsl, frs, hr_at_tau, mae, mape, nsl, rmse, ud, wmape

__all__ = ["evaluate_panel_with_entity_R"]


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
    r"""
    Evaluate an entity-interval panel using entity-level cost ratios.

    This helper evaluates each entity slice of a panel using entity-specific cost
    asymmetry parameters. It is designed to pair naturally with a table that provides,
    for each entity, a cost ratio:

    $$
        R_e = \frac{c_{u,e}}{c_o}
    $$

    and an overbuild cost coefficient $c_o$. For each entity $e$, the implied shortfall
    (underbuild) cost coefficient is:

    $$
        c_{u,e} = R_e \cdot c_o
    $$

    Evaluation flow
    ---------------
    For each entity:

    1. Join the entity-level values $(R_e, c_o)$ onto all intervals for that entity.
    2. Construct per-row arrays $c_u$ and $c_o$ that are constant within the entity slice.
    3. Compute the EB metric suite (CWSL, NSL, UD, HR@$\tau$, FRS) using the entity-specific
       cost parameters, plus common symmetric metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel of interval-level data containing at least:

        - ``entity_col`` (entity identifier)
        - ``y_true_col`` (actuals)
        - ``y_pred_col`` (forecasts)

        If ``sample_weight_col`` is provided, it must also exist in ``df``.
    entity_R : pandas.DataFrame
        Table with one row per entity containing at least:

        - ``entity_col`` (entity identifier)
        - ``R_col`` (cost ratio $R_e$)
        - ``co_col`` (overbuild cost coefficient $c_o$)

        Typically produced by an upstream calibration step (for example, an entity ratio
        estimator).
    entity_col : str, default="entity"
        Column identifying the entity (for example, ``"store_id"``, ``"sku"``, ``"location"``).
    y_true_col : str, default="actual_qty"
        Column containing realized demand / actual values.
    y_pred_col : str, default="forecast_qty"
        Column containing baseline forecast values.
    R_col : str, default="R"
        Column in ``entity_R`` containing the cost ratio $R_e$.
    co_col : str, default="co"
        Column in ``entity_R`` containing the overbuild cost coefficient $c_o$.
    tau : float, default=2.0
        Absolute-error tolerance parameter for the hit-rate metric HR@$\tau$.
    sample_weight_col : str | None, default=None
        Optional column in ``df`` of non-negative sample weights.

    Returns
    -------
    pandas.DataFrame
        One row per entity with columns:

        - ``entity_col`` : entity identifier
        - ``R`` : entity ratio $R_e$
        - ``cu`` : implied underbuild cost coefficient $c_{u,e} = R_e \cdot c_o$
        - ``co`` : overbuild cost coefficient $c_o$
        - ``CWSL`` : cost-weighted service loss
        - ``NSL`` : no-shortfall level (service-oriented)
        - ``UD`` : underbuild depth
        - ``wMAPE`` : weighted mean absolute percentage error (per eb_metrics definition)
        - ``HR@tau`` : hit rate within tolerance $\tau$
        - ``FRS`` : forecast readiness score
        - ``MAE`` : mean absolute error
        - ``RMSE`` : root mean squared error
        - ``MAPE`` : mean absolute percentage error

        If a metric is undefined for a given entity slice (for example, due to a metric-specific
        validation failure), that metric value is returned as NaN for that entity.

    Raises
    ------
    KeyError
        If required columns are missing from ``df`` or ``entity_R``.
    ValueError
        If the merge between ``df`` and ``entity_R`` produces no rows (no overlapping entities).

    Notes
    -----
    - The join uses an inner merge on ``entity_col``. Entities present in ``df`` but missing
      from ``entity_R`` are dropped. This is intentional: evaluation requires cost parameters.
    - Cost arrays are constructed per entity as constants, enabling vectorized evaluation calls.
    - Some metrics in :mod:`eb_metrics.metrics` may not accept sample weights; this function
      calls those metrics unweighted to match their signatures.
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
        raise KeyError(f"Missing required columns in entity_R: {sorted(missing_R)}")

    # Keep only the join + cost columns we care about
    entity_costs = entity_R[[entity_col, R_col, co_col]].copy()

    # Merge entity-level R, co onto panel
    merged = df.merge(entity_costs, on=entity_col, how="inner", validate="m:1")
    if merged.empty:
        raise ValueError(
            "After merging df with entity_R, no rows remain. Check entity identifiers and join keys."
        )

    results: list[dict] = []

    def _safe_metric(fn: Callable[[], float]) -> float:
        """Compute a metric and return NaN if the metric raises ValueError."""
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

        row: dict[str, float] = {
            entity_col: entity_id,
            "R": R_e,
            "cu": R_e * co_e,
            "co": co_e,
        }

        row["CWSL"] = _safe_metric(
            lambda y_true=y_true,
            y_pred=y_pred,
            cu_arr=cu_arr,
            co_arr=co_arr,
            sample_weight=sample_weight: cwsl(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu_arr,
                co=co_arr,
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

        # wMAPE: eb_metrics.wmape has no sample_weight parameter, so call unweighted.
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
            cu_arr=cu_arr,
            co_arr=co_arr,
            sample_weight=sample_weight: frs(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu_arr,
                co=co_arr,
                sample_weight=sample_weight,
            )
        )

        # Symmetric metrics: call unweighted to match eb_metrics signatures.
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

from __future__ import annotations

import numpy as np
import pandas as pd

from eb_metrics.metrics import (
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
from eb_evaluation.dataframe import evaluate_panel_with_entity_R


def _build_entity_panel() -> pd.DataFrame:
    """
    Build a simple panel with two entities over multiple intervals.
    """
    rows = []

    # Entity A
    for t, y in enumerate([10, 12, 15], start=1):
        rows.append(
            {
                "entity": "A",
                "t": t,
                "actual_qty": y,
                "forecast_qty": y + 1,  # slight overbuild
            }
        )

    # Entity B
    for t, y in enumerate([10, 12, 15], start=1):
        rows.append(
            {
                "entity": "B",
                "t": t,
                "actual_qty": y,
                "forecast_qty": y - 1,  # slight shortfall
            }
        )

    return pd.DataFrame(rows)


def test_evaluate_panel_with_entity_R_basic_structure():
    """
    Structural test:
    - one row per entity
    - expected metric columns are present
    - entity-level cost columns (R, cu, co) exist
    """
    panel = _build_entity_panel()

    # Entity-level R and co (could come from estimate_entity_R_from_balance)
    entity_R = pd.DataFrame(
        {
            "entity": ["A", "B"],
            "R": [2.0, 3.0],
            "co": [1.0, 1.0],
        }
    )

    result = evaluate_panel_with_entity_R(
        df=panel,
        entity_R=entity_R,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        R_col="R",
        co_col="co",
        tau=2.0,
    )

    # One row per entity
    assert set(result["entity"]) == {"A", "B"}

    # Cost-related columns
    for col in ["R", "cu", "co"]:
        assert col in result.columns

    # Metric columns
    expected_metric_cols = [
        "CWSL",
        "NSL",
        "UD",
        "wMAPE",
        "HR@tau",
        "FRS",
        "MAE",
        "RMSE",
        "MAPE",
    ]

    for col in expected_metric_cols:
        assert col in result.columns


def test_evaluate_panel_with_entity_R_matches_direct_metrics_for_single_entity():
    """
    Correctness test:
    For a single entity, verify that the metrics computed by
    evaluate_panel_with_entity_R match direct eb_metrics calls
    using per-row cu/co derived from entity-level R and co.
    """
    panel = _build_entity_panel()

    # Restrict to entity A only
    panel_A = panel[panel["entity"] == "A"].reset_index(drop=True)

    entity_R = pd.DataFrame(
        {
            "entity": ["A"],
            "R": [2.0],
            "co": [1.5],
        }
    )

    result = evaluate_panel_with_entity_R(
        df=panel_A,
        entity_R=entity_R,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        R_col="R",
        co_col="co",
        tau=2.0,
    )

    assert len(result) == 1
    row = result.iloc[0]

    # Direct computation
    y_true = panel_A["actual_qty"].to_numpy()
    y_pred = panel_A["forecast_qty"].to_numpy()

    R_e = 2.0
    co_e = 1.5
    cu_arr = np.full_like(y_true, R_e * co_e, dtype=float)
    co_arr = np.full_like(y_true, co_e, dtype=float)

    expected = {
        "CWSL": cwsl(y_true, y_pred, cu=cu_arr, co=co_arr),
        "NSL": nsl(y_true, y_pred),
        "UD": ud(y_true, y_pred),
        "wMAPE": wmape(y_true, y_pred),
        "HR@tau": hr_at_tau(y_true, y_pred, tau=2.0),
        "FRS": frs(y_true, y_pred, cu=cu_arr, co=co_arr),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

    for name, val in expected.items():
        assert np.isclose(row[name], val), f"Mismatch for metric '{name}'"


def test_evaluate_panel_with_entity_R_respects_R_cost_ratio_behavior():
    """
    Behavioral test:
    Entity with higher R (more expensive shortfalls relative to overbuild)
    and a shortfall pattern should have higher CWSL than one with lower R.
    """
    # Build a panel where both entities have shortfalls, but one has higher R
    rows = []
    for entity, R in [("A", 1.5), ("B", 3.0)]:
        for t, y in enumerate([10, 12, 15], start=1):
            rows.append(
                {
                    "entity": entity,
                    "t": t,
                    "actual_qty": y,
                    "forecast_qty": y - 2,  # consistent shortfall
                }
            )

    panel = pd.DataFrame(rows)

    entity_R = pd.DataFrame(
        {
            "entity": ["A", "B"],
            "R": [1.5, 3.0],
            "co": [1.0, 1.0],
        }
    )

    result = evaluate_panel_with_entity_R(
        df=panel,
        entity_R=entity_R,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        R_col="R",
        co_col="co",
        tau=2.0,
    )

    cwsl_A = float(result.loc[result["entity"] == "A", "CWSL"].iloc[0])
    cwsl_B = float(result.loc[result["entity"] == "B", "CWSL"].iloc[0])

    assert cwsl_B > cwsl_A, (
        "Entity with higher shortfall cost ratio R and consistent shortfalls "
        "should have higher CWSL."
    )

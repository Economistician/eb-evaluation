from __future__ import annotations

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
from eb_evaluation.dataframe import evaluate_groups_df


def test_evaluate_groups_df_produces_expected_columns():
    """
    Basic structural test:
    Ensure the output DataFrame contains all expected metric columns
    and exactly one row per group.
    """
    df = pd.DataFrame(
        {
            "store_id": [1, 1, 2, 2],
            "item_id": ["A", "A", "A", "A"],
            "actual_qty": [10, 12, 9, 11],
            "forecast_qty": [11, 13, 8, 10],
        }
    )

    out = evaluate_groups_df(
        df=df,
        group_cols=["store_id", "item_id"],
    )

    expected_cols = {
        "store_id",
        "item_id",
        "CWSL",
        "NSL",
        "UD",
        "wMAPE",
        "HR@tau",
        "FRS",
        "MAE",
        "RMSE",
        "MAPE",
    }

    assert expected_cols.issubset(out.columns)
    assert len(out) == 2  # store 1 + store 2


def test_evaluate_groups_df_matches_direct_calculation_for_one_group():
    """
    Deep correctness test:
    For one specific group, compute all metrics directly using ebmetrics
    and ensure evaluate_groups_df returns identical values.
    """
    df = pd.DataFrame(
        {
            "store_id": [1, 1, 1, 2, 2, 2],
            "item_id": ["A", "A", "B", "A", "A", "B"],
            "actual_qty": [10, 12, 8, 9, 11, 7],
            "forecast_qty": [9, 15, 7, 10, 10, 8],
        }
    )

    cu = 2.0
    co = 1.0
    tau = 2.0

    out = evaluate_groups_df(
        df=df,
        group_cols=["store_id", "item_id"],
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # Pick group (1, A)
    g = df[(df["store_id"] == 1) & (df["item_id"] == "A")]
    y_true = g["actual_qty"].to_numpy()
    y_pred = g["forecast_qty"].to_numpy()

    expected = {
        "CWSL": cwsl(y_true, y_pred, cu=cu, co=co),
        "NSL": nsl(y_true, y_pred),
        "UD": ud(y_true, y_pred),
        "wMAPE": wmape(y_true, y_pred),
        "HR@tau": hr_at_tau(y_true, y_pred, tau=tau),
        "FRS": frs(y_true, y_pred, cu=cu, co=co),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

    row = out[(out["store_id"] == 1) & (out["item_id"] == "A")].iloc[0]

    for metric, val in expected.items():
        assert np.isclose(row[metric], val)


def test_evaluate_groups_df_supports_per_row_cu_co_and_changes_results():
    """
    Behavioral test:
    - Ensure per-row cu/co works.
    - Ensure a group with expensive shortfalls has higher CWSL.
    """

    rows = []

    # Entity A: only overbuild, cheap shortfall
    for t, y in enumerate([10, 12, 15], start=1):
        rows.append(
            {
                "entity": "A",
                "actual_qty": y,
                "forecast_qty": y + 2,
                "cu": 1.0,
                "co": 1.0,
            }
        )

    # Entity B: only shortfall, expensive shortfall
    for t, y in enumerate([10, 12, 15], start=1):
        rows.append(
            {
                "entity": "B",
                "actual_qty": y,
                "forecast_qty": y - 2,
                "cu": 3.0,
                "co": 1.0,
            }
        )

    df = pd.DataFrame(rows)

    out = evaluate_groups_df(
        df=df,
        group_cols=["entity"],
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu="cu",  # per-row cost
        co="co",
        tau=2.0,
    )

    cwsl_A = float(out.loc[out["entity"] == "A", "CWSL"].iloc[0])
    cwsl_B = float(out.loc[out["entity"] == "B", "CWSL"].iloc[0])

    assert cwsl_B > cwsl_A, "More expensive shortfalls should increase CWSL"
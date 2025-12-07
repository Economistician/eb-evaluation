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
)
from eb_evaluation.dataframe import evaluate_hierarchy_df


def _build_sample_df() -> pd.DataFrame:
    """
    Small helper to build a simple hierarchical dataset with:
      - 2 stores
      - 2 items per store
      - multiple intervals
    """
    return pd.DataFrame(
        {
            "store_id": [1, 1, 1, 2, 2, 2],
            "item_id": ["A", "A", "B", "A", "A", "B"],
            "actual_qty": [10, 12, 8, 9, 11, 7],
            "forecast_qty": [9, 15, 7, 10, 10, 8],
        }
    )


def test_evaluate_hierarchy_df_returns_all_levels():
    """
    Structural test:
    Ensure the function returns a dictionary with one entry per level
    and that each DataFrame contains the core metric columns.
    """
    df = _build_sample_df()

    levels = {
        "overall": [],
        "by_store": ["store_id"],
        "by_item": ["item_id"],
        "by_store_item": ["store_id", "item_id"],
    }

    cu = 2.0
    co = 1.0
    tau = 2.0

    result = evaluate_hierarchy_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # We expect all level keys to be present
    assert set(result.keys()) == set(levels.keys())

    expected_metric_cols = {
        "n_intervals",
        "total_demand",
        "cwsl",
        "nsl",
        "ud",
        "wmape",
        "hr_at_tau",
        "frs",
    }

    for level_name, level_df in result.items():
        assert expected_metric_cols.issubset(level_df.columns), (
            f"Missing expected metric columns for level '{level_name}'"
        )


def test_evaluate_hierarchy_df_overall_level_matches_direct_metrics():
    """
    Correctness test:
    For the 'overall' level (no grouping), the metrics should match
    direct calls to ebmetrics on the full DataFrame.
    """
    df = _build_sample_df()

    levels = {"overall": []}
    cu = 2.0
    co = 1.0
    tau = 2.0

    result = evaluate_hierarchy_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    overall_df = result["overall"]
    assert len(overall_df) == 1

    y_true = df["actual_qty"].to_numpy()
    y_pred = df["forecast_qty"].to_numpy()

    expected = {
        "cwsl": cwsl(y_true, y_pred, cu=cu, co=co),
        "nsl": nsl(y_true, y_pred),
        "ud": ud(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "hr_at_tau": hr_at_tau(y_true, y_pred, tau=tau),
        "frs": frs(y_true, y_pred, cu=cu, co=co),
        "n_intervals": len(df),
        "total_demand": float(df["actual_qty"].sum()),
    }

    row = overall_df.iloc[0]
    for name, val in expected.items():
        assert np.isclose(row[name], val), f"Mismatch for metric '{name}'"


def test_evaluate_hierarchy_df_group_level_matches_direct_metrics_for_store():
    """
    Correctness test:
    For a specific group (e.g., store_id == 1), metrics at the 'by_store'
    level should match direct ebmetrics calls computed on that slice.
    """
    df = _build_sample_df()

    levels = {"by_store": ["store_id"]}
    cu = 2.0
    co = 1.0
    tau = 2.0

    result = evaluate_hierarchy_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    by_store = result["by_store"]

    # Check we have both stores
    assert set(by_store["store_id"]) == {1, 2}

    # Focus on store 1
    g = df[df["store_id"] == 1]
    y_true = g["actual_qty"].to_numpy()
    y_pred = g["forecast_qty"].to_numpy()

    expected = {
        "cwsl": cwsl(y_true, y_pred, cu=cu, co=co),
        "nsl": nsl(y_true, y_pred),
        "ud": ud(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "hr_at_tau": hr_at_tau(y_true, y_pred, tau=tau),
        "frs": frs(y_true, y_pred, cu=cu, co=co),
        "n_intervals": len(g),
        "total_demand": float(g["actual_qty"].sum()),
    }

    row = by_store[by_store["store_id"] == 1].iloc[0]
    for name, val in expected.items():
        assert np.isclose(row[name], val), f"Mismatch for metric '{name}' on store 1"
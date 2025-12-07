from __future__ import annotations

import numpy as np
import pandas as pd

from ebmetrics.metrics import cwsl
from eb_evaluation.dataframe import compute_cwsl_df


def test_compute_cwsl_df_scalar_cu_co_matches_core():
    df = pd.DataFrame(
        {
            "actual": [10, 12, 8],
            "forecast": [9, 15, 7],
        }
    )

    cu = 2.0
    co = 1.0

    direct = cwsl(
        y_true=df["actual"].to_numpy(),
        y_pred=df["forecast"].to_numpy(),
        cu=cu,
        co=co,
    )

    via_df = compute_cwsl_df(
        df=df,
        y_true_col="actual",
        y_pred_col="forecast",
        cu=cu,
        co=co,
    )

    assert np.isclose(direct, via_df)


def test_compute_cwsl_df_column_cu_co_matches_core():
    df = pd.DataFrame(
        {
            "actual": [10, 12, 8],
            "forecast": [9, 15, 7],
            "cu_col": [2.0, 2.0, 2.0],
            "co_col": [1.0, 1.0, 1.0],
        }
    )

    direct = cwsl(
        y_true=df["actual"].to_numpy(),
        y_pred=df["forecast"].to_numpy(),
        cu=df["cu_col"].to_numpy(),
        co=df["co_col"].to_numpy(),
    )

    via_df = compute_cwsl_df(
        df=df,
        y_true_col="actual",
        y_pred_col="forecast",
        cu="cu_col",
        co="co_col",
    )

    assert np.isclose(direct, via_df)


def test_compute_cwsl_df_with_sample_weight_matches_core():
    df = pd.DataFrame(
        {
            "actual": [10, 12, 8],
            "forecast": [9, 15, 7],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    cu = 2.0
    co = 1.0

    direct = cwsl(
        y_true=df["actual"].to_numpy(),
        y_pred=df["forecast"].to_numpy(),
        cu=cu,
        co=co,
        sample_weight=df["weight"].to_numpy(),
    )

    via_df = compute_cwsl_df(
        df=df,
        y_true_col="actual",
        y_pred_col="forecast",
        cu=cu,
        co=co,
        sample_weight_col="weight",
    )

    assert np.isclose(direct, via_df)
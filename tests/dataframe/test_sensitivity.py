from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_metrics.metrics import cwsl_sensitivity
from eb_evaluation.dataframe import compute_cwsl_sensitivity_df


def _skip_if_no_eb_optimization():
    try:
        import eb_optimization  # noqa: F401
    except Exception:
        pytest.skip("eb-optimization is required for compute_cwsl_sensitivity_df()")


def test_compute_cwsl_sensitivity_df_matches_core_function():
    _skip_if_no_eb_optimization()

    df = pd.DataFrame(
        {
            "actual": [10.0, 12.0, 8.0],
            "forecast": [9.0, 15.0, 7.0],
        }
    )

    y_true = df["actual"].to_numpy()
    y_pred = df["forecast"].to_numpy()
    R_list = [0.5, 1.0, 2.0, 3.0]
    co = 1.0

    core = cwsl_sensitivity(
        y_true=y_true,
        y_pred=y_pred,
        R_list=R_list,
        co=co,
        sample_weight=None,
    )

    out = compute_cwsl_sensitivity_df(
        df=df,
        y_true_col="actual",
        y_pred_col="forecast",
        R_list=R_list,
        co=co,
        sample_weight_col=None,
    )

    assert list(out["R"].astype(float)) == [float(r) for r in core.keys()]
    for r, cwsl_val in core.items():
        row = float(out.loc[out["R"] == float(r), "CWSL"].iloc[0])
        assert np.isclose(row, float(cwsl_val))


def test_compute_cwsl_sensitivity_df_supports_per_row_co_and_weights():
    _skip_if_no_eb_optimization()

    df = pd.DataFrame(
        {
            "actual": [10.0, 12.0, 8.0],
            "forecast": [9.0, 15.0, 7.0],
            "co_col": [1.0, 2.0, 1.5],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    y_true = df["actual"].to_numpy()
    y_pred = df["forecast"].to_numpy()
    co_arr = df["co_col"].to_numpy()
    w = df["weight"].to_numpy()

    R_list = [0.5, 1.0, 2.0]

    core = cwsl_sensitivity(
        y_true=y_true,
        y_pred=y_pred,
        R_list=R_list,
        co=co_arr,
        sample_weight=w,
    )

    out = compute_cwsl_sensitivity_df(
        df=df,
        y_true_col="actual",
        y_pred_col="forecast",
        R_list=R_list,
        co="co_col",
        sample_weight_col="weight",
    )

    assert set(out["R"].astype(float)) == {float(r) for r in core.keys()}
    for r, cwsl_val in core.items():
        row = float(out.loc[out["R"] == float(r), "CWSL"].iloc[0])
        assert np.isclose(row, float(cwsl_val))


def test_compute_cwsl_sensitivity_df_filters_non_positive_R_and_raises_if_none_valid():
    _skip_if_no_eb_optimization()

    df = pd.DataFrame(
        {
            "actual": [10.0, 12.0],
            "forecast": [9.0, 11.0],
        }
    )

    out = compute_cwsl_sensitivity_df(
        df=df,
        y_true_col="actual",
        y_pred_col="forecast",
        R_list=[-1.0, 0.0, 1.0, 2.0],
        co=1.0,
        sample_weight_col=None,
    )

    assert set(out["R"].astype(float)) == {1.0, 2.0}

    with pytest.raises(ValueError):
        compute_cwsl_sensitivity_df(
            df=df,
            y_true_col="actual",
            y_pred_col="forecast",
            R_list=[-2.0, 0.0],
            co=1.0,
            sample_weight_col=None,
        )
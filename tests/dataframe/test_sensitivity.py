from __future__ import annotations

import numpy as np
import pandas as pd

from ebmetrics.metrics import cwsl_sensitivity
from eb_evaluation.dataframe import compute_cwsl_sensitivity_df


def test_compute_cwsl_sensitivity_df_matches_core_function():
    """
    Basic correctness:

    Ensure that compute_cwsl_sensitivity_df returns the same CWSL values
    as the core ebmetrics.metrics.cwsl_sensitivity helper when using:

        - scalar co
        - a simple R_list
    """
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

    # Same R values and in the same order
    assert list(out["R"].astype(float)) == [float(r) for r in core.keys()]

    # CWSL values should match (within floating tolerance)
    for r, cwsl_val in core.items():
        row = out.loc[out["R"] == float(r), "CWSL"].iloc[0]
        assert np.isclose(row, cwsl_val)


def test_compute_cwsl_sensitivity_df_supports_per_row_co_and_weights():
    """
    Structural + behavioral:

    - Supports co as a column name.
    - Supports sample_weight via column.
    - Still matches cwsl_sensitivity when we replicate the same logic.
    """
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

    # Core behavior with array co + weights
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

    # Same R values
    assert set(out["R"].astype(float)) == set(float(r) for r in core.keys())

    # Check each R's CWSL
    for r, cwsl_val in core.items():
        row = out.loc[out["R"] == float(r), "CWSL"].iloc[0]
        assert np.isclose(row, cwsl_val)


def test_compute_cwsl_sensitivity_df_ignores_non_positive_R_and_raises_if_none_valid():
    """
    - Non-positive R values in R_list should be ignored.
    - If all R values are non-positive, the helper should raise ValueError.
    """
    df = pd.DataFrame(
        {
            "actual": [10.0, 12.0],
            "forecast": [9.0, 11.0],
        }
    )

    # Mixed valid and invalid R -> should work and drop non-positive ones
    out = compute_cwsl_sensitivity_df(
        df=df,
        y_true_col="actual",
        y_pred_col="forecast",
        R_list=[-1.0, 0.0, 1.0, 2.0],
        co=1.0,
        sample_weight_col=None,
    )

    # Only positive Rs should remain
    assert set(out["R"].astype(float)) == {1.0, 2.0}

    # All invalid R -> should raise
    import pytest

    with pytest.raises(ValueError):
        compute_cwsl_sensitivity_df(
            df=df,
            y_true_col="actual",
            y_pred_col="forecast",
            R_list=[-2.0, 0.0],
            co=1.0,
            sample_weight_col=None,
        )
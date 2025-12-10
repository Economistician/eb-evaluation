from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eb_evaluation.adjustment import ReadinessAdjustmentLayer
from ebmetrics.metrics import cwsl


def _make_global_df(n: int = 20) -> pd.DataFrame:
    # Simple panel with systematic underforecast
    rng = np.random.default_rng(0)
    actual = rng.integers(80, 120, size=n)
    forecast = (actual * 0.8).astype(float)  # biased low

    return pd.DataFrame(
        {
            "actual": actual,
            "forecast": forecast,
        }
    )


def _make_segmented_df() -> pd.DataFrame:
    # Two clusters with opposite bias
    # cluster A: underforecast, cluster B: overforecast
    n_per = 10
    actual_a = np.full(n_per, 100.0)
    forecast_a = np.full(n_per, 80.0)  # underforecast

    actual_b = np.full(n_per, 100.0)
    forecast_b = np.full(n_per, 120.0)  # overforecast

    df = pd.DataFrame(
        {
            "cluster": ["A"] * n_per + ["B"] * n_per,
            "actual": np.concatenate([actual_a, actual_b]),
            "forecast": np.concatenate([forecast_a, forecast_b]),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Basic global behavior
# ---------------------------------------------------------------------------


def test_global_uplift_reduces_cwsl_and_adds_column():
    df = _make_global_df()
    cu, co = 2.0, 1.0

    y_true = df["actual"].to_numpy(dtype=float)
    y_pred = df["forecast"].to_numpy(dtype=float)
    base_cwsl = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)

    ral = ReadinessAdjustmentLayer(
        cu=cu,
        co=co,
        uplift_min=1.0,
        uplift_max=1.2,
        grid_step=0.01,
    )

    ral.fit(df, forecast_col="forecast", actual_col="actual")

    # Learned global uplift should be >= 1
    assert ral.global_uplift_ >= 1.0

    # Diagnostics present
    assert not ral.diagnostics_.empty
    assert {"scope", "uplift", "cwsl_before", "cwsl_after"}.issubset(
        ral.diagnostics_.columns
    )

    # Applying uplift should reduce CWSL relative to baseline
    df_adj = ral.transform(df, forecast_col="forecast")
    y_adj = df_adj["readiness_forecast"].to_numpy(dtype=float)
    adj_cwsl = cwsl(y_true=y_true, y_pred=y_adj, cu=cu, co=co)

    assert adj_cwsl <= base_cwsl

    # readiness_forecast should equal forecast * global_uplift_ (up to numeric noise)
    expected = y_pred * ral.global_uplift_
    np.testing.assert_allclose(
        df_adj["readiness_forecast"].to_numpy(dtype=float),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_transform_raises_if_not_fit():
    df = _make_global_df()
    ral = ReadinessAdjustmentLayer()
    with pytest.raises(RuntimeError):
        _ = ral.transform(df, forecast_col="forecast")


# ---------------------------------------------------------------------------
# Segmentation behavior
# ---------------------------------------------------------------------------


def test_segment_specific_uplift_and_fallback_to_global():
    df = _make_segmented_df()
    cu, co = 2.0, 1.0

    ral = ReadinessAdjustmentLayer(
        cu=cu,
        co=co,
        uplift_min=1.0,
        uplift_max=1.15,
        grid_step=0.01,
    )

    ral.fit(
        df,
        forecast_col="forecast",
        actual_col="actual",
        segment_cols=["cluster"],
    )

    # Segment table should contain both clusters
    uplift_table = ral.uplift_table_
    assert set(uplift_table["cluster"].unique()) == {"A", "B"}

    # For cluster A (underforecast), best uplift should sit at upper bound (1.15)
    uplift_a = uplift_table.loc[uplift_table["cluster"] == "A", "uplift"].iloc[0]
    assert pytest.approx(uplift_a, rel=1e-6) == 1.15

    # For cluster B (overforecast), uplift should remain at 1.0
    uplift_b = uplift_table.loc[uplift_table["cluster"] == "B", "uplift"].iloc[0]
    assert pytest.approx(uplift_b, rel=1e-6) == 1.0

    # Build future data with an unseen cluster "C" to exercise global fallback
    df_future = pd.DataFrame(
        {
            "cluster": ["A", "B", "C"],
            "actual": [100.0, 100.0, 100.0],  # not used by transform
            "forecast": [80.0, 120.0, 90.0],
        }
    )

    df_adj = ral.transform(
        df_future,
        forecast_col="forecast",
        segment_cols=["cluster"],
    )

    # Extract applied uplift per row
    applied_uplift = df_adj["readiness_forecast"].to_numpy(dtype=float) / df_adj[
        "forecast"
    ].to_numpy(dtype=float)

    # Row 0: cluster A → segment uplift
    assert pytest.approx(applied_uplift[0], rel=1e-6) == uplift_a
    # Row 1: cluster B → segment uplift
    assert pytest.approx(applied_uplift[1], rel=1e-6) == uplift_b
    # Row 2: cluster C (unseen) → global uplift
    assert pytest.approx(applied_uplift[2], rel=1e-6) == ral.global_uplift_


# ---------------------------------------------------------------------------
# Sample-weighted behavior
# ---------------------------------------------------------------------------


def test_sample_weight_changes_optimal_uplift():
    # Construct a simple case where weights concentrate on underforecasted rows.
    df = pd.DataFrame(
        {
            "actual": [100.0, 100.0],
            "forecast": [80.0, 120.0],  # first row under, second over
            "w": [10.0, 1.0],  # emphasize first row
        }
    )
    cu, co = 2.0, 1.0

    ral_unweighted = ReadinessAdjustmentLayer(
        cu=cu,
        co=co,
        uplift_min=1.0,
        uplift_max=1.2,
        grid_step=0.01,
    )
    ral_unweighted.fit(df, forecast_col="forecast", actual_col="actual")

    ral_weighted = ReadinessAdjustmentLayer(
        cu=cu,
        co=co,
        uplift_min=1.0,
        uplift_max=1.2,
        grid_step=0.01,
    )
    ral_weighted.fit(
        df,
        forecast_col="forecast",
        actual_col="actual",
        sample_weight_col="w",
    )

    # With sample weights emphasizing the underforecasted row, the optimal uplift
    # should be at least as large as in the unweighted case.
    assert ral_weighted.global_uplift_ >= ral_unweighted.global_uplift_
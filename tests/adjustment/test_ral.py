import numpy as np
import pandas as pd
import pytest

from eb_evaluation.adjustment import ReadinessAdjustmentLayer
from eb_metrics.metrics import cwsl


def _make_global_df(n: int = 20) -> pd.DataFrame:
    """Create a simple dataset with systematic underforecast bias."""
    rng = np.random.default_rng(0)
    actual = rng.integers(80, 120, size=n)
    forecast = (actual * 0.8).astype(float)  # biased low
    return pd.DataFrame({"actual": actual, "forecast": forecast})


def _make_segmented_df() -> pd.DataFrame:
    """Create two segments with opposite bias (A under, B over)."""
    n_per = 10
    actual_a = np.full(n_per, 100.0)
    forecast_a = np.full(n_per, 80.0)  # underforecast
    actual_b = np.full(n_per, 100.0)
    forecast_b = np.full(n_per, 120.0)  # overforecast

    return pd.DataFrame(
        {
            "cluster": ["A"] * n_per + ["B"] * n_per,
            "actual": np.concatenate([actual_a, actual_b]),
            "forecast": np.concatenate([forecast_a, forecast_b]),
        }
    )


def test_global_uplift_reduces_cwsl_and_adds_column():
    df = _make_global_df()
    cu, co = 2.0, 1.0

    # Testing RAL's apply behavior (no fit required here)
    ral = ReadinessAdjustmentLayer(
        cu=cu,
        co=co,
        uplift_min=1.0,
        uplift_max=1.2,
        grid_step=0.01,
    )
    ral.transform(df, forecast_col="forecast")

    # Assertions
    assert ral.global_uplift_ is not None
    assert ral.global_uplift_ >= 1.0
    assert not ral.diagnostics_.empty
    assert {"scope", "uplift", "cwsl_before", "cwsl_after"}.issubset(ral.diagnostics_.columns)

    # Applying uplift should reduce CWSL relative to baseline
    y_true = df["actual"].to_numpy(dtype=float)
    y_pred = df["forecast"].to_numpy(dtype=float)
    base_cwsl = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)

    df_adj = ral.transform(df, forecast_col="forecast")
    y_adj = df_adj["readiness_forecast"].to_numpy(dtype=float)
    adj_cwsl = cwsl(y_true=y_true, y_pred=y_adj, cu=cu, co=co)

    assert adj_cwsl <= base_cwsl

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


def test_sample_weight_changes_optimal_uplift():
    # Construct a case where weights concentrate on underforecasted rows.
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

    # Assert the weighted uplift is higher than the unweighted
    assert ral_weighted.global_uplift_ is not None
    assert ral_unweighted.global_uplift_ is not None
    assert ral_weighted.global_uplift_ >= ral_unweighted.global_uplift_


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

    uplift_table = ral.uplift_table_
    assert uplift_table is not None
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

    # Row 0: cluster A -> segment uplift
    assert pytest.approx(applied_uplift[0], rel=1e-6) == uplift_a
    # Row 1: cluster B -> segment uplift
    assert pytest.approx(applied_uplift[1], rel=1e-6) == uplift_b
    # Row 2: cluster C (unseen) -> global uplift
    assert ral.global_uplift_ is not None
    assert pytest.approx(applied_uplift[2], rel=1e-6) == ral.global_uplift_


# -------------------------------------------------------------------
# PR3: Canonical RAL application utilities (governance-aware apply)
# -------------------------------------------------------------------


def _make_apply_ral_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_entity_id": [1, 1, 2, 2],
            "yhat_ral_raw": [-1.0, 3.2, 7.9, 8.1],
        }
    )


def _make_decisions_df() -> pd.DataFrame:
    # entity 1: continuous (no snap)
    # entity 2: pack-aware (snap to unit=4)
    return pd.DataFrame(
        {
            "forecast_entity_id": [1, 2],
            "snap_required": [False, True],
            "snap_unit": [np.nan, 4.0],
            "recommended_mode": ["continuous", "pack_aware"],
        }
    )


def test_apply_ral_continuous_nonneg_clip_applies_without_snapping() -> None:
    """
    Continuous path:
    - no snapping
    - nonneg clip to zero
    """
    from eb_evaluation.adjustment.ral import apply_ral

    df = _make_apply_ral_df()
    decisions = _make_decisions_df()

    out = apply_ral(
        df=df,
        decisions=decisions,
        join_keys=["forecast_entity_id"],
        pred_col="yhat_ral_raw",
        output_col="yhat_ral_governed",
        snap_mode="ceil",
        nonneg_mode="clip_zero",
    )

    # Entity 1 should not be snapped; only nonneg should apply.
    sub = out.loc[out["forecast_entity_id"] == 1, "yhat_ral_governed"].to_numpy(dtype=float)
    np.testing.assert_allclose(sub, np.asarray([0.0, 3.2], dtype=float), rtol=0, atol=1e-12)


def test_apply_ral_pack_aware_snaps_to_grid_and_applies_nonneg() -> None:
    """
    Pack-aware path:
    - snapping to grid applies to governed preds
    - nonneg applies after snapping
    """
    from eb_evaluation.adjustment.ral import apply_ral

    df = _make_apply_ral_df()
    decisions = _make_decisions_df()

    out = apply_ral(
        df=df,
        decisions=decisions,
        join_keys=["forecast_entity_id"],
        pred_col="yhat_ral_raw",
        output_col="yhat_ral_governed",
        snap_mode="ceil",
        nonneg_mode="clip_zero",
    )

    # Entity 2 should be snapped to multiples of 4 with ceil snapping:
    # 7.9 -> 8, 8.1 -> 12
    sub = out.loc[out["forecast_entity_id"] == 2, "yhat_ral_governed"].to_numpy(dtype=float)
    np.testing.assert_allclose(sub, np.asarray([8.0, 12.0], dtype=float), rtol=0, atol=1e-12)


def test_apply_ral_round_mode_changes_snapped_value() -> None:
    """
    Snap-mode behavior should be testable: round should differ from ceil for 8.1.
    """
    from eb_evaluation.adjustment.ral import apply_ral

    df = _make_apply_ral_df()
    decisions = _make_decisions_df()

    out_round = apply_ral(
        df=df,
        decisions=decisions,
        join_keys=["forecast_entity_id"],
        pred_col="yhat_ral_raw",
        output_col="yhat_ral_governed",
        snap_mode="round",
        nonneg_mode="allow",
    )

    sub = out_round.loc[out_round["forecast_entity_id"] == 2, "yhat_ral_governed"].to_numpy(
        dtype=float
    )
    # 7.9 rounds to 8, 8.1 rounds to 8 (nearest multiple of 4)
    np.testing.assert_allclose(sub, np.asarray([8.0, 8.0], dtype=float), rtol=0, atol=1e-12)


def test_apply_ral_raises_on_missing_decision_rows() -> None:
    """
    apply_ral should fail loudly if a join key doesn't find a governance decision.
    This prevents silent "policy missing" behavior.
    """
    from eb_evaluation.adjustment.ral import apply_ral

    df = _make_apply_ral_df()
    decisions = _make_decisions_df().loc[_make_decisions_df()["forecast_entity_id"] == 1].copy()

    with pytest.raises(ValueError):
        _ = apply_ral(
            df=df,
            decisions=decisions,
            join_keys=["forecast_entity_id"],
            pred_col="yhat_ral_raw",
            output_col="yhat_ral_governed",
            snap_mode="ceil",
            nonneg_mode="clip_zero",
        )

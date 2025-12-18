import numpy as np
import pandas as pd
import pytest

from eb_evaluation.features.feature_engineer import FeatureEngineer, FeatureConfig


def _make_panel_df(
    n_entities: int = 2,
    n_steps: int = 16,
    start: str = "2024-01-01",
    freq: str = "h",  # pandas deprecates "H" in favor of "h"
) -> pd.DataFrame:
    """Helper to build a simple panel DataFrame."""
    frames = []
    for e in range(n_entities):
        ts = pd.date_range(start=start, periods=n_steps, freq=freq)
        df_e = pd.DataFrame(
            {
                "entity_id": e,
                "timestamp": ts,
                "target": np.linspace(10, 20, n_steps) + e,
                "promo": np.where(np.arange(n_steps) % 4 == 0, 1.0, 0.0),
                "store_type": f"type_{e}",
            }
        )
        frames.append(df_e)
    return pd.concat(frames, ignore_index=True)


def test_basic_feature_engineering_shapes_and_names():
    df = _make_panel_df()
    fe = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")

    config = FeatureConfig(
        lag_steps=[1, 2],
        rolling_windows=[3],
        rolling_stats=["mean", "max"],
        calendar_features=["hour", "dow", "month", "is_weekend"],
        use_cyclical_time=True,
        regressor_cols=["promo"],
        static_cols=["store_type"],
    )

    X, y, feature_names = fe.transform(df, config)

    # We should have fewer rows than original because of lags / rolling windows
    assert X.shape[0] < df.shape[0]
    assert X.shape[0] == y.shape[0]

    # Check that key feature types are present
    for name in ["lag_1", "lag_2", "roll_3_mean", "roll_3_max"]:
        assert name in feature_names

    # Calendar + cyclical features
    for name in ["hour", "dayofweek", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert name in feature_names

    # Static + regressors carried through
    assert "store_type" in feature_names
    assert "promo" in feature_names

    # No NaNs or infs in outputs
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_raises_on_non_monotonic_timestamps_per_entity():
    df = _make_panel_df()
    # Create a duplicate timestamp for an entity to break strict monotonicity
    dup_row = df.iloc[0].copy()
    dup_row["timestamp"] = df.loc[0, "timestamp"]
    df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)

    fe = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")
    config = FeatureConfig()

    with pytest.raises(ValueError):
        fe.transform(df, config)


def test_raises_on_negative_target_values():
    df = _make_panel_df()
    df.loc[df.index[0], "target"] = -1.0  # inject a negative value

    fe = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")
    config = FeatureConfig()

    with pytest.raises(ValueError):
        fe.transform(df, config)


def test_auto_regressor_detection_and_default_config():
    # promo is numeric and should be auto-detected as a regressor
    df = _make_panel_df()
    fe = FeatureEngineer(entity_col="entity_id", timestamp_col="timestamp", target_col="target")

    config = FeatureConfig(
        lag_steps=[1],
        rolling_windows=[2],
        rolling_stats=["mean"],
        calendar_features=["hour"],
        static_cols=["store_type"],
        regressor_cols=None,  # trigger auto-detection
    )

    X, y, feature_names = fe.transform(df, config)

    # Expected core features
    assert "lag_1" in feature_names
    assert "roll_2_mean" in feature_names
    assert "hour" in feature_names
    assert "store_type" in feature_names
    # promo should have been auto-detected
    assert "promo" in feature_names

    assert X.shape[0] == y.shape[0]
import numpy as np
import pandas as pd

from eb_evaluation.model_selection import (
    compare_forecasts,
    select_model_by_cwsl,
    select_model_by_cwsl_cv,
)


class MeanEstimator:
    """
    Simple baseline estimator that predicts the mean of y_train.
    """

    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self.mean_ = float(y_arr.mean())
        return self

    def predict(self, X):
        n = len(X)
        return np.full(shape=n, fill_value=self.mean_, dtype=float)


class ZeroEstimator:
    """
    Baseline estimator that always predicts zero.
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        n = len(X)
        return np.zeros(shape=n, dtype=float)


def test_compare_forecasts_basic_ranking():
    y_true = np.array([10.0, 12.0, 8.0], dtype=float)

    forecasts = {
        "perfect": np.array([10.0, 12.0, 8.0], dtype=float),
        "bad": np.array([0.0, 0.0, 0.0], dtype=float),
    }

    cu, co = 2.0, 1.0

    df = compare_forecasts(
        y_true=y_true,
        forecasts=forecasts,
        cu=cu,
        co=co,
        sample_weight=None,
        tau=2.0,
    )

    # Index and key columns are present
    assert set(df.index) == {"perfect", "bad"}
    for col in ["CWSL", "NSL", "UD", "wMAPE", "MAE", "RMSE", "MAPE"]:
        assert col in df.columns

    # Perfect model should have lower CWSL than the bad one
    assert df.loc["perfect", "CWSL"] < df.loc["bad", "CWSL"]


def test_compare_forecasts_validates_inputs():
    y_true_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

    forecasts = {"m": np.array([1.0, 2.0])}

    cu, co = 1.0, 1.0

    import pytest

    with pytest.raises(ValueError):
        compare_forecasts(
            y_true=y_true_2d,
            forecasts=forecasts,
            cu=cu,
            co=co,
        )

    with pytest.raises(ValueError):
        compare_forecasts(
            y_true=np.array([1.0, 2.0]),
            forecasts={},
            cu=cu,
            co=co,
        )


def _build_simple_regression_data():
    X = np.arange(10).reshape(-1, 1).astype(float)
    y = np.linspace(5.0, 15.0, num=10)
    return X, y


def test_select_model_by_cwsl_prefers_better_model():
    import pytest

    X, y = _build_simple_regression_data()

    # Use last 3 points as validation
    X_train, X_val = X[:-3], X[-3:]
    y_train, y_val = y[:-3], y[-3:]

    models = {
        "mean": MeanEstimator(),
        "zero": ZeroEstimator(),
    }

    best_name, best_model, results = select_model_by_cwsl(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cu=2.0,
        co=1.0,
        sample_weight_val=None,
    )

    assert best_name in models
    assert best_model is not None
    assert best_name in results.index
    assert {"CWSL", "RMSE", "wMAPE"}.issubset(results.columns)

    # The mean estimator should beat the zero estimator on this data
    assert best_name == "mean"


def test_select_model_by_cwsl_accepts_sample_weight():
    X, y = _build_simple_regression_data()
    X_train, X_val = X[:-3], X[-3:]
    y_train, y_val = y[:-3], y[-3:]

    models = {"mean": MeanEstimator(), "zero": ZeroEstimator()}
    sample_weight_val = np.ones_like(y_val, dtype=float)

    # Should run without error when weights are provided
    best_name, best_model, results = select_model_by_cwsl(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cu=2.0,
        co=1.0,
        sample_weight_val=sample_weight_val,
    )

    assert best_name in models
    assert best_model is not None
    assert isinstance(results, pd.DataFrame)


def test_select_model_by_cwsl_cv_basic():
    X, y = _build_simple_regression_data()

    models = {
        "mean": MeanEstimator(),
        "zero": ZeroEstimator(),
    }

    best_name, best_model, results = select_model_by_cwsl_cv(
        models=models,
        X=X,
        y=y,
        cu=2.0,
        co=1.0,
        cv=3,
        sample_weight=np.ones_like(y, dtype=float),
    )

    assert best_name in models
    assert best_model is not None

    # Results table has expected structure
    assert set(results.index) == set(models.keys())
    expected_cols = {
        "CWSL_mean",
        "CWSL_std",
        "RMSE_mean",
        "RMSE_std",
        "wMAPE_mean",
        "wMAPE_std",
        "n_folds",
    }
    assert expected_cols.issubset(results.columns)
    assert (results["n_folds"] == 3).all()

    # On this data, the mean estimator should still win
    assert best_name == "mean"


def test_select_model_by_cwsl_cv_validates_arguments():
    import pytest

    X = np.arange(5).reshape(-1, 1).astype(float)
    y = np.linspace(0.0, 1.0, num=5)

    models = {"mean": MeanEstimator()}

    # cv < 2 should raise
    with pytest.raises(ValueError):
        select_model_by_cwsl_cv(
            models=models,
            X=X,
            y=y,
            cu=1.0,
            co=1.0,
            cv=1,
        )

    # sample_weight length mismatch should raise
    with pytest.raises(ValueError):
        select_model_by_cwsl_cv(
            models=models,
            X=X,
            y=y,
            cu=1.0,
            co=1.0,
            cv=2,
            sample_weight=np.ones(3, dtype=float),
        )

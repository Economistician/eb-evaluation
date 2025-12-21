import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from eb_evaluation.model_selection import CWSLRegressor, ElectricBarometer
from eb_metrics.metrics import cwsl


def _make_simple_data(n: int = 20):
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = 5.0 + 1.2 * X.ravel()  # simple upward trend
    return X, y


def test_cwsl_regressor_cv_basic_selection():
    """CWSLRegressor (CV mode) should pick the better model and expose metadata."""
    X, y = _make_simple_data()

    models = {
        "mean": DummyRegressor(strategy="mean"),
        "zero": DummyRegressor(strategy="constant", constant=0.0),
    }

    reg = CWSLRegressor(
        models=models,
        cu=2.0,
        co=1.0,
        selection_mode="cv",
        cv=3,
        random_state=42,
    )

    reg.fit(X, y)

    # Basic fitted attributes
    assert isinstance(reg.selector_, ElectricBarometer)
    assert reg.best_name_ in models
    assert reg.best_estimator_ is not None
    assert reg.results_ is not None
    assert reg.n_features_in_ == 1

    # On this simple series, "mean" should dominate "zero"
    assert reg.best_name_ == "mean"

    # Validation metrics should be populated
    assert reg.validation_cwsl_ is not None
    assert reg.validation_rmse_ is not None
    assert reg.validation_wmape_ is not None


def test_cwsl_regressor_holdout_mode_runs():
    """Holdout mode with validation_fraction should fit without error."""
    X, y = _make_simple_data()

    models = {
        "mean": DummyRegressor(strategy="mean"),
        "zero": DummyRegressor(strategy="constant", constant=0.0),
    }

    reg = CWSLRegressor(
        models=models,
        cu=2.0,
        co=1.0,
        selection_mode="holdout",
        validation_fraction=0.3,
        random_state=123,
    )

    reg.fit(X, y)

    assert isinstance(reg.selector_, ElectricBarometer)
    assert reg.selector_.selection_mode == "holdout"
    assert reg.best_estimator_ is not None
    assert reg.best_name_ in models
    assert reg.results_ is not None


def test_cwsl_regressor_score_matches_negative_cwsl():
    """score(X, y) should return negative CWSL, higher-is-better."""
    X, y = _make_simple_data()

    # Single model so selection is trivial
    models = {
        "mean": DummyRegressor(strategy="mean"),
    }

    reg = CWSLRegressor(
        models=models,
        cu=2.0,
        co=1.0,
        selection_mode="cv",
        cv=3,
        random_state=7,
    )

    reg.fit(X, y)

    score = reg.score(X, y)
    y_pred = reg.predict(X)

    expected_cost = cwsl(
        y_true=y,
        y_pred=y_pred,
        cu=2.0,
        co=1.0,
        sample_weight=None,
    )

    # sklearn convention: higher is better, so score = -cost
    assert np.isclose(score, -expected_cost)
    assert score <= 0.0  # CWSL is non-negative, so score should be <= 0


def test_cwsl_regressor_score_with_sample_weight():
    """score should accept sample_weight and still return negative CWSL."""
    X, y = _make_simple_data()

    models = {"mean": DummyRegressor(strategy="mean")}

    reg = CWSLRegressor(
        models=models,
        cu=2.0,
        co=1.0,
        selection_mode="cv",
        cv=3,
        random_state=0,
    )
    reg.fit(X, y)

    sample_weight = np.linspace(0.5, 1.5, num=len(y))
    score = reg.score(X, y, sample_weight=sample_weight)
    y_pred = reg.predict(X)

    expected_cost = cwsl(
        y_true=y,
        y_pred=y_pred,
        cu=2.0,
        co=1.0,
        sample_weight=sample_weight,
    )

    assert np.isclose(score, -expected_cost)


def test_cwsl_regressor_predict_before_fit_raises():
    """Calling predict() before fit() should raise a clear error."""
    X, _ = _make_simple_data()

    models = {"mean": DummyRegressor(strategy="mean")}
    reg = CWSLRegressor(models=models)

    with pytest.raises(RuntimeError):
        reg.predict(X)


def test_cwsl_regressor_get_set_params_roundtrip():
    """get_params / set_params provide a minimal sklearn-like API."""
    models = {"mean": DummyRegressor(strategy="mean")}
    reg = CWSLRegressor(
        models=models,
        cu=2.0,
        co=1.0,
        selection_mode="cv",
        cv=3,
        validation_fraction=0.25,
        random_state=42,
    )

    params = reg.get_params()
    assert params["cu"] == 2.0
    assert params["selection_mode"] == "cv"

    # Change a couple of params and ensure they are updated
    reg.set_params(selection_mode="holdout", cv=5)
    assert reg.selection_mode == "holdout"
    assert reg.cv == 5

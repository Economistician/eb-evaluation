"""
Tests for eb_evaluation.model_selection.electric_barometer.ElectricBarometer.

These tests are intentionally "contract tests":
- verify include/exclude filtering
- verify error_policy behavior
- verify metric routing (cwsl/rmse/wmape)
- verify holdout + cv modes produce expected fitted state and audit fields
- verify tie_breaker + tie_tol behavior
- verify validate_inputs behavior (strict/coerce/off)

They avoid testing sklearn correctness or metric math (owned by eb-metrics).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from eb_evaluation.model_selection.electric_barometer import ElectricBarometer

# ---------------------------------------------------------------------
# Simple deterministic toy models for selection behavior
# ---------------------------------------------------------------------


class _ZeroModel:
    """Predicts all zeros."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class _OneModel:
    """Predicts all ones."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=float)


class _BadFitModel:
    """Always fails during fit."""

    def fit(self, X, y):
        raise RuntimeError("boom")

    def predict(self, X):
        raise RuntimeError("boom")


class _OneModelWithParamsSmall:
    """Predicts all ones; exposes a small parameter set for 'simpler' tie-breaking."""

    def get_params(self, deep: bool = True):
        _ = deep
        return {"alpha": 1.0}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=float)


class _OneModelWithParamsLarge:
    """Predicts all ones; exposes a larger parameter set for 'simpler' tie-breaking."""

    def get_params(self, deep: bool = True):
        _ = deep
        return {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=float)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _holdout_data():
    """
    Make a tiny dataset where:
    - y_val is all ones
    - _OneModel is perfect on val; _ZeroModel is imperfect
    This makes selection deterministic for all three metrics.
    """
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.ones(10, dtype=float)

    X_tr, y_tr = X[:6], y[:6]
    X_va, y_va = X[6:], y[6:]
    return X_tr, y_tr, X_va, y_va


def _cv_data(n: int = 12):
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.ones(n, dtype=float)
    return X, y


# ---------------------------------------------------------------------
# Constructor filtering (include/exclude)
# ---------------------------------------------------------------------


def test_init_include_filters_models():
    models = {"zero": _ZeroModel(), "one": _OneModel()}
    eb = ElectricBarometer(models=models, include={"one"})
    assert list(eb.models.keys()) == ["one"]
    assert eb.candidate_names_ == ["one"]


def test_init_exclude_filters_models():
    models = {"zero": _ZeroModel(), "one": _OneModel()}
    eb = ElectricBarometer(models=models, exclude={"zero"})
    assert list(eb.models.keys()) == ["one"]
    assert eb.candidate_names_ == ["one"]


def test_init_include_unknown_raises():
    models = {"zero": _ZeroModel()}
    with pytest.raises(ValueError, match=r"include contains unknown model names"):
        ElectricBarometer(models=models, include={"nope"})


def test_init_exclude_unknown_raises():
    models = {"zero": _ZeroModel()}
    with pytest.raises(ValueError, match=r"exclude contains unknown model names"):
        ElectricBarometer(models=models, exclude={"nope"})


# ---------------------------------------------------------------------
# Holdout mode basic behavior + audit fields
# ---------------------------------------------------------------------


MetricName = Literal["cwsl", "rmse", "wmape"]


@pytest.mark.parametrize("metric", ["cwsl", "rmse", "wmape"])
def test_holdout_fit_selects_expected_model(metric: MetricName):
    X_tr, y_tr, X_va, y_va = _holdout_data()

    models = {"zero": _ZeroModel(), "one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        metric=metric,  # deterministic winner should always be "one"
        error_policy="raise",
        selection_mode="holdout",
    )

    eb.fit(X_tr, y_tr, X_va, y_va)

    assert eb.best_name_ == "one"
    assert eb.best_model_ is not None
    assert eb.results_ is not None
    assert set(eb.results_.columns) == {"CWSL", "RMSE", "wMAPE"}
    assert set(eb.results_.index) == {"zero", "one"}

    # Audit fields should be reset/populated for each fit call
    assert eb.candidate_names_ == ["zero", "one"]
    assert eb.evaluated_names_ == ["zero", "one"]
    assert eb.stopped_early_ is False
    assert eb.stop_reason_ is None


def test_predict_requires_fit():
    models = {"zero": _ZeroModel()}
    eb = ElectricBarometer(models=models)
    with pytest.raises(RuntimeError, match=r"has not been fit yet"):
        eb.predict(np.zeros((3, 1), dtype=float))


def test_cwsl_score_works():
    models = {"one": _OneModel()}
    eb = ElectricBarometer(models=models, cu=2.0, co=1.0)
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    score = eb.cwsl_score(y_true=y_true, y_pred=y_pred)
    assert isinstance(score, float)
    assert score == 0.0


# ---------------------------------------------------------------------
# Error policy behavior
# ---------------------------------------------------------------------


def test_error_policy_skip_records_failure_and_selects_good():
    X_tr, y_tr, X_va, y_va = _holdout_data()

    models = {"bad": _BadFitModel(), "one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        metric="cwsl",
        error_policy="skip",
        selection_mode="holdout",
    )

    eb.fit(X_tr, y_tr, X_va, y_va)

    assert eb.best_name_ == "one"
    assert "bad" in eb.failures_
    assert eb.results_ is not None
    assert list(eb.results_.index) == ["one"]  # only successful model included

    # Evaluated should include attempted models, even if they fail
    assert eb.evaluated_names_ == ["bad", "one"]


def test_error_policy_warn_skip_emits_warning():
    X_tr, y_tr, X_va, y_va = _holdout_data()

    models = {"bad": _BadFitModel(), "one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        metric="cwsl",
        error_policy="warn_skip",
        selection_mode="holdout",
    )

    with pytest.warns(RuntimeWarning, match=r"Skipping model"):
        eb.fit(X_tr, y_tr, X_va, y_va)

    assert eb.best_name_ == "one"
    assert "bad" in eb.failures_


def test_error_policy_raise_raises():
    X_tr, y_tr, X_va, y_va = _holdout_data()

    models = {"bad": _BadFitModel()}
    eb = ElectricBarometer(
        models=models,
        metric="cwsl",
        error_policy="raise",
        selection_mode="holdout",
    )

    with pytest.raises(RuntimeError, match=r"failed"):
        eb.fit(X_tr, y_tr, X_va, y_va)


# ---------------------------------------------------------------------
# CV mode behavior + audit fields
# ---------------------------------------------------------------------


def test_cv_fit_selects_expected_model_and_refits():
    X, y = _cv_data(n=12)

    models = {"zero": _ZeroModel(), "one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        metric="cwsl",
        error_policy="raise",
        selection_mode="cv",
        cv=3,
        random_state=0,
    )

    # In cv mode, X_val/y_val are ignored by implementation, but signature requires them.
    eb.fit(X, y, X, y)

    assert eb.best_name_ == "one"
    assert eb.best_model_ is not None
    assert eb.results_ is not None
    assert set(eb.results_.columns) == {"CWSL", "RMSE", "wMAPE"}
    assert set(eb.results_.index) == {"zero", "one"}

    # Audit fields
    assert eb.candidate_names_ == ["zero", "one"]
    assert eb.evaluated_names_ == ["zero", "one"]
    assert eb.stopped_early_ is False
    assert eb.stop_reason_ is None


def test_cv_invalid_cv_raises():
    X, y = _cv_data(n=5)
    models = {"one": _OneModel()}
    eb = ElectricBarometer(models=models, selection_mode="cv", cv=10)
    with pytest.raises(ValueError, match=r"Invalid number of folds"):
        eb.fit(X, y, X, y)


# ---------------------------------------------------------------------
# Tie-breaking behavior
# ---------------------------------------------------------------------


def test_tie_breaker_name_picks_lexicographically_smallest():
    """
    Two models are exactly tied (both perfect); tie_breaker='name' should choose by name.
    """
    X_tr, y_tr, X_va, y_va = _holdout_data()

    models = {"b": _OneModel(), "a": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        metric="cwsl",
        error_policy="raise",
        selection_mode="holdout",
        tie_tol=0.0,
        tie_breaker="name",
    )

    eb.fit(X_tr, y_tr, X_va, y_va)
    assert eb.best_name_ == "a"


def test_tie_breaker_simpler_prefers_smaller_param_surface():
    """
    Two models are exactly tied (both perfect); tie_breaker='simpler' should prefer the
    one with fewer parameters (via get_params length heuristic).
    """
    X_tr, y_tr, X_va, y_va = _holdout_data()

    models = {"small": _OneModelWithParamsSmall(), "large": _OneModelWithParamsLarge()}
    eb = ElectricBarometer(
        models=models,
        metric="rmse",
        error_policy="raise",
        selection_mode="holdout",
        tie_tol=0.0,
        tie_breaker="simpler",
    )

    eb.fit(X_tr, y_tr, X_va, y_va)
    assert eb.best_name_ == "small"


def test_tie_tol_allows_near_ties_to_be_considered():
    """
    tie_tol should treat models within best + tol as tied.
    Here both are perfect anyway, but this test asserts the parameter is accepted and
    the selection remains deterministic under tie_breaker='name'.
    """
    X_tr, y_tr, X_va, y_va = _holdout_data()

    models = {"b": _OneModel(), "a": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        metric="wmape",
        error_policy="raise",
        selection_mode="holdout",
        tie_tol=0.5,
        tie_breaker="name",
    )

    eb.fit(X_tr, y_tr, X_va, y_va)
    assert eb.best_name_ == "a"


# ---------------------------------------------------------------------
# Input validation behavior
# ---------------------------------------------------------------------


def test_validate_inputs_strict_rejects_nan_in_y():
    X_tr, y_tr, X_va, y_va = _holdout_data()
    y_tr_bad = y_tr.copy()
    y_tr_bad[0] = np.nan

    models = {"one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        selection_mode="holdout",
        error_policy="raise",
        validate_inputs="strict",
    )

    with pytest.raises(ValueError, match=r"contains NaN or inf"):
        eb.fit(X_tr, y_tr_bad, X_va, y_va)


def test_validate_inputs_strict_rejects_non_numeric_object_X():
    """
    Under strict validation, object dtype inputs are rejected (even if coercible).
    """
    X_tr, y_tr, X_va, y_va = _holdout_data()

    # Make X_train object dtype (coercible strings)
    X_tr_obj = X_tr.astype(object)
    X_tr_obj[:, 0] = [str(int(v)) for v in X_tr[:, 0]]

    models = {"one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        selection_mode="holdout",
        error_policy="raise",
        validate_inputs="strict",
    )

    with pytest.raises(ValueError, match=r"must be numeric"):
        eb.fit(X_tr_obj, y_tr, X_va, y_va)


def test_validate_inputs_coerce_accepts_coercible_object_X():
    """
    Under coerce, object dtype inputs should be coerced to float (if possible).
    """
    X_tr, y_tr, X_va, y_va = _holdout_data()

    X_tr_obj = X_tr.astype(object)
    X_tr_obj[:, 0] = [str(int(v)) for v in X_tr[:, 0]]

    X_va_obj = X_va.astype(object)
    X_va_obj[:, 0] = [str(int(v)) for v in X_va[:, 0]]

    models = {"one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        selection_mode="holdout",
        error_policy="raise",
        validate_inputs="coerce",
    )

    eb.fit(X_tr_obj, y_tr, X_va_obj, y_va)
    assert eb.best_name_ == "one"


def test_validate_inputs_off_allows_nan_in_X_for_models_that_ignore_X():
    """
    With validate_inputs='off', we should not error on NaNs. Our toy model ignores X, so
    selection should still succeed.
    """
    X_tr, y_tr, X_va, y_va = _holdout_data()

    X_tr_bad = X_tr.copy()
    X_tr_bad[0, 0] = np.nan

    models = {"one": _OneModel()}
    eb = ElectricBarometer(
        models=models,
        selection_mode="holdout",
        error_policy="raise",
        validate_inputs="off",
    )

    eb.fit(X_tr_bad, y_tr, X_va, y_va)
    assert eb.best_name_ == "one"

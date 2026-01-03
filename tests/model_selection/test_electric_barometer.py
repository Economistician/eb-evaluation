"""
Tests for eb_evaluation.model_selection.electric_barometer.ElectricBarometer.

These tests are intentionally "contract tests":
- verify include/exclude filtering
- verify error_policy behavior
- verify metric routing (cwsl/rmse/wmape)
- verify holdout + cv modes produce expected fitted state and audit fields

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

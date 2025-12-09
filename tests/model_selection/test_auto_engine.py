from __future__ import annotations

import numpy as np
import pytest

from eb_evaluation.model_selection import AutoEngine, ElectricBarometer


def _dummy_data():
    X = np.zeros((10, 2), dtype=float)
    y = np.linspace(0.0, 1.0, 10)
    return X, y


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_autoengine_invalid_costs_raise():
    with pytest.raises(ValueError):
        AutoEngine(cu=0.0, co=1.0)

    with pytest.raises(ValueError):
        AutoEngine(cu=1.0, co=0.0)


def test_autoengine_invalid_selection_mode_raises():
    with pytest.raises(ValueError):
        AutoEngine(selection_mode="not-a-mode")  # type: ignore[arg-type]


def test_autoengine_invalid_speed_raises():
    with pytest.raises(ValueError):
        AutoEngine(speed="turbo")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Model zoo + ElectricBarometer wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("speed", ["fast", "balanced", "slow"])
def test_build_selector_returns_electric_barometer(speed: str):
    X, y = _dummy_data()
    engine = AutoEngine(cu=2.0, co=1.0, speed=speed)

    eb = engine.build_selector(X, y)

    assert isinstance(eb, ElectricBarometer)
    assert hasattr(eb, "models")
    assert isinstance(eb.models, dict)

    # At least the baselines should always be present
    for name in ["dummy_mean", "linear", "ridge", "lasso"]:
        assert name in eb.models

    # And we should have more than just the baselines
    assert len(eb.models) >= 2


def test_optional_dependencies_are_gated(monkeypatch):
    """
    If _has_package reports False for optional deps, the corresponding
    entries should not appear in the model zoo.
    """
    engine = AutoEngine(cu=2.0, co=1.0, speed="balanced")

    def fake_has_package(name: str) -> bool:
        # Pretend that no optional libs are available
        if name in {"xgboost", "lightgbm", "catboost"}:
            return False
        return False

    monkeypatch.setattr(engine, "_has_package", fake_has_package)

    models = engine._make_base_models()

    assert "xgb" not in models
    assert "lgbm" not in models
    assert "catboost" not in models

    # Baselines should still be there
    assert "dummy_mean" in models
    assert "linear" in models


def test_autoengine_repr_round_trips_core_config():
    engine = AutoEngine(
        cu=2.0,
        co=1.0,
        tau=3.0,
        selection_mode="cv",
        cv=4,
        random_state=42,
        speed="slow",
    )

    text = repr(engine)
    # Sanity checks: key parameters appear in repr
    assert "AutoEngine" in text
    assert "cu=2.0" in text
    assert "co=1.0" in text
    assert "selection_mode='cv'" in text
    assert "speed='slow'" in text
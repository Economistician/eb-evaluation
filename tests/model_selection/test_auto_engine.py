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
    engine = AutoEngine(cu=2.0, co=1.0, speed=speed)  # type: ignore[arg-type]

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
    engine = AutoEngine(cu=2.0, co=1.0, speed="balanced")  # type: ignore[arg-type]

    def fake_has_package(name: str) -> bool:
        return False

    monkeypatch.setattr(engine, "_has_package", fake_has_package)

    models = engine._make_base_models()

    assert "xgb" not in models
    assert "lgbm" not in models
    assert "catboost" not in models
    assert "prophet" not in models

    # Baselines should still be there
    assert "dummy_mean" in models
    assert "linear" in models


def test_prophet_only_included_for_slow_when_available(monkeypatch):
    """
    Prophet is time-series oriented and should only be included in the zoo
    when speed="slow" AND the optional dependency is available.
    """

    # Force prophet "available" and all other optionals "unavailable" to make this deterministic.
    def fake_has_package(name: str) -> bool:
        return name == "prophet"

    # fast => prophet should not be present
    engine_fast = AutoEngine(cu=2.0, co=1.0, speed="fast")  # type: ignore[arg-type]
    monkeypatch.setattr(engine_fast, "_has_package", fake_has_package)
    models_fast = engine_fast._make_base_models()
    assert "prophet" not in models_fast

    # balanced => prophet should not be present
    engine_bal = AutoEngine(cu=2.0, co=1.0, speed="balanced")  # type: ignore[arg-type]
    monkeypatch.setattr(engine_bal, "_has_package", fake_has_package)
    models_bal = engine_bal._make_base_models()
    assert "prophet" not in models_bal

    # slow => prophet should be present (if import works)
    engine_slow = AutoEngine(cu=2.0, co=1.0, speed="slow")  # type: ignore[arg-type]
    monkeypatch.setattr(engine_slow, "_has_package", fake_has_package)
    models_slow = engine_slow._make_base_models()

    # If prophet is installed but misconfigured, AutoEngine intentionally skips it;
    # in that case, this assertion could be flaky. Prefer to skip if Prophet isn't importable.
    try:
        import prophet
    except Exception:
        pytest.skip("prophet is not importable in this test environment")

    assert "prophet" in models_slow


def test_available_models_matches_built_zoo(monkeypatch):
    """
    available_models() should reflect the same keys as the built zoo, and
    should remain deterministic under gating of optional dependencies.
    """
    engine = AutoEngine(cu=2.0, co=1.0, speed="balanced")  # type: ignore[arg-type]

    def fake_has_package(name: str) -> bool:
        return False

    monkeypatch.setattr(engine, "_has_package", fake_has_package)

    available = engine.available_models()
    zoo = engine.build_zoo()

    assert available == list(zoo.keys())

    # Core models should always be present
    for name in ["dummy_mean", "linear", "ridge", "lasso", "rf", "gbr"]:
        assert name in available


def test_build_selector_supports_include_exclude(monkeypatch):
    """
    build_selector should filter the zoo based on include/exclude.
    """
    X, y = _dummy_data()
    engine = AutoEngine(cu=2.0, co=1.0, speed="balanced")  # type: ignore[arg-type]

    def fake_has_package(name: str) -> bool:
        return False

    monkeypatch.setattr(engine, "_has_package", fake_has_package)

    eb = engine.build_selector(X, y, include={"linear", "ridge", "rf"}, exclude={"rf"})

    assert set(eb.models.keys()) == {"linear", "ridge"}


def test_build_selector_include_unknown_raises(monkeypatch):
    """
    Unknown include names should raise (validated by AutoEngine against zoo keys).
    """
    X, y = _dummy_data()
    engine = AutoEngine(cu=2.0, co=1.0, speed="fast")  # type: ignore[arg-type]

    def fake_has_package(name: str) -> bool:
        return False

    monkeypatch.setattr(engine, "_has_package", fake_has_package)

    with pytest.raises(ValueError):
        engine.build_selector(X, y, include={"does_not_exist"})


def test_autoengine_repr_round_trips_core_config():
    engine = AutoEngine(
        cu=2.0,
        co=1.0,
        tau=3.0,
        selection_mode="cv",
        cv=4,
        random_state=42,
        speed="slow",  # type: ignore[arg-type]
    )

    text = repr(engine)
    # Sanity checks: key parameters appear in repr
    assert "AutoEngine" in text
    assert "cu=2.0" in text
    assert "co=1.0" in text
    assert "selection_mode='cv'" in text
    assert "speed='slow'" in text

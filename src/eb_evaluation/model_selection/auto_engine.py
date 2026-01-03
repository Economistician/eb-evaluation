"""
Auto model-zoo builder for Electric Barometer selection.

This module defines :class:`~eb_evaluation.model_selection.auto_engine.AutoEngine`, a
convenience factory that constructs an unfitted
:class:`~eb_evaluation.model_selection.electric_barometer.ElectricBarometer` with a curated
set of candidate regressors (a "model zoo") and asymmetric cost parameters.

The intent is to provide a batteries-included entry point:

- choose asymmetric costs (cu, co)
- choose a speed preset (fast, balanced, slow)
- optionally include additional engines (XGBoost, LightGBM, CatBoost) when installed
- get back an unfitted selector ready for cost-aware model selection

Model selection is performed using cost-aware criteria (e.g., CWSL) rather than symmetric
error alone, enabling operationally aligned choices.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Literal

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from eb_adapters import CatBoostAdapter, LightGBMRegressorAdapter
from eb_evaluation.model_selection.electric_barometer import ElectricBarometer

SpeedType = Literal["fast", "balanced", "slow"]


class AutoEngine:
    """
    Convenience factory for :class:`~eb_evaluation.model_selection.electric_barometer.ElectricBarometer`.

    AutoEngine builds an ElectricBarometer with a curated set of candidate models chosen
    by a speed preset:

    - ``speed="fast"``: small, inexpensive zoo; suitable for quick experiments and CI.
    - ``speed="balanced"`` (default): trade-off between runtime and modeling power.
    - ``speed="slow"``: heavier ensembles/boosting; use when wall-clock time is acceptable.

    Asymmetric costs define the primary selection objective via the cost ratio ``R = cu / co``.

    Parameters
    ----------
    cu
        Underbuild (shortfall) cost per unit. Must be strictly positive.
    co
        Overbuild (excess) cost per unit. Must be strictly positive.
    tau
        Tolerance parameter forwarded to ElectricBarometer for optional diagnostics (e.g., HR@tau).
    selection_mode
        Selection strategy used by ElectricBarometer. Must be ``"holdout"`` or ``"cv"``.
    cv
        Number of folds when ``selection_mode="cv"``.
    random_state
        Seed used for stochastic models and (when applicable) cross-validation.
    speed
        Controls which models are included and their approximate complexity.

    Notes
    -----
    Optional engines are included only when their packages are installed: ``xgboost``,
    ``lightgbm``, ``catboost``.
    """

    def __init__(
        self,
        *,
        cu: float = 2.0,
        co: float = 1.0,
        tau: float = 2.0,
        selection_mode: str = "holdout",
        cv: int = 3,
        random_state: int | None = None,
        speed: SpeedType = "balanced",
    ) -> None:
        if cu <= 0 or co <= 0:
            raise ValueError("AutoEngine: cu and co must be strictly positive.")

        if selection_mode not in {"holdout", "cv"}:
            raise ValueError(
                f"AutoEngine: selection_mode must be 'holdout' or 'cv', got {selection_mode!r}."
            )

        if speed not in {"fast", "balanced", "slow"}:
            raise ValueError(
                f"AutoEngine.speed must be one of 'fast', 'balanced', 'slow'; got {speed!r}."
            )

        self.cu = float(cu)
        self.co = float(co)
        self.tau = float(tau)
        self.selection_mode = selection_mode
        self.cv = int(cv)
        self.random_state = random_state
        self.speed: SpeedType = speed

    @staticmethod
    def _has_package(name: str) -> bool:
        """Return True if the given package can be imported (without importing it)."""
        return importlib.util.find_spec(name) is not None

    def _make_base_models(self) -> dict[str, Any]:
        """
        Build the default model zoo for the chosen speed preset.

        Returns
        -------
        dict[str, Any]
            Mapping of ``{name: estimator}`` for candidate regressors.

        Notes
        -----
        The zoo always includes inexpensive baselines, then adds heavier tree/boosting models
        depending on ``speed``. Optional engines (XGBoost, LightGBM, CatBoost) are included
        only if installed and importable.
        """
        models: dict[str, Any] = {}

        # Always-include baselines
        models["dummy_mean"] = DummyRegressor(strategy="mean")
        models["linear"] = LinearRegression()
        models["ridge"] = Ridge(alpha=1.0)
        models["lasso"] = Lasso(alpha=0.001)

        # Tree / ensemble configs differ by speed
        if self.speed == "fast":
            rf_estimators = 30
            gbr_estimators = 60
            gbr_lr = 0.1
        elif self.speed == "balanced":
            rf_estimators = 100
            gbr_estimators = 100
            gbr_lr = 0.1
        else:  # "slow"
            rf_estimators = 300
            gbr_estimators = 300
            gbr_lr = 0.05

        models["rf"] = RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=None,
            n_jobs=-1,
            random_state=self.random_state,
        )

        models["gbr"] = GradientBoostingRegressor(
            n_estimators=gbr_estimators,
            learning_rate=gbr_lr,
            max_depth=3,
            random_state=self.random_state,
        )

        # Optional: XGBoost (sklearn API)
        if self._has_package("xgboost"):
            try:  # pragma: no cover - depends on optional pkg
                from xgboost import XGBRegressor  # type: ignore

                if self.speed == "fast":
                    xgb_estimators = 60
                    xgb_lr = 0.15
                elif self.speed == "balanced":
                    xgb_estimators = 120
                    xgb_lr = 0.1
                else:  # "slow"
                    xgb_estimators = 300
                    xgb_lr = 0.05

                models["xgb"] = XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=xgb_estimators,
                    max_depth=4,
                    learning_rate=xgb_lr,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.random_state,
                )
            except Exception:
                # If xgboost is partially installed or misconfigured, skip it.
                pass

        # Optional: LightGBM via adapter
        if self._has_package("lightgbm"):
            try:  # pragma: no cover - depends on optional pkg
                if self.speed == "fast":
                    lgbm_estimators = 80
                    lgbm_lr = 0.1
                elif self.speed == "balanced":
                    lgbm_estimators = 150
                    lgbm_lr = 0.07
                else:  # "slow"
                    lgbm_estimators = 300
                    lgbm_lr = 0.05

                models["lgbm"] = LightGBMRegressorAdapter(
                    n_estimators=lgbm_estimators,
                    learning_rate=lgbm_lr,
                    max_depth=-1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.random_state,
                )
            except Exception:
                # Adapter will raise if lightgbm is missing; just skip.
                pass

        # Optional: CatBoost via adapter (mainly for balanced/slow)
        if self._has_package("catboost") and self.speed in {"balanced", "slow"}:
            try:  # pragma: no cover - depends on optional pkg
                if self.speed == "balanced":
                    cb_iterations = 120
                    cb_lr = 0.1
                else:  # "slow"
                    cb_iterations = 300
                    cb_lr = 0.05

                models["catboost"] = CatBoostAdapter(
                    iterations=cb_iterations,
                    depth=4,
                    learning_rate=cb_lr,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=self.random_state,
                )
            except Exception:
                # If CatBoost is unavailable or misconfigured, skip it.
                pass

        return models

    def available_models(self) -> list[str]:
        """
        List the model names that would be included in the zoo for this AutoEngine.

        This reflects:
        - the selected ``speed`` preset, and
        - which optional packages are available in the current environment.

        Returns
        -------
        list[str]
            Model names in deterministic insertion order.
        """
        return list(self._make_base_models().keys())

    def build_zoo(self) -> dict[str, Any]:
        """
        Build and return the unfitted model zoo.

        Returns
        -------
        dict[str, Any]
            Mapping of ``{name: estimator}`` for candidate regressors.

        Notes
        -----
        The returned dict is a shallow copy so callers may filter/mutate it without
        affecting future calls.
        """
        return dict(self._make_base_models())

    def build_selector(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        metric: ElectricBarometer._MetricName = "cwsl",
        error_policy: ElectricBarometer._ErrorPolicy = "warn_skip",
        time_budget_s: float | None = None,
        per_model_time_budget_s: float | None = None,
        refit_on_full: bool = False,
    ) -> ElectricBarometer:
        """
        Build an unfitted ElectricBarometer configured with the default model zoo.

        Parameters
        ----------
        X
            Feature matrix. Currently unused by the builder (reserved for future heuristics).
        y
            Target vector. Currently unused by the builder (reserved for future heuristics).
        include
            Optional allowlist of model names to include from the zoo.
        exclude
            Optional blocklist of model names to exclude from the zoo.
        metric
            Selection objective used by ElectricBarometer to choose the winning model.
        error_policy
            Behavior when a candidate model fails to fit/predict or otherwise errors.
        time_budget_s
            Optional wall-clock time budget (seconds) for the full selection run.
        per_model_time_budget_s
            Optional wall-clock time budget (seconds) per candidate model.
        refit_on_full
            Forwarded to ElectricBarometer; controls whether the winning model is refit on
            train+validation in holdout mode.

        Returns
        -------
        ElectricBarometer
            Unfitted selector instance.
        """
        _ = (X, y)  # reserved for future use

        models = self._make_base_models()

        # Delegate include/exclude validation and filtering to ElectricBarometer
        # to avoid duplicating the filtering rules in two places.
        return ElectricBarometer(
            models=models,
            cu=self.cu,
            co=self.co,
            tau=self.tau,
            selection_mode=self.selection_mode,
            cv=self.cv,
            random_state=self.random_state,
            include=include,
            exclude=exclude,
            metric=metric,
            error_policy=error_policy,
            time_budget_s=time_budget_s,
            per_model_time_budget_s=per_model_time_budget_s,
            refit_on_full=refit_on_full,
        )

    def __repr__(self) -> str:
        return (
            f"AutoEngine(cu={self.cu}, co={self.co}, tau={self.tau}, "
            f"selection_mode={self.selection_mode!r}, cv={self.cv}, "
            f"random_state={self.random_state!r}, speed={self.speed!r})"
        )

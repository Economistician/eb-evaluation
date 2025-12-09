from __future__ import annotations

from typing import Any, Dict, Optional, Literal
import importlib.util

import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from eb_adapters import LightGBMRegressorAdapter, CatBoostAdapter
from eb_evaluation.model_selection.electric_barometer import ElectricBarometer

SpeedType = Literal["fast", "balanced", "slow"]


class AutoEngine:
    """
    AutoEngine: convenience factory for ElectricBarometer with a curated
    model zoo and CWSL settings.

    It builds an ElectricBarometer with a set of candidate models chosen
    based on a simple `speed` preset:

    - speed="fast"
        Small, cheap model zoo. Good for quick experiments or CI.
    - speed="balanced" (default)
        Solid all-rounder. Good trade-off between runtime and accuracy.
    - speed="slow"
        Heavier tree/boosting configs. Use when you really care about
        squeezing out extra performance and can afford wall-clock time.

    Parameters
    ----------
    cu : float
        Underbuild (shortfall) cost per unit.
    co : float
        Overbuild (excess) cost per unit.
    tau : float, default 2.0
        Reserved for future diagnostics; forwarded to ElectricBarometer.
    selection_mode : {"holdout", "cv"}, default "holdout"
        How ElectricBarometer should select models.
    cv : int, default 3
        Number of folds when selection_mode="cv".
    random_state : int or None, default None
        Seed used for tree/boosting models and CV.
    speed : {"fast", "balanced", "slow"}, default "balanced"
        Controls which models are included and how "heavy" they are
        parametrized.

    Notes
    -----
    - Optional engines (xgboost, lightgbm, catboost) are included only
      if the corresponding packages are installed.
    - build_selector() currently does *not* use X, y directly, but they
      are accepted so that future heuristics (e.g. n_samples-based
      decisions) can be added without breaking the API.
    """

    def __init__(
        self,
        *,
        cu: float = 2.0,
        co: float = 1.0,
        tau: float = 2.0,
        selection_mode: str = "holdout",
        cv: int = 3,
        random_state: Optional[int] = None,
        speed: SpeedType = "balanced",
    ) -> None:
        if cu <= 0 or co <= 0:
            raise ValueError("AutoEngine: cu and co must be strictly positive.")

        if selection_mode not in {"holdout", "cv"}:
            raise ValueError(
                "AutoEngine: selection_mode must be 'holdout' or 'cv', "
                f"got {selection_mode!r}."
            )

        if speed not in {"fast", "balanced", "slow"}:
            raise ValueError(
                "AutoEngine.speed must be one of 'fast', 'balanced', 'slow'; "
                f"got {speed!r}."
            )

        self.cu = float(cu)
        self.co = float(co)
        self.tau = float(tau)
        self.selection_mode = selection_mode
        self.cv = int(cv)
        self.random_state = random_state
        self.speed: SpeedType = speed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_package(name: str) -> bool:
        """Return True if the given package can be imported."""
        return importlib.util.find_spec(name) is not None

    def _make_base_models(self) -> Dict[str, Any]:
        """
        Build the default model zoo for the chosen speed preset.

        Returns
        -------
        models : dict[str, Any]
            Keys are model names, values are estimator objects.
        """
        models: Dict[str, Any] = {}

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
                # If xgboost is partially installed or misconfigured,
                # we simply skip it rather than failing AutoEngine.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_selector(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ElectricBarometer:
        """
        Build an ElectricBarometer configured with a default model zoo.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Currently unused by the builder, but accepted for future
            heuristics (e.g. auto-detect very small samples).
        y : array-like of shape (n_samples,)
            Currently unused by the builder.

        Returns
        -------
        eb : ElectricBarometer
            An *unfitted* ElectricBarometer instance. Call:

                eb.fit(X_train, y_train, X_val, y_val)

            to perform cost-aware model selection.
        """
        # X, y are not used yet; they are passed to keep API future-proof.
        _ = (X, y)

        models = self._make_base_models()

        eb = ElectricBarometer(
            models=models,
            cu=self.cu,
            co=self.co,
            tau=self.tau,
            selection_mode=self.selection_mode,
            cv=self.cv,
            random_state=self.random_state,
        )
        return eb

    def __repr__(self) -> str:
        return (
            f"AutoEngine(cu={self.cu}, co={self.co}, tau={self.tau}, "
            f"selection_mode={self.selection_mode!r}, cv={self.cv}, "
            f"random_state={self.random_state!r}, speed={self.speed!r})"
        )
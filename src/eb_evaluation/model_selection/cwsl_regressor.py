from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .electric_barometer import ElectricBarometer
from ebmetrics.metrics import cwsl


class CWSLRegressor:
    """
    CWSLRegressor: scikit-learn-style estimator that selects among multiple
    candidate models using Cost-Weighted Service Loss (CWSL).

    It wraps ElectricBarometer and exposes a familiar fit / predict / score API.

    Parameters
    ----------
    models : dict[str, Any]
        Dictionary of candidate models or pipelines. Keys are model names,
        values are estimators or adapters implementing:

            fit(X, y)
            predict(X)

        This can be any mix of:
        - scikit-learn regressors (LinearRegression, RandomForest, etc.)
        - Pipelines
        - XGBoost / LightGBM via their sklearn APIs
        - Adapters like ProphetAdapter, SarimaxAdapter, etc.

    cu : float, default 2.0
        Underbuild (shortfall) cost per unit.

    co : float, default 1.0
        Overbuild (excess) cost per unit.

    tau : float, default 2.0
        Reserved for downstream diagnostics; mirrored into ElectricBarometer.

    training_mode : {"selection_only"}, default "selection_only"
        Currently only "selection_only" is supported by ElectricBarometer.

    refit_on_full : bool, default True
        In **holdout** mode:
            If True, after picking the best model on the validation set,
            the winner is refit on all data (train ∪ val) before being
            exposed via best_estimator_ and predict().
        In **cv** mode:
            Entire dataset is always used to refit the winning model.

    selection_mode : {"holdout", "cv"}, default "cv"
        - "holdout": CWSLRegressor will internally split (X, y) into
          train / validation using `validation_fraction` and call
          ElectricBarometer in holdout mode.
        - "cv": CWSLRegressor will pass the full (X, y) into ElectricBarometer
          in CV mode and let it handle K-fold splitting.

    cv : int, default 3
        Number of folds for CV when selection_mode="cv".

    validation_fraction : float, default 0.2
        Fraction of samples to use as validation when selection_mode="holdout".
        Must be in (0, 1). At least 1 sample (and at least 1 train sample)
        is always enforced.

    random_state : int or None, default None
        Seed for internal shuffling and CV splitting.

    Attributes
    ----------
    selector_ : ElectricBarometer
        The underlying selector instance used for this fit() call.

    best_name_ : str or None
        Name of the winning model among the candidates.

    best_estimator_ : Any or None
        The fitted winning model (or pipeline / adapter).

    results_ : pandas.DataFrame or None
        Comparison table from ElectricBarometer (CWSL / RMSE / wMAPE scores).

    validation_cwsl_ : float or None
        CWSL score of the winning model on validation (holdout) or mean CV.

    validation_rmse_ : float or None
        RMSE score of the winning model on validation or mean CV.

    validation_wmape_ : float or None
        wMAPE score of the winning model on validation or mean CV.

    n_features_in_ : int
        Number of features seen during fit().
    """

    def __init__(
        self,
        models: Dict[str, Any],
        cu: float = 2.0,
        co: float = 1.0,
        tau: float = 2.0,
        training_mode: str = "selection_only",
        refit_on_full: bool = True,
        selection_mode: str = "cv",
        cv: int = 3,
        validation_fraction: float = 0.2,
        random_state: Optional[int] = None,
    ) -> None:
        if not models:
            raise ValueError("CWSLRegressor requires at least one candidate model.")

        if selection_mode not in {"holdout", "cv"}:
            raise ValueError(
                f"selection_mode must be 'holdout' or 'cv'; got {selection_mode!r}."
            )

        if selection_mode == "holdout":
            if not (0.0 < validation_fraction < 1.0):
                raise ValueError(
                    "validation_fraction must lie strictly between 0 and 1; "
                    f"got {validation_fraction!r}."
                )

        self.models: Dict[str, Any] = models
        self.cu: float = float(cu)
        self.co: float = float(co)
        self.tau: float = float(tau)
        self.training_mode: str = training_mode
        self.refit_on_full: bool = bool(refit_on_full)
        self.selection_mode: str = selection_mode
        self.cv: int = int(cv)
        self.validation_fraction: float = float(validation_fraction)
        self.random_state: Optional[int] = random_state

        # Fitted attributes (populated during fit)
        self.selector_: Optional[ElectricBarometer] = None
        self.best_name_: Optional[str] = None
        self.best_estimator_: Any = None
        self.results_: Any = None

        self.validation_cwsl_: Optional[float] = None
        self.validation_rmse_: Optional[float] = None
        self.validation_wmape_: Optional[float] = None
        self.n_features_in_: Optional[int] = None

    # ------------------------------------------------------------------
    # Core estimator API
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CWSLRegressor":
        """
        Fit CWSLRegressor on (X, y) by internally using ElectricBarometer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional
            Optional per-sample weights. In CV mode they are passed through
            to ElectricBarometer so each validation fold is cost-weighted.
            In holdout mode they are currently ignored (matching EB behavior).

        Returns
        -------
        self : CWSLRegressor
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = X_arr.shape[0]
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"X and y have incompatible shapes: "
                f"X.shape[0]={n_samples}, y.shape[0]={y_arr.shape[0]}."
            )

        self.n_features_in_ = X_arr.shape[1]

        sw_arr: Optional[np.ndarray] = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
            if sw_arr.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight has length {sw_arr.shape[0]}, "
                    f"but X has {n_samples} rows."
                )

        # Create the underlying ElectricBarometer selector
        selector = ElectricBarometer(
            models=self.models,
            cu=self.cu,
            co=self.co,
            tau=self.tau,
            training_mode=self.training_mode,
            refit_on_full=self.refit_on_full,
            selection_mode=self.selection_mode,
            cv=self.cv,
            random_state=self.random_state,
        )

        if self.selection_mode == "holdout":
            # Manual train/validation split using validation_fraction
            n_val = int(round(self.validation_fraction * n_samples))
            if n_val <= 0:
                n_val = 1
            if n_val >= n_samples:
                n_val = n_samples - 1

            rng = np.random.default_rng(self.random_state)
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            sw_train = sw_arr[train_idx] if sw_arr is not None else None
            sw_val = sw_arr[val_idx] if sw_arr is not None else None

            selector.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                sample_weight_train=sw_train,
                sample_weight_val=sw_val,
            )
        else:
            # CV mode: ElectricBarometer handles its own K-fold splitting.
            selector.fit(
                X_train=X_arr,
                y_train=y_arr,
                X_val=X_arr,  # ignored in cv mode
                y_val=y_arr,  # ignored in cv mode
                sample_weight_train=sw_arr,
                sample_weight_val=None,
            )

        self.selector_ = selector
        self.best_name_ = selector.best_name_
        self.best_estimator_ = selector.best_model_
        self.results_ = selector.results_

        self.validation_cwsl_ = selector.validation_cwsl_
        self.validation_rmse_ = selector.validation_rmse_
        self.validation_wmape_ = selector.validation_wmape_

        return self

    def predict(self, X) -> np.ndarray:
        """
        Generate predictions from the selected best model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if self.best_estimator_ is None:
            raise RuntimeError(
                "CWSLRegressor has not been fit yet. Call .fit(X, y) first."
            )

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_pred = self.best_estimator_.predict(X_arr)
        return np.asarray(y_pred, dtype=float)

    def score(
        self,
        X,
        y,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Return a sklearn-style score using **negative CWSL**.

        Higher is better (less cost), consistent with sklearn conventions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like, optional

        Returns
        -------
        score : float
            Negative CWSL on the provided data.
        """
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} "
                "are not compatible."
            )

        sw_arr: Optional[np.ndarray] = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)

        cost = cwsl(
            y_true=y_true,
            y_pred=y_pred,
            cu=self.cu,
            co=self.co,
            sample_weight=sw_arr,
        )
        return -float(cost)

    # ------------------------------------------------------------------
    # Convenience properties & parameter API
    # ------------------------------------------------------------------
    @property
    def r_(self) -> float:
        """Cost ratio R = cu / co."""
        return self.cu / self.co

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Minimal sklearn-compatible get_params implementation."""
        return {
            "models": self.models,
            "cu": self.cu,
            "co": self.co,
            "tau": self.tau,
            "training_mode": self.training_mode,
            "refit_on_full": self.refit_on_full,
            "selection_mode": self.selection_mode,
            "cv": self.cv,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "CWSLRegressor":
        """Minimal sklearn-compatible set_params implementation."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter {key!r} for CWSLRegressor. "
                    f"Valid parameters are: {list(self.get_params().keys())}"
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        model_names = list(self.models.keys())
        return (
            "CWSLRegressor("
            f"models={model_names}, "
            f"cu={self.cu}, co={self.co}, "
            f"selection_mode={self.selection_mode!r}, "
            f"cv={self.cv}, "
            f"validation_fraction={self.validation_fraction}, "
            f"random_state={self.random_state!r}"
            ")"
        )
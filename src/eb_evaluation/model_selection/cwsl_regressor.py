r"""
scikit-learn-style estimator wrapper for cost-aware model selection.

This module defines `CWSLRegressor`, a lightweight wrapper around `ElectricBarometer`
that exposes a familiar ``fit`` / ``predict`` / ``score`` API.

The estimator selects among a set of candidate models using Cost-Weighted Service Loss (CWSL)
as the selection criterion. Candidate models are trained using their native objectives
(e.g., squared error), but selection is performed using asymmetric operational costs.

The primary cost preference can be summarized by the cost ratio:

$$
    R = \frac{c_u}{c_o}
$$

where $c_u$ is the underbuild (shortfall) cost per unit and $c_o$ is the overbuild (excess)
cost per unit.
"""

from typing import Any

import numpy as np

from eb_metrics.metrics import cwsl

from .electric_barometer import ElectricBarometer


class CWSLRegressor:
    r"""
    scikit-learn-style estimator that selects among candidate models using CWSL.

    This class wraps `ElectricBarometer` and exposes:

    - ``fit(X, y)``: perform cost-aware model selection
    - ``predict(X)``: predict using the selected best estimator
    - ``score(X, y)``: return a sklearn-style score based on **negative CWSL**

    Parameters
    ----------
    models : dict[str, Any]
        Mapping of candidate model name to an **unfitted** estimator implementing:

        - ``fit(X, y)``
        - ``predict(X)``

        Models may be scikit-learn regressors, pipelines, or EB adapters.
    cu : float, default=2.0
        Underbuild (shortfall) cost per unit. Must be strictly positive.
    co : float, default=1.0
        Overbuild (excess) cost per unit. Must be strictly positive.
    tau : float, default=2.0
        Tolerance parameter forwarded to ElectricBarometer for optional diagnostics (e.g., HR@tau).
    training_mode : {"selection_only"}, default="selection_only"
        Training behavior; currently ElectricBarometer supports selection-only mode.
    refit_on_full : bool, default=True
        Refit behavior after selection.

        - In holdout mode: if True, refit the winning model on train + validation.
        - In CV mode: the winning model is refit on the full dataset.
    selection_mode : {"holdout", "cv"}, default="cv"
        How selection is performed:

        - ``"holdout"``: split internally using ``validation_fraction``.
        - ``"cv"``: perform K-fold selection inside ElectricBarometer.
    cv : int, default=3
        Number of folds when ``selection_mode="cv"``.
    validation_fraction : float, default=0.2
        Fraction of samples used for validation when ``selection_mode="holdout"``.
        Must be in (0, 1).
    random_state : int | None, default=None
        Seed used for internal shuffling and CV splitting.

    Attributes
    ----------
    selector_ : ElectricBarometer | None
        Underlying selector instance used in the most recent fit.
    best_name_ : str | None
        Name of the winning model.
    best_estimator_ : Any | None
        Fitted winning estimator.
    results_ : Any
        Comparison table produced by ElectricBarometer (typically a pandas DataFrame).
    validation_cwsl_ : float | None
        CWSL score of the winning model on validation (holdout) or mean CV.
    validation_rmse_ : float | None
        RMSE score of the winning model on validation or mean CV.
    validation_wmape_ : float | None
        wMAPE score of the winning model on validation or mean CV.
    n_features_in_ : int | None
        Number of features observed during fit.

    Notes
    -----
    ``score`` returns **negative CWSL** to align with sklearn conventions (higher is better).
    """

    def __init__(
        self,
        models: dict[str, Any],
        cu: float = 2.0,
        co: float = 1.0,
        tau: float = 2.0,
        training_mode: str = "selection_only",
        refit_on_full: bool = True,
        selection_mode: str = "cv",
        cv: int = 3,
        validation_fraction: float = 0.2,
        random_state: int | None = None,
    ) -> None:
        if not models:
            raise ValueError("CWSLRegressor requires at least one candidate model.")

        if selection_mode not in {"holdout", "cv"}:
            raise ValueError(
                f"selection_mode must be 'holdout' or 'cv'; got {selection_mode!r}."
            )

        if selection_mode == "holdout" and not (0.0 < validation_fraction < 1.0):
            raise ValueError(
                "validation_fraction must lie strictly between 0 and 1; "
                f"got {validation_fraction!r}."
            )

        if cu <= 0 or co <= 0:
            raise ValueError("cu and co must be strictly positive.")

        if cv < 2 and selection_mode == "cv":
            raise ValueError(f"cv must be >= 2 when selection_mode='cv'; got {cv}.")

        self.models: dict[str, Any] = models
        self.cu: float = float(cu)
        self.co: float = float(co)
        self.tau: float = float(tau)
        self.training_mode: str = training_mode
        self.refit_on_full: bool = bool(refit_on_full)
        self.selection_mode: str = selection_mode
        self.cv: int = int(cv)
        self.validation_fraction: float = float(validation_fraction)
        self.random_state: int | None = random_state

        # Fitted attributes
        self.selector_: ElectricBarometer | None = None
        self.best_name_: str | None = None
        self.best_estimator_: Any = None
        self.results_: Any = None

        self.validation_cwsl_: float | None = None
        self.validation_rmse_: float | None = None
        self.validation_wmape_: float | None = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        X,
        y,
        sample_weight: np.ndarray | None = None,
    ) -> "CWSLRegressor":
        """
        Fit CWSLRegressor on (X, y) by delegating selection to ElectricBarometer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target vector.
        sample_weight : numpy.ndarray of shape (n_samples,), optional
            Optional per-sample weights. In CV mode these are passed through to
            ElectricBarometer so validation folds are cost-weighted. In holdout mode,
            they are passed through to ElectricBarometer (behavior depends on EB).

        Returns
        -------
        CWSLRegressor
            Fitted instance.

        Raises
        ------
        ValueError
            If X/y shapes are incompatible or sample_weight length mismatches.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = X_arr.shape[0]
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"X and y have incompatible shapes: X.shape[0]={n_samples}, y.shape[0]={y_arr.shape[0]}."
            )

        self.n_features_in_ = int(X_arr.shape[1])

        sw_arr: np.ndarray | None = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
            if sw_arr.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight has length {sw_arr.shape[0]}, but X has {n_samples} rows."
                )

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
            n_val = round(self.validation_fraction * n_samples)
            n_val = max(1, min(n_val, n_samples - 1))

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
        Generate predictions from the selected best estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted values.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
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
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """
        Compute a sklearn-style score using **negative CWSL**.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target vector.
        sample_weight : numpy.ndarray of shape (n_samples,), optional
            Optional per-sample weights passed to CWSL.

        Returns
        -------
        float
            Negative CWSL on the provided data (higher is better).
        """
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} are not compatible."
            )

        sw_arr: np.ndarray | None = None
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

    @property
    def r_(self) -> float:
        """
        Cost ratio.

        Returns
        -------
        float
            The ratio $R = c_u / c_o$.
        """
        return self.cu / self.co

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Minimal sklearn-compatible ``get_params`` implementation.

        Parameters
        ----------
        deep : bool, default=True
            Included for sklearn compatibility. Nested parameter extraction is not
            performed in this lightweight implementation.

        Returns
        -------
        dict[str, Any]
            Parameter mapping.
        """
        _ = deep
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
        """
        Minimal sklearn-compatible ``set_params`` implementation.

        Parameters
        ----------
        **params
            Parameters to set on this instance.

        Returns
        -------
        CWSLRegressor
            Updated instance.

        Raises
        ------
        ValueError
            If an invalid parameter name is provided.
        """
        valid = set(self.get_params().keys())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for CWSLRegressor. Valid parameters are: {sorted(valid)}"
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

r"""
Cost-aware model selection using the Electric Barometer workflow.

This module defines ``ElectricBarometer``, a lightweight selector that evaluates a set of
candidate regressors using Cost-Weighted Service Loss (CWSL) as the primary objective and
selects the model that minimizes expected operational cost.

Selection preference is governed by asymmetric unit costs:

- ``cu``: underbuild (shortfall) cost per unit
- ``co``: overbuild (excess) cost per unit

A convenient summary is the cost ratio:

$$
    R = \frac{c_u}{c_o}
$$

Notes
-----
ElectricBarometer is intentionally a selector (not a trainer that optimizes CWSL directly).
Candidate models are trained using their native objectives (e.g., squared error) and are
selected using CWSL on validation data (holdout) or across folds (CV).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eb_adapters import clone_model as _clone_model
from eb_metrics.metrics import cwsl, rmse, wmape

from .compare import select_model_by_cwsl


class ElectricBarometer:
    r"""
    Cost-aware selector that chooses the best model by minimizing CWSL.

    ElectricBarometer evaluates each candidate model on either:

    - a provided train/validation split (``selection_mode="holdout"``), or
    - K-fold cross-validation on the provided dataset (``selection_mode="cv"``),

    and selects the model with the lowest Cost-Weighted Service Loss (CWSL).
    For interpretability, it also reports reference symmetric diagnostics (RMSE, wMAPE).

    Operational preference is captured by asymmetric costs and the induced ratio:

    $$
        R = \frac{c_u}{c_o}
    $$

    Parameters
    ----------
    models : dict[str, Any]
        Mapping of candidate model name to an unfitted estimator implementing:

        - ``fit(X, y)``
        - ``predict(X)``

        Models can be scikit-learn regressors/pipelines or EB adapters implementing
        the same interface.
    cu : float, default=2.0
        Underbuild (shortfall) cost per unit. Must be strictly positive.
    co : float, default=1.0
        Overbuild (excess) cost per unit. Must be strictly positive.
    tau : float, default=2.0
        Reserved for downstream diagnostics (e.g., HR@τ) that may be integrated
        into selection reporting. Currently not used in the selection criterion.
    training_mode : {"selection_only"}, default="selection_only"
        Training behavior. In the current implementation, candidate models are trained
        using their native objectives and only selection is cost-aware.
    refit_on_full : bool, default=False
        Refit behavior in holdout mode:

        - If True, after selecting the best model by validation CWSL, refit a fresh clone
          of the winning model on train and validation.
        - If False, keep the fitted winning model as trained on the training split
          (and selected on the validation split).

        In CV mode, the selected model is always refit on the full dataset provided to
        ``fit`` (i.e., ``X_train, y_train``).
    selection_mode : {"holdout", "cv"}, default="holdout"
        Selection strategy:

        - ``"holdout"``: use the provided ``(X_train, y_train, X_val, y_val)``.
        - ``"cv"``: ignore ``X_val, y_val`` and run K-fold selection on ``X_train, y_train``.
    cv : int, default=3
        Number of folds when ``selection_mode="cv"``. Must be at least 2.
    random_state : int | None, default=None
        Seed used for CV shuffling/splitting.

    Attributes
    ----------
    best_name_ : str | None
        Name of the winning model after calling ``fit``.
    best_model_ : Any | None
        Fitted estimator corresponding to ``best_name_``.
    results_ : pandas.DataFrame | None
        Per-model comparison table.

        - In holdout mode: output of ``select_model_by_cwsl``.
        - In CV mode: mean scores across folds with columns ``["CWSL", "RMSE", "wMAPE"]``.
    validation_cwsl_ : float | None
        CWSL of the winning model on validation (holdout) or mean across folds (CV).
    validation_rmse_ : float | None
        RMSE of the winning model on validation (holdout) or mean across folds (CV).
    validation_wmape_ : float | None
        wMAPE of the winning model on validation (holdout) or mean across folds (CV).
    """

    def __init__(
        self,
        models: dict[str, Any],
        cu: float = 2.0,
        co: float = 1.0,
        tau: float = 2.0,
        training_mode: str = "selection_only",
        refit_on_full: bool = False,
        selection_mode: str = "holdout",
        cv: int = 3,
        random_state: int | None = None,
    ) -> None:
        if not models:
            raise ValueError("ElectricBarometer requires at least one candidate model.")

        if training_mode != "selection_only":
            raise ValueError(
                "ElectricBarometer currently supports only training_mode='selection_only'."
            )

        if cu <= 0 or co <= 0:
            raise ValueError("cu and co must be strictly positive.")

        if selection_mode not in {"holdout", "cv"}:
            raise ValueError(
                f"selection_mode must be either 'holdout' or 'cv'; got {selection_mode!r}."
            )

        if selection_mode == "cv" and cv < 2:
            raise ValueError(f"In CV mode, cv must be at least 2; got {cv!r}.")

        self.models: dict[str, Any] = models
        self.cu: float = float(cu)
        self.co: float = float(co)
        self.tau: float = float(tau)
        self.training_mode: str = training_mode
        self.refit_on_full: bool = bool(refit_on_full)
        self.selection_mode: str = selection_mode
        self.cv: int = int(cv)
        self.random_state: int | None = random_state

        # Fitted state
        self.best_name_: str | None = None
        self.best_model_: Any | None = None
        self.results_: pd.DataFrame | None = None

        self.validation_cwsl_: float | None = None
        self.validation_rmse_: float | None = None
        self.validation_wmape_: float | None = None

    @property
    def r_(self) -> float:
        r"""
        Cost ratio.

        Returns
        -------
        float
            The ratio:

            $$
                R = \frac{c_u}{c_o}
            $$
        """
        return self.cu / self.co

    def _cv_evaluate_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Evaluate candidate models via simple K-fold CV and return mean scores.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : numpy.ndarray of shape (n_samples,)
            Target vector.
        sample_weight : numpy.ndarray of shape (n_samples,), optional
            Optional non-negative weights. When provided, weights are subset to each
            fold's validation indices and passed to CWSL. RMSE and wMAPE remain
            unweighted in the current eb-metrics implementation.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by model name with columns:

            - ``"CWSL"``: mean CWSL across folds
            - ``"RMSE"``: mean RMSE across folds
            - ``"wMAPE"``: mean wMAPE across folds

        Raises
        ------
        ValueError
            If ``cv`` is invalid for the number of samples, or if sample_weight has
            an incompatible length.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = X_arr.shape[0]
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have the same number of rows; got {n_samples} and {y_arr.shape[0]}."
            )

        k = int(self.cv)
        if k < 2 or k > n_samples:
            raise ValueError(
                f"Invalid number of folds cv={k} for n_samples={n_samples}."
            )

        sw_arr: np.ndarray | None = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
            if sw_arr.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight must have length {n_samples}; got {sw_arr.shape[0]}."
                )

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[: n_samples % k] += 1

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + int(fold_size)
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            folds.append((train_idx, val_idx))
            current = stop

        rows: list[dict[str, float | str]] = []

        for model_name, base_model in self.models.items():
            cwsl_scores: list[float] = []
            rmse_scores: list[float] = []
            wmape_scores: list[float] = []

            for train_idx, val_idx in folds:
                X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

                model = _clone_model(base_model)
                model.fit(X_tr, y_tr)
                y_pred = np.asarray(model.predict(X_va), dtype=float)

                sw_va = sw_arr[val_idx] if sw_arr is not None else None

                cwsl_scores.append(
                    float(
                        cwsl(
                            y_true=y_va,
                            y_pred=y_pred,
                            cu=self.cu,
                            co=self.co,
                            sample_weight=sw_va,
                        )
                    )
                )
                rmse_scores.append(float(rmse(y_true=y_va, y_pred=y_pred)))
                wmape_scores.append(float(wmape(y_true=y_va, y_pred=y_pred)))

            rows.append(
                {
                    "model": model_name,
                    "CWSL": float(np.mean(cwsl_scores)),
                    "RMSE": float(np.mean(rmse_scores)),
                    "wMAPE": float(np.mean(wmape_scores)),
                }
            )

        return pd.DataFrame(rows).set_index("model")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        refit_on_full: bool | None = None,
    ) -> ElectricBarometer:
        """
        Fit candidate models and select the best one by minimizing CWSL.
        """
        _ = sample_weight_val  # reserved for future use

        refit_flag = (
            self.refit_on_full if refit_on_full is None else bool(refit_on_full)
        )

        if self.selection_mode == "holdout":
            best_name, best_model, results = select_model_by_cwsl(
                models=self.models,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                cu=self.cu,
                co=self.co,
                sample_weight_val=None,
            )

            self.best_name_ = best_name
            self.results_ = results

            self.validation_cwsl_ = None
            self.validation_rmse_ = None
            self.validation_wmape_ = None

            try:
                row = results.loc[best_name]
                if "CWSL" in row:
                    self.validation_cwsl_ = float(row["CWSL"])
                if "RMSE" in row:
                    self.validation_rmse_ = float(row["RMSE"])
                if "wMAPE" in row:
                    self.validation_wmape_ = float(row["wMAPE"])
            except Exception:
                pass

            best_model_refit = best_model
            if refit_flag and hasattr(best_model_refit, "fit"):
                X_full = np.concatenate(
                    [np.asarray(X_train), np.asarray(X_val)], axis=0
                )
                y_full = np.concatenate(
                    [np.asarray(y_train, dtype=float), np.asarray(y_val, dtype=float)],
                    axis=0,
                )

                best_model_refit = _clone_model(best_model_refit)
                best_model_refit.fit(X_full, y_full)

            self.best_model_ = best_model_refit
            return self

        results = self._cv_evaluate_models(
            X=np.asarray(X_train),
            y=np.asarray(y_train, dtype=float),
            sample_weight=sample_weight_train,
        )
        self.results_ = results

        best_name = results["CWSL"].idxmin()
        self.best_name_ = str(best_name)

        row = results.loc[best_name]
        self.validation_cwsl_ = float(row["CWSL"])
        self.validation_rmse_ = float(row["RMSE"])
        self.validation_wmape_ = float(row["wMAPE"])

        base_model = self.models[self.best_name_]
        best_model_refit = _clone_model(base_model)
        best_model_refit.fit(np.asarray(X_train), np.asarray(y_train, dtype=float))
        self.best_model_ = best_model_refit

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the selected best model."""
        if self.best_model_ is None:
            raise RuntimeError(
                "ElectricBarometer has not been fit yet. Call .fit(...) first (holdout or cv mode)."
            )

        y_pred = self.best_model_.predict(X)
        return np.asarray(y_pred, dtype=float)

    def cwsl_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
        cu: float | None = None,
        co: float | None = None,
    ) -> float:
        """Compute CWSL using this selector's costs (or overrides)."""
        cu_eff = float(self.cu if cu is None else cu)
        co_eff = float(self.co if co is None else co)

        return float(
            cwsl(
                y_true=np.asarray(y_true, dtype=float),
                y_pred=np.asarray(y_pred, dtype=float),
                cu=cu_eff,
                co=co_eff,
                sample_weight=sample_weight,
            )
        )

    def __repr__(self) -> str:
        model_names = list(self.models.keys())
        best = self.best_name_ if self.best_name_ is not None else "None"
        return (
            f"ElectricBarometer(models={model_names}, "
            f"cu={self.cu}, co={self.co}, tau={self.tau}, "
            f"refit_on_full={self.refit_on_full}, "
            f"selection_mode={self.selection_mode!r}, "
            f"cv={self.cv}, random_state={self.random_state!r}, "
            f"best_name_={best!r})"
        )

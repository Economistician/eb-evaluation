from __future__ import annotations

from typing import Iterable, Mapping, Union, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from ebmetrics.metrics import (
    cwsl,
    nsl,
    ud,
    wmape,
    hr_at_tau,
    frs,
    mae,
    rmse,
    mape,
)

ArrayLike = Union[Iterable[float], np.ndarray]


def compare_forecasts(
    y_true: ArrayLike,
    forecasts: Mapping[str, ArrayLike],
    cu: Union[float, ArrayLike],
    co: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
    tau: Union[float, ArrayLike] = 2.0,
) -> pd.DataFrame:
    """
    Compare multiple forecast models on the same target series using
    CWSL and related metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand values.

    forecasts : mapping from str to array-like
        Dictionary mapping model names to their forecasted values.
        Each value must be array-like of shape (n_samples,).

        Example
        -------
        >>> forecasts = {
        ...     "model_a": [9, 15, 7],
        ...     "model_b": [10, 12, 8],
        ... }

    cu : float or array-like of shape (n_samples,)
        Underbuild (shortfall) cost per unit.

    co : float or array-like of shape (n_samples,)
        Overbuild (excess) cost per unit.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. These are passed
        through to metrics that support a sample_weight argument
        (CWSL, NSL, UD, HR@tau, FRS, MAE).

    tau : float or array-like, optional (default = 2.0)
        Tolerance parameter for HR@τ.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by model name with columns:
        ["CWSL", "NSL", "UD", "wMAPE", "HR@tau", "FRS", "MAE", "RMSE", "MAPE"].
    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional; got shape {y_true_arr.shape}")

    if not forecasts:
        raise ValueError("forecasts mapping is empty; provide at least one model.")

    rows: Dict[str, Dict[str, float]] = {}

    for model_name, y_pred in forecasts.items():
        # Metrics that accept sample_weight get it; others are called without.
        metrics_row = {
            "CWSL": cwsl(y_true_arr, y_pred, cu=cu, co=co, sample_weight=sample_weight),
            "NSL": nsl(y_true_arr, y_pred, sample_weight=sample_weight),
            "UD": ud(y_true_arr, y_pred, sample_weight=sample_weight),
            "wMAPE": wmape(y_true_arr, y_pred),
            "HR@tau": hr_at_tau(
                y_true_arr, y_pred, tau=tau, sample_weight=sample_weight
            ),
            "FRS": frs(y_true_arr, y_pred, cu=cu, co=co, sample_weight=sample_weight),
            "MAE": mae(y_true_arr, y_pred),
            "RMSE": rmse(y_true_arr, y_pred),
            "MAPE": mape(y_true_arr, y_pred),
        }
        rows[model_name] = metrics_row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    return df


def select_model_by_cwsl(
    models: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    cu: float,
    co: float,
    sample_weight_val=None,
) -> Tuple[str, Any, pd.DataFrame]:
    """
    Fit multiple models normally, then select the best one based on CWSL
    evaluated on a validation set.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping from model name to an unfitted estimator object that
        implements .fit(X, y) and .predict(X). These can be scikit-learn
        style estimators or any object with that interface.

    X_train, y_train :
        Training data used to fit each model using its native loss
        (typically MSE / RMSE).

    X_val, y_val :
        Validation data used only for evaluation. CWSL and baseline
        metrics are computed on (y_val, model.predict(X_val)).

    cu : float
        Underbuild (shortfall) cost per unit for CWSL.

    co : float
        Overbuild (excess) cost per unit for CWSL.

    sample_weight_val : array-like or None, optional
        Optional sample weights for the validation set, passed into
        the metrics that support them.

    Returns
    -------
    best_name : str
        Name of the model with the lowest CWSL on the validation set.

    best_model : estimator
        The fitted estimator corresponding to `best_name`.

    results : pandas.DataFrame
        DataFrame indexed by model name with columns:
        ['CWSL', 'RMSE', 'wMAPE'].
    """
    y_val_arr = np.asarray(y_val, dtype=float)

    rows = []
    best_name: Optional[str] = None
    best_model: Any | None = None
    best_cwsl = np.inf

    for name, model in models.items():
        # Fit the model normally on the training data
        fitted = model.fit(X_train, y_train)

        # Predict on validation set
        y_pred_val = np.asarray(fitted.predict(X_val), dtype=float)

        # CWSL uses weights if provided
        cwsl_val = cwsl(
            y_true=y_val_arr,
            y_pred=y_pred_val,
            cu=cu,
            co=co,
            sample_weight=sample_weight_val,
        )
        # rmse/wmape are currently unweighted in eb-metrics
        rmse_val = rmse(
            y_true=y_val_arr,
            y_pred=y_pred_val,
        )
        wmape_val = wmape(
            y_true=y_val_arr,
            y_pred=y_pred_val,
        )

        rows.append(
            {
                "model": name,
                "CWSL": cwsl_val,
                "RMSE": rmse_val,
                "wMAPE": wmape_val,
            }
        )

        if cwsl_val < best_cwsl:
            best_cwsl = cwsl_val
            best_name = name
            best_model = fitted

    results = pd.DataFrame(rows).set_index("model")

    if best_name is None or best_model is None:
        raise ValueError("No models were evaluated. Check the `models` dict.")

    return best_name, best_model, results


def select_model_by_cwsl_cv(
    models: Dict[str, Any],
    X,
    y,
    *,
    cu: float,
    co: float,
    cv: int = 5,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[str, Any, pd.DataFrame]:
    """
    Select a model by **cross-validated CWSL** instead of a single
    train/validation split.

    This is a simple K-fold CV engine that:

      * Splits the data into `cv` folds.
      * For each model and each fold:
          - Fits on (K-1) folds.
          - Evaluates on the held-out fold using CWSL, RMSE, and wMAPE.
      * Averages metrics across folds for each model.
      * Chooses the model with the **lowest mean CWSL**.
      * Refits that winning model once on the **full dataset (X, y)**.

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping from model name to an unfitted estimator object that
        implements .fit(X, y) and .predict(X). These can be scikit-learn
        style estimators or any object with that interface.

    X : array-like of shape (n_samples, n_features)
        Full feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    cu : float
        Underbuild (shortfall) cost per unit for CWSL.

    co : float
        Overbuild (excess) cost per unit for CWSL.

    cv : int, default 5
        Number of CV folds. Must be >= 2.

    sample_weight : array-like of shape (n_samples,), optional
        Optional per-sample weights used **only in the metric
        calculations** for CWSL; RMSE and wMAPE remain unweighted.

    Returns
    -------
    best_name : str
        Name of the model with the lowest mean CWSL across folds.

    best_model : estimator
        The estimator trained on **all (X, y)** for the chosen model.

    results : pandas.DataFrame
        DataFrame indexed by model name with columns:

            - 'CWSL_mean', 'CWSL_std'
            - 'RMSE_mean', 'RMSE_std'
            - 'wMAPE_mean', 'wMAPE_std'
            - 'n_folds'
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y must have the same number of rows; "
            f"got {X_arr.shape[0]} and {y_arr.shape[0]}"
        )

    n_samples = X_arr.shape[0]
    if cv < 2:
        raise ValueError(f"cv must be at least 2; got {cv}")

    indices = np.arange(n_samples)
    folds = np.array_split(indices, cv)

    if sample_weight is not None:
        sw_arr = np.asarray(sample_weight, dtype=float)
        if sw_arr.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight must have length {n_samples}; got {sw_arr.shape[0]}"
            )
    else:
        sw_arr = None

    rows = []
    best_name: Optional[str] = None
    best_model: Any | None = None
    best_cwsl_mean = np.inf

    for name, model in models.items():
        cwsl_scores = []
        rmse_scores = []
        wmape_scores = []

        for i, val_idx in enumerate(folds):
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])

            X_train = X_arr[train_idx]
            y_train = y_arr[train_idx]
            X_val = X_arr[val_idx]
            y_val = y_arr[val_idx]

            if sw_arr is not None:
                sw_val = sw_arr[val_idx]
            else:
                sw_val = None

            # Fit the model on this fold's training data (unweighted for now)
            fitted = model.fit(X_train, y_train)

            # Predict on validation fold
            y_pred_val = np.asarray(fitted.predict(X_val), dtype=float)

            # Metrics for this fold
            cwsl_val = cwsl(
                y_true=y_val,
                y_pred=y_pred_val,
                cu=cu,
                co=co,
                sample_weight=sw_val,
            )
            rmse_val = rmse(
                y_true=y_val,
                y_pred=y_pred_val,
            )
            wmape_val = wmape(
                y_true=y_val,
                y_pred=y_pred_val,
            )

            cwsl_scores.append(cwsl_val)
            rmse_scores.append(rmse_val)
            wmape_scores.append(wmape_val)

        cwsl_scores_arr = np.asarray(cwsl_scores, dtype=float)
        rmse_scores_arr = np.asarray(rmse_scores, dtype=float)
        wmape_scores_arr = np.asarray(wmape_scores, dtype=float)

        row = {
            "model": name,
            "CWSL_mean": float(np.mean(cwsl_scores_arr)),
            "CWSL_std": float(np.std(cwsl_scores_arr, ddof=0)),
            "RMSE_mean": float(np.mean(rmse_scores_arr)),
            "RMSE_std": float(np.std(rmse_scores_arr, ddof=0)),
            "wMAPE_mean": float(np.mean(wmape_scores_arr)),
            "wMAPE_std": float(np.std(wmape_scores_arr, ddof=0)),
            "n_folds": cv,
        }
        rows.append(row)

        if row["CWSL_mean"] < best_cwsl_mean:
            best_cwsl_mean = row["CWSL_mean"]
            best_name = name
            best_model = model  # will refit on full data below

    results = pd.DataFrame(rows).set_index("model")

    if best_name is None or best_model is None:
        raise ValueError("No models were evaluated. Check the `models` dict.")

    # Final refit of the best model on *all* data
    best_model.fit(X_arr, y_arr)

    return best_name, best_model, results
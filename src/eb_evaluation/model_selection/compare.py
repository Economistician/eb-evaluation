r"""
Forecast comparison and cost-aware model selection helpers.

This module provides evaluation-oriented utilities built on top of
``eb_metrics.metrics``:

- ``compare_forecasts`` computes CWSL and related diagnostics for multiple forecast
  vectors against a common target series.
- ``select_model_by_cwsl`` fits candidate estimators (using their native training
  objective) and selects the model with the lowest validation CWSL.
- ``select_model_by_cwsl_cv`` performs K-fold cross-validation, selecting the model
  with the lowest mean CWSL and refitting it on the full dataset.

CWSL is evaluated with asymmetric costs for underbuild and overbuild, typically summarized
by a cost ratio:

$$
    R = \frac{c_u}{c_o}
$$

where $c_u$ is the cost per unit of shortfall and $c_o$ is the cost per unit of excess.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from eb_metrics.metrics import (
    cwsl,
    frs,
    hr_at_tau,
    mae,
    mape,
    nsl,
    rmse,
    ud,
    wmape,
)

ArrayLike = Iterable[float] | np.ndarray


def compare_forecasts(
    y_true: ArrayLike,
    forecasts: Mapping[str, ArrayLike],
    cu: float | ArrayLike,
    co: float | ArrayLike,
    sample_weight: ArrayLike | None = None,
    tau: float | ArrayLike = 2.0,
) -> pd.DataFrame:
    r"""
    Compare multiple forecast models on the same target series.

    For each forecast vector, compute CWSL and a standard set of diagnostics:

    - CWSL
    - NSL
    - UD
    - wMAPE
    - HR@tau
    - FRS
    - MAE
    - RMSE
    - MAPE

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual (ground-truth) values.
    forecasts : Mapping[str, array-like]
        Mapping from model name to forecast vector. Each forecast must be shape ``(n_samples,)``.
    cu : float or array-like of shape (n_samples,)
        Underbuild (shortfall) cost per unit.
    co : float or array-like of shape (n_samples,)
        Overbuild (excess) cost per unit.
    sample_weight : array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. Passed to metrics that support
        ``sample_weight`` (CWSL, NSL, UD, HR@tau, FRS). Metrics that are currently unweighted
        in ``eb_metrics`` (e.g., wMAPE, MAE, RMSE, MAPE) are computed without weights.
    tau : float or array-like, default=2.0
        Tolerance parameter for HR@tau. May be scalar or per-interval.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by model name with columns:
        ``["CWSL", "NSL", "UD", "wMAPE", "HR@tau", "FRS", "MAE", "RMSE", "MAPE"]``.

    Raises
    ------
    ValueError
        If ``y_true`` is not 1D, if ``forecasts`` is empty, or if any forecast length
        is incompatible with ``y_true``.
    """
    y_true_arr = np.asarray(y_true, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional; got shape {y_true_arr.shape}")

    if not forecasts:
        raise ValueError("forecasts mapping is empty; provide at least one model.")

    n = y_true_arr.shape[0]

    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=float)
        if sw.ndim != 1 or sw.shape[0] != n:
            raise ValueError(
                f"sample_weight must be 1D with length {n}; got shape {sw.shape}"
            )
        sample_weight_val: ArrayLike | None = sw
    else:
        sample_weight_val = None

    rows: dict[str, dict[str, float]] = {}

    for model_name, y_pred in forecasts.items():
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_pred_arr.ndim != 1 or y_pred_arr.shape[0] != n:
            raise ValueError(
                f"Forecast {model_name!r} must be 1D with length {n}; got shape {y_pred_arr.shape}"
            )

        metrics_row = {
            "CWSL": float(
                cwsl(
                    y_true_arr,
                    y_pred_arr,
                    cu=cu,
                    co=co,
                    sample_weight=sample_weight_val,
                )
            ),
            "NSL": float(nsl(y_true_arr, y_pred_arr, sample_weight=sample_weight_val)),
            "UD": float(ud(y_true_arr, y_pred_arr, sample_weight=sample_weight_val)),
            # wMAPE in eb_metrics is unweighted
            "wMAPE": float(wmape(y_true_arr, y_pred_arr)),
            "HR@tau": float(
                hr_at_tau(
                    y_true_arr,
                    y_pred_arr,
                    tau=tau,
                    sample_weight=sample_weight_val,
                )
            ),
            "FRS": float(
                frs(
                    y_true_arr,
                    y_pred_arr,
                    cu=cu,
                    co=co,
                    sample_weight=sample_weight_val,
                )
            ),
            # Symmetric metrics are currently unweighted in eb_metrics
            "MAE": float(mae(y_true_arr, y_pred_arr)),
            "RMSE": float(rmse(y_true_arr, y_pred_arr)),
            "MAPE": float(mape(y_true_arr, y_pred_arr)),
        }
        rows[str(model_name)] = metrics_row

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "model"
    return out


def select_model_by_cwsl(
    models: dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    cu: float,
    co: float,
    sample_weight_val=None,
) -> tuple[str, Any, pd.DataFrame]:
    r"""
    Fit multiple models, then select the best by validation CWSL.

    Each estimator is fit on ``(X_train, y_train)`` using its native objective
    (typically MSE/RMSE), then evaluated on the validation set via CWSL:

    $$
        \text{CWSL} = \mathrm{cwsl}(y_{\mathrm{val}}, \hat{y}_{\mathrm{val}}; c_u, c_o)
    $$

    The model with the lowest CWSL is returned, along with a compact results table.

    Parameters
    ----------
    models : dict[str, Any]
        Mapping from model name to an **unfitted** estimator implementing:

        - ``fit(X, y)``
        - ``predict(X)``
    X_train, y_train
        Training data used to fit each model.
    X_val, y_val
        Validation data used only for evaluation.
    cu : float
        Underbuild (shortfall) cost per unit for CWSL.
    co : float
        Overbuild (excess) cost per unit for CWSL.
    sample_weight_val : array-like or None, optional
        Optional per-interval weights for the validation set, passed to CWSL.

    Returns
    -------
    best_name : str
        Name of the model with the lowest CWSL on the validation set.
    best_model : Any
        Fitted estimator corresponding to ``best_name``.
    results : pandas.DataFrame
        DataFrame indexed by model name with columns ``["CWSL", "RMSE", "wMAPE"]``.

    Raises
    ------
    ValueError
        If no models are evaluated.

    Notes
    -----
    - RMSE and wMAPE are computed unweighted (consistent with current eb_metrics behavior).
    - This function is intentionally simple and does not handle time-series splitting; callers
      should ensure the split is appropriate.
    """
    y_val_arr = np.asarray(y_val, dtype=float)

    rows = []
    best_name: str | None = None
    best_model: Any | None = None
    best_cwsl = np.inf

    for name, model in models.items():
        fitted = model.fit(X_train, y_train)
        y_pred_val = np.asarray(fitted.predict(X_val), dtype=float)

        cwsl_val = float(
            cwsl(
                y_true=y_val_arr,
                y_pred=y_pred_val,
                cu=cu,
                co=co,
                sample_weight=sample_weight_val,
            )
        )
        rmse_val = float(rmse(y_true=y_val_arr, y_pred=y_pred_val))
        wmape_val = float(wmape(y_true=y_val_arr, y_pred=y_pred_val))

        rows.append(
            {"model": name, "CWSL": cwsl_val, "RMSE": rmse_val, "wMAPE": wmape_val}
        )

        if cwsl_val < best_cwsl:
            best_cwsl = cwsl_val
            best_name = str(name)
            best_model = fitted

    results = pd.DataFrame(rows).set_index("model")

    if best_name is None or best_model is None:
        raise ValueError("No models were evaluated. Check the `models` dict.")

    return best_name, best_model, results


def select_model_by_cwsl_cv(
    models: dict[str, Any],
    X,
    y,
    *,
    cu: float,
    co: float,
    cv: int = 5,
    sample_weight: np.ndarray | None = None,
) -> tuple[str, Any, pd.DataFrame]:
    r"""
    Select a model by cross-validated CWSL and refit on the full dataset.

    This is a simple K-fold cross-validation loop:

    1. Split indices into ``cv`` folds.
    2. For each model and fold:
       - fit on (cv - 1) folds
       - evaluate on the held-out fold using CWSL, RMSE, and wMAPE
    3. Aggregate metrics across folds for each model.
    4. Choose the model with the **lowest mean CWSL**.
    5. Refit the chosen model once on all data ``(X, y)``.

    Parameters
    ----------
    models : dict[str, Any]
        Mapping from model name to an **unfitted** estimator implementing
        ``fit`` and ``predict``.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    cu : float
        Underbuild (shortfall) cost per unit for CWSL.
    co : float
        Overbuild (excess) cost per unit for CWSL.
    cv : int, default=5
        Number of folds. Must be >= 2.
    sample_weight : numpy.ndarray of shape (n_samples,), optional
        Optional per-sample weights used **only** for CWSL metric calculation.
        RMSE and wMAPE remain unweighted.

    Returns
    -------
    best_name : str
        Model name with the lowest mean CWSL across folds.
    best_model : Any
        The chosen estimator refit on all data.
    results : pandas.DataFrame
        DataFrame indexed by model name with columns:

        - ``CWSL_mean``, ``CWSL_std``
        - ``RMSE_mean``, ``RMSE_std``
        - ``wMAPE_mean``, ``wMAPE_std``
        - ``n_folds``

    Raises
    ------
    ValueError
        If X/y dimensions mismatch, ``cv < 2``, sample_weight length mismatch,
        or no models are evaluated.

    Notes
    -----
    This function uses a naive split of indices into contiguous folds via
    ``numpy.array_split``. For time-series problems, callers should prefer
    time-aware splitting outside this helper.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y must have the same number of rows; got {X_arr.shape[0]} and {y_arr.shape[0]}"
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
    best_name: str | None = None
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

            sw_val = sw_arr[val_idx] if sw_arr is not None else None

            fitted = model.fit(X_train, y_train)
            y_pred_val = np.asarray(fitted.predict(X_val), dtype=float)

            cwsl_scores.append(
                float(
                    cwsl(
                        y_true=y_val,
                        y_pred=y_pred_val,
                        cu=cu,
                        co=co,
                        sample_weight=sw_val,
                    )
                )
            )
            rmse_scores.append(float(rmse(y_true=y_val, y_pred=y_pred_val)))
            wmape_scores.append(float(wmape(y_true=y_val, y_pred=y_pred_val)))

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
            "n_folds": int(cv),
        }
        rows.append(row)

        if row["CWSL_mean"] < best_cwsl_mean:
            best_cwsl_mean = row["CWSL_mean"]
            best_name = str(name)
            best_model = model  # refit below

    results = pd.DataFrame(rows).set_index("model")

    if best_name is None or best_model is None:
        raise ValueError("No models were evaluated. Check the `models` dict.")

    best_model.fit(X_arr, y_arr)

    return best_name, best_model, results

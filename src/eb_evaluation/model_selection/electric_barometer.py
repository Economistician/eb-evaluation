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
selected using a chosen selection objective on validation data (holdout) or across folds (CV).
"""

from __future__ import annotations

from collections.abc import Sized
import time
from typing import Any, Literal, cast
import warnings

import numpy as np
import pandas as pd

from eb_adapters import clone_model as _clone_model
from eb_metrics.metrics import cwsl, rmse, wmape


class ElectricBarometer:
    r"""
    Cost-aware selector that chooses the best model by minimizing a selection objective.

    ElectricBarometer evaluates each candidate model on either:

    - a provided train/validation split (``selection_mode="holdout"``), or
    - K-fold cross-validation on the provided dataset (``selection_mode="cv"``),

    and selects the model with the best (lowest) score under the chosen selection objective.
    For interpretability, it also reports reference diagnostics (CWSL, RMSE, wMAPE).

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
    include : set[str] | None, default=None
        Optional allowlist of model names to include from ``models``. If provided,
        only these names are retained (after validation).
    exclude : set[str] | None, default=None
        Optional blocklist of model names to exclude from ``models`` (after validation).
        Applied after ``include`` filtering.
    metric : {"cwsl", "rmse", "wmape"}, default="cwsl"
        Selection objective used to choose the winning model.
        All metrics are computed and reported; this parameter determines which column
        is optimized.
    tie_tol : float, default=0.0
        Absolute tolerance applied to the selection metric when determining ties.
        Any model with score <= (best_score + tie_tol) is considered tied.
    tie_breaker : {"metric", "simpler", "name"}, default="metric"
        How to break ties among models within ``tie_tol`` of the best score.

        - ``"metric"``: choose the tied model with the lowest metric
          (deterministic by insertion/index order)
        - ``"simpler"``: prefer a "simpler" model based on a lightweight heuristic
        - ``"name"``: choose lexicographically smallest model name
    validate_inputs : {"strict", "coerce", "off"}, default="strict"
        Input validation level.

        - ``"strict"``: require numeric arrays and error on NaN/inf
        - ``"coerce"``: coerce to float and error on NaN/inf
        - ``"off"``: minimal validation (legacy behavior)
    error_policy : {"raise", "skip", "warn_skip"}, default="warn_skip"
        Behavior when a candidate model fails to fit/predict or otherwise errors.

        - ``"raise"``: raise immediately
        - ``"skip"``: skip failing models silently (recorded in ``failures_``)
        - ``"warn_skip"``: warn and skip (recorded in ``failures_``)
    time_budget_s : float | None, default=None
        Optional wall-clock time budget (seconds) for the full selection run. If exceeded,
        remaining models are not evaluated.
        Note: this cannot forcibly interrupt a model already running; it gates starting
        new candidates and can mark a candidate as timed out if it exceeds budgets.
    per_model_time_budget_s : float | None, default=None
        Optional wall-clock time budget (seconds) per candidate model (across folds in CV).
        If exceeded, that model is marked as timed out and skipped (or raises under
        ``error_policy="raise"``).
    cu : float, default=2.0
        Underbuild (shortfall) cost per unit. Must be strictly positive.
    co : float, default=1.0
        Overbuild (excess) cost per unit. Must be strictly positive.
    tau : float, default=2.0
        Reserved for downstream diagnostics (e.g., HR@τ) that may be integrated
        into selection reporting. Currently not used in the selection criterion.
    training_mode : {"selection_only"}, default="selection_only"
        Training behavior. In the current implementation, candidate models are trained
        using their native objectives and only selection is external.
    refit_on_full : bool, default=False
        Refit behavior in holdout mode:

        - If True, after selecting the best model by the chosen metric on validation data,
          refit a fresh clone of the winning model on train and validation.
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

        - In holdout mode: one row per model with columns ``["CWSL", "RMSE", "wMAPE"]``
        - In CV mode: mean scores across folds with the same columns
    failures_ : dict[str, str]
        Mapping of model name to a failure reason for models that errored or timed out.
    validation_cwsl_ : float | None
        CWSL of the winning model on validation (holdout) or mean across folds (CV).
    validation_rmse_ : float | None
        RMSE of the winning model on validation (holdout) or mean across folds (CV).
    validation_wmape_ : float | None
        wMAPE of the winning model on validation (holdout) or mean across folds (CV).

    candidate_names_ : list[str]
        Names of candidate models remaining after include/exclude filtering.
    evaluated_names_ : list[str]
        Names of models that were actually attempted during the most recent ``fit``.
    stopped_early_ : bool
        Whether evaluation stopped early due to the global time budget.
    stop_reason_ : str | None
        If ``stopped_early_`` is True, a human-readable reason string.
    """

    _MetricName = Literal["cwsl", "rmse", "wmape"]
    _ErrorPolicy = Literal["raise", "skip", "warn_skip"]
    _TieBreaker = Literal["metric", "simpler", "name"]
    _ValidateInputs = Literal["strict", "coerce", "off"]

    _SCORE_COLUMNS: tuple[str, str, str] = ("CWSL", "RMSE", "wMAPE")

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
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        metric: _MetricName = "cwsl",
        tie_tol: float = 0.0,
        tie_breaker: _TieBreaker = "metric",
        validate_inputs: _ValidateInputs = "strict",
        error_policy: _ErrorPolicy = "warn_skip",
        time_budget_s: float | None = None,
        per_model_time_budget_s: float | None = None,
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

        if metric not in {"cwsl", "rmse", "wmape"}:
            raise ValueError(f"metric must be one of {{'cwsl','rmse','wmape'}}; got {metric!r}.")

        if float(tie_tol) < 0:
            raise ValueError("tie_tol must be non-negative.")

        if tie_breaker not in {"metric", "simpler", "name"}:
            raise ValueError(
                f"tie_breaker must be one of {{'metric','simpler','name'}}; got {tie_breaker!r}."
            )

        if validate_inputs not in {"strict", "coerce", "off"}:
            raise ValueError(
                "validate_inputs must be one of {'strict','coerce','off'}; "
                f"got {validate_inputs!r}."
            )

        if error_policy not in {"raise", "skip", "warn_skip"}:
            raise ValueError(
                f"error_policy must be one of {{'raise','skip','warn_skip'}}; got {error_policy!r}."
            )

        if time_budget_s is not None and float(time_budget_s) <= 0:
            raise ValueError("time_budget_s must be positive when provided.")

        if per_model_time_budget_s is not None and float(per_model_time_budget_s) <= 0:
            raise ValueError("per_model_time_budget_s must be positive when provided.")

        # ---- include/exclude filtering (validated, deterministic) ----
        model_names = set(models.keys())

        if include is not None:
            include_set = set(include)
            unknown = sorted(include_set - model_names)
            if unknown:
                raise ValueError(
                    "include contains unknown model names: "
                    f"{unknown}. Available: {sorted(model_names)}"
                )
            filtered: dict[str, Any] = {k: models[k] for k in models if k in include_set}
        else:
            filtered = dict(models)

        if exclude is not None:
            exclude_set = set(exclude)
            unknown = sorted(exclude_set - model_names)
            if unknown:
                raise ValueError(
                    "exclude contains unknown model names: "
                    f"{unknown}. Available: {sorted(model_names)}"
                )
            filtered = {k: v for k, v in filtered.items() if k not in exclude_set}

        if not filtered:
            raise ValueError(
                "No candidate models remain after include/exclude filtering. "
                f"Original: {sorted(models.keys())}"
            )

        self.models: dict[str, Any] = filtered
        self.cu: float = float(cu)
        self.co: float = float(co)
        self.tau: float = float(tau)
        self.training_mode: str = training_mode
        self.refit_on_full: bool = bool(refit_on_full)
        self.selection_mode: str = selection_mode
        self.cv: int = int(cv)
        self.random_state: int | None = random_state

        self.metric: ElectricBarometer._MetricName = metric
        self.tie_tol: float = float(tie_tol)
        self.tie_breaker: ElectricBarometer._TieBreaker = tie_breaker
        self.validate_inputs: ElectricBarometer._ValidateInputs = validate_inputs

        self.error_policy: ElectricBarometer._ErrorPolicy = error_policy
        self.time_budget_s: float | None = None if time_budget_s is None else float(time_budget_s)
        self.per_model_time_budget_s: float | None = (
            None if per_model_time_budget_s is None else float(per_model_time_budget_s)
        )

        # Fitted state
        self.best_name_: str | None = None
        self.best_model_: Any | None = None
        self.results_: pd.DataFrame | None = None
        self.failures_: dict[str, str] = {}

        self.validation_cwsl_: float | None = None
        self.validation_rmse_: float | None = None
        self.validation_wmape_: float | None = None

        # Run audit state (populated/reset on each fit)
        self.candidate_names_: list[str] = list(self.models.keys())
        self.evaluated_names_: list[str] = []
        self.stopped_early_: bool = False
        self.stop_reason_: str | None = None

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

    def _handle_failure(self, *, model_name: str, reason: str) -> None:
        self.failures_[model_name] = reason
        if self.error_policy == "raise":
            raise RuntimeError(f"Model {model_name!r} failed: {reason}")
        if self.error_policy == "warn_skip":
            warnings.warn(f"Skipping model {model_name!r}: {reason}", RuntimeWarning, stacklevel=2)

    def _ensure_numeric_finite(
        self,
        *,
        name: str,
        arr: np.ndarray,
        require_1d: bool = False,
        require_2d: bool = False,
    ) -> np.ndarray:
        """
        Validate numeric dtype and finite values according to validate_inputs.

        - strict: require numeric dtype and finite
        - coerce: coerce to float then require finite
        - off: return as-is
        """
        if self.validate_inputs == "off":
            return arr

        if self.validate_inputs == "coerce":
            arr2 = np.asarray(arr, dtype=float)
        else:  # "strict"
            arr2 = np.asarray(arr)
            if not np.issubdtype(arr2.dtype, np.number):
                raise ValueError(
                    f"{name} must be numeric under validate_inputs='strict'; got {arr2.dtype}."
                )

        if require_1d and arr2.ndim != 1:
            raise ValueError(f"{name} must be 1D; got shape {arr2.shape}.")
        if require_2d and arr2.ndim != 2:
            raise ValueError(f"{name} must be 2D; got shape {arr2.shape}.")

        if not np.all(np.isfinite(arr2)):
            raise ValueError(
                f"{name} contains NaN or inf under validate_inputs={self.validate_inputs!r}."
            )

        return arr2

    def _score_row(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> dict[str, float]:
        y_true_f = np.asarray(y_true, dtype=float)
        y_pred_f = np.asarray(y_pred, dtype=float)

        out: dict[str, float] = {}
        out["CWSL"] = float(
            cwsl(
                y_true=y_true_f,
                y_pred=y_pred_f,
                cu=self.cu,
                co=self.co,
                sample_weight=sample_weight,
            )
        )
        out["RMSE"] = float(rmse(y_true=y_true_f, y_pred=y_pred_f))
        out["wMAPE"] = float(wmape(y_true=y_true_f, y_pred=y_pred_f))
        return out

    def _select_column_name(self) -> str:
        # Metric names are lower-case; result columns use conventional casing.
        if self.metric == "cwsl":
            return "CWSL"
        if self.metric == "rmse":
            return "RMSE"
        return "wMAPE"

    def _empty_results(self) -> pd.DataFrame:
        """
        Create an empty results table with stable schema and index name.

        This avoids pandas-stubs/pyright false positives around `DataFrame(columns=[...])`
        while keeping runtime behavior identical.
        """
        return pd.DataFrame(
            {
                "CWSL": pd.Series(dtype=float),
                "RMSE": pd.Series(dtype=float),
                "wMAPE": pd.Series(dtype=float),
            },
            index=pd.Index([], name="model"),
        )

    def _argmin_with_nan_safe(self, series: pd.Series) -> str:
        # pandas typing is broad; force Series-shaped output for pyright/pandas-stubs.
        numeric = pd.to_numeric(series, errors="coerce")
        if isinstance(numeric, pd.Series):
            s = numeric
        else:
            # Fallback: preserve the original index if available; otherwise default.
            idx = getattr(series, "index", None)
            s = pd.Series(numeric, index=idx)

        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            raise RuntimeError("No valid candidate scores were produced; all models failed.")
        return str(s.idxmin())

    def _model_complexity_score(self, model: Any) -> float:
        """
        Best-effort heuristic for "simpler" tie-breaking.

        Lower is simpler. This is intentionally lightweight and stable (not perfect).
        """
        score = 0.0

        get_params = getattr(model, "get_params", None)
        if callable(get_params):
            try:
                params = get_params(deep=True)

                # Pyright-friendly: only call len() when the returned object is Sized.
                # (sklearn typically returns dict[str, Any].)
                if isinstance(params, Sized):
                    score += float(len(cast(Sized, params)))
                else:
                    score += 1000.0
            except Exception:
                score += 1000.0

        for attr in ("n_estimators", "iterations", "num_iterations", "max_iter", "n_iter_"):
            if hasattr(model, attr):
                try:
                    val = getattr(model, attr)
                    if val is not None:
                        score += float(val)
                except Exception:
                    score += 50.0

        for attr in ("max_depth", "depth", "num_leaves"):
            if hasattr(model, attr):
                try:
                    val = getattr(model, attr)
                    if val is not None:
                        v = float(val)
                        if v > 0:
                            score += v
                except Exception:
                    score += 10.0

        return score

    def _select_best_name(self, results: pd.DataFrame) -> str:
        """
        Select the best model name using metric + tie-breaking rules.

        Ties are defined as all candidates with score <= best_score + tie_tol.
        """
        select_col = self._select_column_name()

        selected = results[select_col]
        if not isinstance(selected, pd.Series):
            raise RuntimeError(f"Expected Series for selection column {select_col!r}.")

        # Reuse existing NaN/inf-safe conversion logic, but keep the cleaned Series for ties.
        numeric = pd.to_numeric(selected, errors="coerce")
        if isinstance(numeric, pd.Series):
            s = numeric
        else:
            idx = getattr(selected, "index", None)
            s = pd.Series(numeric, index=idx)

        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            raise RuntimeError("No valid candidate scores were produced; all models failed.")

        best_score = float(s.min())
        tol = float(self.tie_tol)

        # Use `.loc[...]` to force Series output under pandas-stubs/pyright (avoid ndarray typing).
        tied = s.loc[s <= (best_score + tol)]
        tied_names = [str(i) for i in tied.index]

        if len(tied_names) == 1:
            return tied_names[0]

        if self.tie_breaker == "name":
            return sorted(tied_names)[0]

        if self.tie_breaker == "simpler":
            scored: list[tuple[float, str]] = []
            for name in tied_names:
                base = self.models.get(name)
                if base is None:
                    continue
                scored.append((self._model_complexity_score(base), name))

            if scored:
                scored.sort(key=lambda t: (t[0], t[1]))
                return scored[0][1]

            # Defensive fallback
            return sorted(tied_names)[0]

        # "metric" (default): deterministic by Series index order
        return str(s.idxmin())

    def _cv_evaluate_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Evaluate candidate models via simple K-fold CV and return mean scores.

        Notes on budgets
        ----------------
        Budgets are enforced as gating/marking checks and cannot forcibly interrupt a fit
        mid-call. If a model exceeds ``per_model_time_budget_s`` across folds, it is marked
        as timed out and skipped (or raises under ``error_policy="raise"``).
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if self.validate_inputs != "off":
            X_arr = self._ensure_numeric_finite(name="X", arr=X_arr, require_2d=True)
            y_arr = self._ensure_numeric_finite(name="y", arr=y_arr, require_1d=True)
            y_arr = np.asarray(y_arr, dtype=float)

        n_samples = X_arr.shape[0]
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have the same number of rows; got {n_samples} and {y_arr.shape[0]}."
            )

        k = int(self.cv)
        if k < 2 or k > n_samples:
            raise ValueError(f"Invalid number of folds cv={k} for n_samples={n_samples}.")

        sw_arr: np.ndarray | None = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
            if self.validate_inputs != "off":
                sw_arr = self._ensure_numeric_finite(
                    name="sample_weight", arr=sw_arr, require_1d=True
                )
                sw_arr = np.asarray(sw_arr, dtype=float)

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

        run_start = time.perf_counter()

        for model_name, base_model in self.models.items():
            self.evaluated_names_.append(str(model_name))

            if (
                self.time_budget_s is not None
                and (time.perf_counter() - run_start) > self.time_budget_s
            ):
                self.stopped_early_ = True
                self.stop_reason_ = f"global time budget exceeded ({self.time_budget_s}s)"
                self._handle_failure(
                    model_name=str(model_name),
                    reason=f"global time budget exceeded ({self.time_budget_s}s); stopping evaluation",
                )
                break  # remaining models not evaluated

            model_start = time.perf_counter()

            cwsl_scores: list[float] = []
            rmse_scores: list[float] = []
            wmape_scores: list[float] = []

            model_failed = False

            for fold_i, (train_idx, val_idx) in enumerate(folds):
                if self.per_model_time_budget_s is not None:
                    elapsed_model = time.perf_counter() - model_start
                    if elapsed_model > self.per_model_time_budget_s:
                        self._handle_failure(
                            model_name=str(model_name),
                            reason=(
                                f"per-model time budget exceeded ({self.per_model_time_budget_s}s) "
                                f"during CV (fold {fold_i + 1}/{k})"
                            ),
                        )
                        model_failed = True
                        break

                X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
                sw_va = sw_arr[val_idx] if sw_arr is not None else None

                try:
                    model = _clone_model(base_model)
                    model.fit(X_tr, y_tr)
                    y_pred = np.asarray(model.predict(X_va), dtype=float)

                    scores = self._score_row(y_true=y_va, y_pred=y_pred, sample_weight=sw_va)
                    cwsl_scores.append(scores["CWSL"])
                    rmse_scores.append(scores["RMSE"])
                    wmape_scores.append(scores["wMAPE"])
                except Exception as e:
                    self._handle_failure(
                        model_name=str(model_name),
                        reason=f"{type(e).__name__}: {e}",
                    )
                    model_failed = True
                    break

            if model_failed:
                continue

            rows.append(
                {
                    "model": str(model_name),
                    "CWSL": float(np.mean(cwsl_scores)),
                    "RMSE": float(np.mean(rmse_scores)),
                    "wMAPE": float(np.mean(wmape_scores)),
                }
            )

        if not rows:
            return self._empty_results()

        return pd.DataFrame.from_records(rows).set_index("model")

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
        Fit candidate models and select the best one by minimizing the chosen metric.
        """
        # Reset run-audit state for this fit call
        self.candidate_names_ = list(self.models.keys())
        self.evaluated_names_ = []
        self.stopped_early_ = False
        self.stop_reason_ = None

        refit_flag = self.refit_on_full if refit_on_full is None else bool(refit_on_full)

        if self.selection_mode == "holdout":
            X_tr = np.asarray(X_train)
            y_tr = np.asarray(y_train, dtype=float)
            X_va = np.asarray(X_val)
            y_va = np.asarray(y_val, dtype=float)

            if X_tr.ndim == 1:
                X_tr = X_tr.reshape(-1, 1)
            if X_va.ndim == 1:
                X_va = X_va.reshape(-1, 1)

            if self.validate_inputs != "off":
                X_tr = self._ensure_numeric_finite(name="X_train", arr=X_tr, require_2d=True)
                y_tr = self._ensure_numeric_finite(name="y_train", arr=y_tr, require_1d=True)
                X_va = self._ensure_numeric_finite(name="X_val", arr=X_va, require_2d=True)
                y_va = self._ensure_numeric_finite(name="y_val", arr=y_va, require_1d=True)
                y_tr = np.asarray(y_tr, dtype=float)
                y_va = np.asarray(y_va, dtype=float)

            if y_tr.shape[0] != X_tr.shape[0]:
                raise ValueError(
                    f"X_train and y_train must have matching rows; got {X_tr.shape[0]} and {y_tr.shape[0]}."
                )
            if y_va.shape[0] != X_va.shape[0]:
                raise ValueError(
                    f"X_val and y_val must have matching rows; got {X_va.shape[0]} and {y_va.shape[0]}."
                )

            sw_val: np.ndarray | None = None
            if sample_weight_val is not None:
                sw_val = np.asarray(sample_weight_val, dtype=float)
                if self.validate_inputs != "off":
                    sw_val = self._ensure_numeric_finite(
                        name="sample_weight_val", arr=sw_val, require_1d=True
                    )
                    sw_val = np.asarray(sw_val, dtype=float)

                if sw_val.shape[0] != y_va.shape[0]:
                    raise ValueError(
                        f"sample_weight_val must have length {y_va.shape[0]}; got {sw_val.shape[0]}."
                    )

            rows: list[dict[str, float | str]] = []

            run_start = time.perf_counter()

            for model_name, base_model in self.models.items():
                self.evaluated_names_.append(str(model_name))

                if (
                    self.time_budget_s is not None
                    and (time.perf_counter() - run_start) > self.time_budget_s
                ):
                    self.stopped_early_ = True
                    self.stop_reason_ = f"global time budget exceeded ({self.time_budget_s}s)"
                    self._handle_failure(
                        model_name=str(model_name),
                        reason=f"global time budget exceeded ({self.time_budget_s}s); stopping evaluation",
                    )
                    break  # remaining models not evaluated

                model_start = time.perf_counter()

                try:
                    model = _clone_model(base_model)
                    model.fit(X_tr, y_tr)

                    if self.per_model_time_budget_s is not None:
                        elapsed = time.perf_counter() - model_start
                        if elapsed > self.per_model_time_budget_s:
                            self._handle_failure(
                                model_name=str(model_name),
                                reason=(
                                    f"per-model time budget exceeded ({self.per_model_time_budget_s}s) "
                                    f"after fit"
                                ),
                            )
                            continue

                    y_pred = np.asarray(model.predict(X_va), dtype=float)

                    if self.per_model_time_budget_s is not None:
                        elapsed = time.perf_counter() - model_start
                        if elapsed > self.per_model_time_budget_s:
                            self._handle_failure(
                                model_name=str(model_name),
                                reason=(
                                    f"per-model time budget exceeded ({self.per_model_time_budget_s}s) "
                                    f"after predict"
                                ),
                            )
                            continue

                    scores = self._score_row(y_true=y_va, y_pred=y_pred, sample_weight=sw_val)
                    rows.append(
                        {
                            "model": str(model_name),
                            "CWSL": float(scores["CWSL"]),
                            "RMSE": float(scores["RMSE"]),
                            "wMAPE": float(scores["wMAPE"]),
                        }
                    )
                except Exception as e:
                    self._handle_failure(
                        model_name=str(model_name),
                        reason=f"{type(e).__name__}: {e}",
                    )
                    continue

            results = (
                pd.DataFrame.from_records(rows).set_index("model")
                if rows
                else self._empty_results()
            )
            self.results_ = results

            best_name = self._select_best_name(results)
            self.best_name_ = best_name

            row = results.loc[best_name]
            self.validation_cwsl_ = float(row["CWSL"]) if "CWSL" in row else None
            self.validation_rmse_ = float(row["RMSE"]) if "RMSE" in row else None
            self.validation_wmape_ = float(row["wMAPE"]) if "wMAPE" in row else None

            try:
                # Refit or keep: we always refit a fresh clone here (deterministic, cloneable).
                base_winner = self.models[best_name]
                best_model_fitted = _clone_model(base_winner)

                if refit_flag:
                    X_full = np.concatenate([X_tr, X_va], axis=0)
                    y_full = np.concatenate([y_tr, y_va], axis=0)
                    best_model_fitted.fit(X_full, y_full)
                else:
                    best_model_fitted.fit(X_tr, y_tr)
            except Exception as e:
                raise RuntimeError(
                    f"Selected model {best_name!r} but failed to fit winning model for inference: "
                    f"{type(e).__name__}: {e}"
                ) from e

            self.best_model_ = best_model_fitted
            return self

        # CV mode (selection uses metric column on mean scores)
        results = self._cv_evaluate_models(
            X=np.asarray(X_train),
            y=np.asarray(y_train, dtype=float),
            sample_weight=sample_weight_train,
        )
        self.results_ = results

        best_name = self._select_best_name(results)
        self.best_name_ = best_name

        row = results.loc[best_name]
        self.validation_cwsl_ = float(row["CWSL"]) if "CWSL" in row else None
        self.validation_rmse_ = float(row["RMSE"]) if "RMSE" in row else None
        self.validation_wmape_ = float(row["wMAPE"]) if "wMAPE" in row else None

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
            f"metric={self.metric!r}, tie_tol={self.tie_tol}, tie_breaker={self.tie_breaker!r}, "
            f"validate_inputs={self.validate_inputs!r}, error_policy={self.error_policy!r}, "
            f"cu={self.cu}, co={self.co}, tau={self.tau}, "
            f"refit_on_full={self.refit_on_full}, "
            f"selection_mode={self.selection_mode!r}, "
            f"cv={self.cv}, random_state={self.random_state!r}, "
            f"best_name_={best!r})"
        )

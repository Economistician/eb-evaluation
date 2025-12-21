from __future__ import annotations

"""
Feature engineering for panel time series.

This module provides a lightweight, frequency-agnostic feature engineering utility for
panel time-series data (entity × timestamp). It is intentionally *stateless*: each call
to :meth:`FeatureEngineer.transform` constructs features from the provided input DataFrame
and configuration.

The feature set is designed for classical supervised learning pipelines (e.g., tree models,
linear models, shallow neural nets) that expect a fixed-width design matrix ``X`` and target
vector ``y``.

Features
--------
Given an entity identifier column ``e`` and a target series ``y_t`` (per entity):

1) Lag features:
$$
    \mathrm{lag}_k(t) = y_{t-k}
$$

2) Rolling window statistics over the last ``w`` observations:
$$
    \mathrm{roll\_mean}_w(t) = \frac{1}{w}\sum_{j=1}^{w} y_{t-j}
$$

(Other statistics supported: std, min, max, sum, median.)

3) Calendar features derived from timestamp:
hour, day-of-week, day-of-month, month, and weekend indicator.

4) Optional cyclical encodings for periodic calendar features:
$$
    \sin\left(2\pi \frac{\mathrm{hour}}{24}\right), \quad
    \cos\left(2\pi \frac{\mathrm{hour}}{24}\right)
$$

and similarly for day-of-week with period 7.

5) Optional passthrough features:
numeric regressors and static metadata columns.

All lag and rolling windows are expressed in *index steps* at the input frequency. This keeps
the transformer frequency-agnostic (5-min, 30-min, hourly, daily, etc.).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class FeatureConfig:
    """
    Configuration for panel time-series feature engineering.

    Notes
    -----
    - All lag steps and rolling window lengths are expressed in **index steps** (rows) at the
      input frequency, not in wall-clock units.
    - Lags and rolling windows are computed **within each entity**.
    - If ``dropna=True``, rows that lack full lag/rolling history are dropped after feature
      construction.

    Attributes
    ----------
    lag_steps : Sequence[int] | None
        Positive lag offsets (in steps) applied to the target. For each ``k`` in ``lag_steps``,
        the feature ``lag_k`` is added.
    rolling_windows : Sequence[int] | None
        Positive rolling window lengths (in steps) applied to the target. For each ``w`` in
        ``rolling_windows`` and each stat in ``rolling_stats``, the feature ``roll_{w}_{stat}``
        is added.
    rolling_stats : Sequence[str]
        Rolling statistics to compute. Allowed values are:
        ``{"mean", "std", "min", "max", "sum", "median"}``.
    calendar_features : Sequence[str]
        Calendar features derived from ``timestamp_col``. Allowed values:
        ``{"hour", "dow", "dom", "month", "is_weekend"}``.
    use_cyclical_time : bool
        If True, add sine/cosine encodings for hour and day-of-week when those base columns
        are present.
    regressor_cols : Sequence[str] | None
        Numeric external regressors to pass through. If None, numeric columns are auto-detected
        excluding entity/timestamp/target and ``static_cols``.
    static_cols : Sequence[str] | None
        Entity-level metadata columns already present on the input DataFrame. These are passed
        through directly as features.
    dropna : bool
        If True, drop rows with NaNs in any engineered feature columns (typically caused by lags
        and rolling windows).

    """

    lag_steps: Optional[Sequence[int]] = field(default_factory=lambda: [1, 2, 24])

    rolling_windows: Optional[Sequence[int]] = field(default_factory=lambda: [3, 24])
    rolling_stats: Sequence[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "sum"])

    calendar_features: Sequence[str] = field(default_factory=lambda: ["hour", "dow", "month", "is_weekend"])
    use_cyclical_time: bool = True

    regressor_cols: Optional[Sequence[str]] = None
    static_cols: Optional[Sequence[str]] = None

    dropna: bool = True


class FeatureEngineer:
    """
    Transform panel time-series data into a model-ready (X, y, feature_names) triple.

    The input is expected to be a long-form DataFrame with at least:

    - ``entity_col``: series identifier (e.g., store_id, sku_id)
    - ``timestamp_col``: datetime-like timestamp (ideally ``datetime64[ns]``)
    - ``target_col``: numeric target to be predicted

    The transformer computes features strictly within entity, requires timestamps to be
    strictly increasing within each entity, and can optionally drop rows lacking sufficient
    history.

    Notes
    -----
    - This class does not store fitted state. Feature generation is deterministic given the
      inputs.
    - Non-numeric feature columns (e.g., static categorical metadata) are encoded using pandas
      categorical codes, producing stable integer codes for the values present in the provided
      DataFrame.

    """

    def __init__(
        self,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        target_col: str = "target",
    ) -> None:
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col

    def transform(
        self,
        df: pd.DataFrame,
        config: FeatureConfig,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        r"""
        Transform a panel DataFrame into (X, y, feature_names).

        Parameters
        ----------
        df : pandas.DataFrame
            Input panel data containing at least ``entity_col``, ``timestamp_col``, and
            ``target_col``.
        config : FeatureConfig
            Feature engineering configuration describing which features to build.

        Returns
        -------
        X : numpy.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : numpy.ndarray, shape (n_samples,)
            Target vector aligned with ``X``.
        feature_names : list[str]
            Names corresponding to the columns of ``X``, in order.

        Raises
        ------
        KeyError
            If required columns or configured passthrough columns are missing.
        ValueError
            If timestamps are not strictly increasing within each entity, if the target contains
            negative values, or if non-finite values are present in the resulting ``X``/``y``.

        """
        self._validate_input(df)

        # Work on a copy, sorted by entity / timestamp
        df = df.copy()
        df = df.sort_values([self.entity_col, self.timestamp_col])
        self._validate_monotonic(df)

        # ------------------------------------------------------------------
        # Identify passthrough columns
        # ------------------------------------------------------------------
        static_cols = list(config.static_cols or [])
        for col in static_cols:
            if col not in df.columns:
                raise KeyError(f"Static column {col!r} not found in DataFrame.")

        if config.regressor_cols is not None:
            regressor_cols = list(config.regressor_cols)
        else:
            exclude = {self.entity_col, self.timestamp_col, self.target_col, *static_cols}
            numeric_cols = df.select_dtypes(include=["number"]).columns
            regressor_cols = [c for c in numeric_cols if c not in exclude]

        for col in regressor_cols:
            if col not in df.columns:
                raise KeyError(f"Regressor column {col!r} not found in DataFrame.")

        feature_cols: List[str] = []

        # ------------------------------------------------------------------
        # 1) Lag features on target
        # ------------------------------------------------------------------
        if config.lag_steps:
            for k in config.lag_steps:
                if k <= 0:
                    raise ValueError(f"Lag steps must be positive; got {k}.")
                col_name = f"lag_{k}"
                df[col_name] = df.groupby(self.entity_col)[self.target_col].shift(k)
                feature_cols.append(col_name)

        # ------------------------------------------------------------------
        # 2) Rolling window statistics on target
        # ------------------------------------------------------------------
        allowed_stats = {"mean", "std", "min", "max", "sum", "median"}
        for s in config.rolling_stats:
            if s not in allowed_stats:
                raise ValueError(
                    f"Unsupported rolling stat {s!r}. Allowed: {sorted(allowed_stats)}"
                )

        if config.rolling_windows:
            for w in config.rolling_windows:
                if w <= 0:
                    raise ValueError(f"Rolling window must be positive; got {w}.")
                group_series = df.groupby(self.entity_col)[self.target_col]
                roll = group_series.rolling(window=w, min_periods=w)

                for stat in config.rolling_stats:
                    col_name = f"roll_{w}_{stat}"
                    if stat == "mean":
                        values = roll.mean()
                    elif stat == "std":
                        values = roll.std()
                    elif stat == "min":
                        values = roll.min()
                    elif stat == "max":
                        values = roll.max()
                    elif stat == "sum":
                        values = roll.sum()
                    elif stat == "median":
                        values = roll.median()
                    else:  # pragma: no cover
                        raise RuntimeError(f"Unexpected rolling stat {stat!r}.")

                    df[col_name] = values.reset_index(level=0, drop=True)
                    feature_cols.append(col_name)

        # ------------------------------------------------------------------
        # 3) Calendar / time features
        # ------------------------------------------------------------------
        ts = pd.to_datetime(df[self.timestamp_col])

        calendar_cols: List[str] = []
        for name in config.calendar_features:
            if name == "hour":
                col = "hour"
                df[col] = ts.dt.hour.astype("int16")
                calendar_cols.append(col)
            elif name == "dow":
                col = "dayofweek"
                df[col] = ts.dt.dayofweek.astype("int16")
                calendar_cols.append(col)
            elif name == "dom":
                col = "dayofmonth"
                df[col] = ts.dt.day.astype("int16")
                calendar_cols.append(col)
            elif name == "month":
                col = "month"
                df[col] = ts.dt.month.astype("int16")
                calendar_cols.append(col)
            elif name == "is_weekend":
                col = "is_weekend"
                df[col] = ts.dt.dayofweek.isin([5, 6]).astype("int8")
                calendar_cols.append(col)
            else:
                raise ValueError(
                    f"Unsupported calendar feature {name!r}. "
                    "Allowed: 'hour', 'dow', 'dom', 'month', 'is_weekend'."
                )

        feature_cols.extend(calendar_cols)

        # Optional cyclical encodings for hour & day-of-week
        if config.use_cyclical_time:
            if "hour" in calendar_cols:
                hour = df["hour"].astype(float)
                df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
                df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
                feature_cols.extend(["hour_sin", "hour_cos"])

            if "dayofweek" in calendar_cols:
                dow = df["dayofweek"].astype(float)
                df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
                df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
                feature_cols.extend(["dow_sin", "dow_cos"])

        # ------------------------------------------------------------------
        # 4) Static metadata & external regressors
        # ------------------------------------------------------------------
        feature_cols.extend(static_cols)
        feature_cols.extend(regressor_cols)

        # ------------------------------------------------------------------
        # Final cleaning & extraction
        # ------------------------------------------------------------------
        df = df[~df[self.target_col].isna()]

        # For demand-like series, enforce non-negative target.
        if (df[self.target_col] < 0).any():
            raise ValueError("Negative values found in target column; expected >= 0.")

        if config.dropna:
            df = df.dropna(subset=feature_cols)

        feature_frame = df[feature_cols].copy()

        # Encode non-numeric features as categorical codes.
        for col in feature_frame.columns:
            if not is_numeric_dtype(feature_frame[col]):
                feature_frame[col] = feature_frame[col].astype("category").cat.codes.astype("float64")

        X_values = feature_frame.to_numpy(dtype=float)
        y_values = df[self.target_col].to_numpy(dtype=float)

        if not np.isfinite(X_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in feature matrix X.")
        if not np.isfinite(y_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in target vector y.")

        return X_values, y_values, feature_cols

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate required columns exist on the input DataFrame."""
        required = {self.entity_col, self.timestamp_col, self.target_col}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"Input DataFrame missing required columns: {sorted(missing)}.")

    def _validate_monotonic(self, df: pd.DataFrame) -> None:
        """
        Ensure timestamps are strictly increasing within each entity.

        Raises
        ------
        ValueError
            If any entity has non-increasing timestamps.

        """
        grp = df.groupby(self.entity_col)[self.timestamp_col]
        diffs = grp.diff()
        if (diffs <= pd.Timedelta(0)).dropna().any():
            raise ValueError("Timestamps must be strictly increasing within each entity.")
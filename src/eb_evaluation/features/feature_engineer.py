from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class FeatureConfig:
    """
    Configuration for time-series feature engineering.

    All lags / windows are expressed in *index steps* of the input frequency.
    This keeps the transformer frequency-agnostic (5-min, 30-min, hourly, daily, etc.).
    """

    # Lag features on the target
    lag_steps: Optional[Sequence[int]] = field(default_factory=lambda: [1, 2, 24])

    # Rolling windows (in steps) on the target
    rolling_windows: Optional[Sequence[int]] = field(default_factory=lambda: [3, 24])
    rolling_stats: Sequence[str] = field(
        default_factory=lambda: ["mean", "std", "min", "max", "sum"]
    )

    # Calendar / time features derived from timestamp
    # Allowed values: "hour", "dow", "dom", "month", "is_weekend"
    calendar_features: Sequence[str] = field(
        default_factory=lambda: ["hour", "dow", "month", "is_weekend"]
    )
    use_cyclical_time: bool = True  # sin / cos encodings for hour / dow when present

    # Optional passthrough columns
    regressor_cols: Optional[Sequence[str]] = None  # numeric external regressors
    static_cols: Optional[Sequence[str]] = None  # entity-level metadata already on df

    # Behaviour on NaNs created by lags/rolls
    dropna: bool = True


class FeatureEngineer:
    """
    Panel time-series → model-ready (X, y, feature_names) transformer.

    This class is intentionally stateless: each call to `transform` uses the
    provided configuration and dataframe to build features from scratch.

    Expected input format
    ---------------------
    A long-form pandas DataFrame with at least:

        - entity_col   : series / entity identifier (e.g., store_id, sku_id)
        - timestamp_col: datetime64[ns] timestamp
        - target_col   : numeric target to forecast

    Optional additional columns may include:

        - external regressors (weather, promos, etc.)
        - static metadata (region, store_type, etc.) – one value per entity
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(
        self,
        df: pd.DataFrame,
        config: FeatureConfig,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transform a panel time-series DataFrame into (X, y, feature_names).

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing at least entity, timestamp, and target columns.

        config : FeatureConfig
            Configuration describing which features to build.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        feature_names : list of str
            Names corresponding to columns in X, in order.
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
            # Auto-detect numeric regressors: numeric columns that are not
            # entity/timestamp/target or declared static columns.
            exclude = {
                self.entity_col,
                self.timestamp_col,
                self.target_col,
                *static_cols,
            }
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
                df[col_name] = (
                    df.groupby(self.entity_col)[self.target_col].shift(k)
                )
                feature_cols.append(col_name)

        # ------------------------------------------------------------------
        # 2) Rolling window statistics on target
        # ------------------------------------------------------------------
        allowed_stats = {"mean", "std", "min", "max", "sum", "median"}
        for s in config.rolling_stats:
            if s not in allowed_stats:
                raise ValueError(
                    f"Unsupported rolling stat {s!r}. "
                    f"Allowed: {sorted(allowed_stats)}"
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
                    else:  # pragma: no cover - guarded by allowed_stats
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
            if "hour" in [c for c in calendar_cols]:
                hour = df["hour"].astype(float)
                df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
                df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
                feature_cols.extend(["hour_sin", "hour_cos"])

            if "dayofweek" in [c for c in calendar_cols]:
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
        # Remove rows with invalid target
        df = df[~df[self.target_col].isna()]
        # (Optional) enforce non-negative target for demand-like series
        if (df[self.target_col] < 0).any():
            raise ValueError("Negative values found in target column; expected >= 0.")

        if config.dropna:
            df = df.dropna(subset=feature_cols)

        # Build feature frame and encode any non-numeric columns
        feature_frame = df[feature_cols].copy()

        for col in feature_frame.columns:
            if not is_numeric_dtype(feature_frame[col]):
                feature_frame[col] = (
                    feature_frame[col]
                    .astype("category")
                    .cat.codes.astype("float64")
                )

        X_values = feature_frame.to_numpy(dtype=float)
        y_values = df[self.target_col].to_numpy(dtype=float)

        if not np.isfinite(X_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in feature matrix X.")
        if not np.isfinite(y_values).all():
            raise ValueError("Non-finite values (NaN/inf) found in target vector y.")

        return X_values, y_values, feature_cols

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {self.entity_col, self.timestamp_col, self.target_col}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(
                f"Input DataFrame missing required columns: {sorted(missing)}."
            )

    def _validate_monotonic(self, df: pd.DataFrame) -> None:
        """
        Ensure timestamps are strictly increasing within each entity_id.
        """
        grp = df.groupby(self.entity_col)[self.timestamp_col]
        # diff > 0 for all but first row of each group
        diffs = grp.diff()
        if (diffs <= pd.Timedelta(0)).dropna().any():
            raise ValueError(
                "Timestamps must be strictly increasing within each entity_id."
            )
"""
Internal utilities for the Readiness Adjustment Layer (RAL).

These helpers are *not* part of the public API and are used only to keep
readiness_adjustment.py clean and focused on the main algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Array validation utilities
# ---------------------------------------------------------------------------

def validate_numeric_array(arr, name: str = "array") -> np.ndarray:
    """
    Ensure an input can be safely interpreted as a 1D or 2D float array.

    Parameters
    ----------
    arr : array-like
        Input array (list, numpy array, pandas Series).

    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        A float64 array.

    Raises
    ------
    ValueError
        If array contains NaN or non-finite values.
    """
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    elif isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 0:
        raise ValueError(f"{name} must not be a scalar.")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return arr


# ---------------------------------------------------------------------------
# Safe statistical helpers
# ---------------------------------------------------------------------------

def safe_mean(values: np.ndarray) -> float:
    """
    A safe mean that returns 0.0 for empty inputs.

    Useful when computing uplift averages for groups that may have no valid rows
    after filtering.

    Parameters
    ----------
    values : np.ndarray

    Returns
    -------
    float
    """
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Groupby helpers
# ---------------------------------------------------------------------------

def groupby_apply_values(
    df: pd.DataFrame,
    group_cols,
    value_col: str,
    func,
) -> pd.DataFrame:
    """
    Apply a function to a numeric column grouped by entity/cluster columns.

    Returns a DataFrame with:

        group_cols + [value_col]

    where value_col contains the result of the group-level computation.

    Parameters
    ----------
    df : DataFrame
        Input panel data.

    group_cols : str or list[str]
        Columns to group by.

    value_col : str
        Column for which the statistic is computed.

    func : Callable
        Function taking a 1D numeric array and returning a scalar.

    Returns
    -------
    DataFrame
        A tidy DataFrame: group → aggregated value.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Group and compute
    grouped = (
        df.groupby(group_cols)[value_col]
        .apply(lambda s: func(validate_numeric_array(s.to_numpy())))
        .reset_index()
        .rename(columns={value_col: f"{value_col}_agg"})
    )

    return grouped
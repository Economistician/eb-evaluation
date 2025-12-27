"""
Internal utilities for the Readiness Adjustment Layer.

This module contains small, focused helpers used by the Readiness Adjustment Layer (RAL)
implementation. These utilities are **not** part of the public API and may change without
notice.

The intent is to keep the main algorithm (and its public surface area) in
`eb_evaluation.adjustment.ral` clean and readable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | Sequence[float] | pd.Series | pd.DataFrame


# ---------------------------------------------------------------------------
# Array validation utilities
# ---------------------------------------------------------------------------
def validate_numeric_array(arr: ArrayLike, name: str = "array") -> np.ndarray:
    """Validate and coerce an input to a finite float NumPy array.

    This helper is intentionally strict: it ensures the array is not scalar and that
    all values are finite. This is appropriate for evaluation utilities where silent
    propagation of NaNs can lead to misleading metric comparisons.

    Parameters
    ----------
    arr
        Input array-like object. Common inputs include lists, NumPy arrays, pandas Series,
        or a single-column DataFrame.
    name
        Name used in error messages to make debugging easier.

    Returns
    -------
    numpy.ndarray
        A NumPy array of dtype ``float64``. The returned array may be 1D or 2D depending on
        the input.

    Raises
    ------
    ValueError
        If ``arr`` is scalar (0-dimensional) or contains NaN/infinite values.

    Notes
    -----
    - If you pass a DataFrame, its underlying NumPy representation is used (i.e., you are
      responsible for selecting appropriate columns before calling this helper).
    - This helper does *not* drop missing values. If you need filtering behavior, perform
      it upstream and call this only once the array should be clean.
    """
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.to_numpy()

    out = np.asarray(arr, dtype=float)

    if out.ndim == 0:
        raise ValueError(f"{name} must be an array-like (not a scalar).")

    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return out


# ---------------------------------------------------------------------------
# Safe statistical helpers
# ---------------------------------------------------------------------------
def safe_mean(values: np.ndarray) -> float:
    """Compute a mean with a defined result for empty inputs.

    Parameters
    ----------
    values
        Numeric array. Typically a 1D array of values after filtering.

    Returns
    -------
    float
        The arithmetic mean of ``values``. If ``values`` is empty (``values.size == 0``),
        returns ``0.0``.

    Notes
    -----
    This is mainly used in group-level computations where a group may end up empty after
    filtering invalid rows. Returning ``0.0`` is a pragmatic default for diagnostics; it
    should not be used as a substitute for input validation in the primary metric pathway.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Groupby helpers
# ---------------------------------------------------------------------------
def groupby_apply_values(
    df: pd.DataFrame,
    group_cols: str | Sequence[str],
    value_col: str,
    func: Callable[[np.ndarray], float],
) -> pd.DataFrame:
    """Apply a numeric reducer to a column, grouped by one or more keys.

    This helper groups ``df`` by ``group_cols`` and applies ``func`` to the values of
    ``value_col`` for each group. The function is called with a validated, finite float
    array.

    Parameters
    ----------
    df
        Input DataFrame containing grouping keys and the numeric value column.
    group_cols
        Column name or sequence of column names to group by.
    value_col
        Name of the column whose values are passed to ``func``.
    func
        Reducer function taking a 1D NumPy array and returning a scalar (float).

    Returns
    -------
    pandas.DataFrame
        A tidy DataFrame with columns:

        - ``group_cols`` (one column per grouping key)
        - ``f"{value_col}_agg"`` (the aggregated scalar result)

    Raises
    ------
    KeyError
        If ``value_col`` or any ``group_cols`` are missing from ``df``.
    ValueError
        If group values contain NaN/infinite values (via :func:`validate_numeric_array`).

    Examples
    --------
    >>> out = groupby_apply_values(df, ["cluster", "daypart"], "uplift", np.mean)
    >>> out.columns
    Index(['cluster', 'daypart', 'uplift_agg'], dtype='object')
    """
    if isinstance(group_cols, str):
        group_cols_seq: list[str] = [group_cols]
    else:
        group_cols_seq = list(group_cols)

    missing = [c for c in [*group_cols_seq, value_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    agg_col = f"{value_col}_agg"

    grouped = (
        df.groupby(group_cols_seq, dropna=False)[value_col]
        .apply(lambda s: func(validate_numeric_array(s.to_numpy(), name=value_col)))
        .reset_index()
        .rename(columns={value_col: agg_col})
    )

    return grouped

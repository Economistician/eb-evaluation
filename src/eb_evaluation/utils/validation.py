from __future__ import annotations

r"""
Lightweight DataFrame validation utilities.

This module provides small, explicit validation helpers for pandas DataFrames
used throughout the Electric Barometer evaluation and model-selection stack.

The intent is to:
- fail fast with clear error messages,
- keep validation logic centralized and reusable,
- distinguish *data validation errors* from other ValueErrors.

These helpers are intentionally minimal and do not attempt schema inference
or coercion; they only assert required structural properties.
"""

from typing import Sequence

import pandas as pd


class DataFrameValidationError(ValueError):
    r"""
    Error raised when a pandas DataFrame fails a validation check.

    This is a thin subclass of :class:`ValueError` that allows callers to
    explicitly catch DataFrame-related validation issues and distinguish
    them from other value errors (e.g., numerical domain errors).
    """


def ensure_columns_present(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    context: str | None = None,
) -> None:
    """
    Ensure that all required columns are present in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    required : sequence of str
        Column names that must be present in ``df``.
    context : str | None, optional
        Optional context string (e.g., function or module name) included
        in the error message to aid debugging.

    Raises
    ------
    DataFrameValidationError
        If one or more required columns are missing.

    Notes
    -----
    This function performs a *presence-only* check. It does not validate
    column dtypes or contents.
    """
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return

    prefix = f"[{context}] " if context is not None else ""
    raise DataFrameValidationError(
        f"{prefix}DataFrame is missing required columns: {missing}"
    )


def ensure_non_empty(
    df: pd.DataFrame,
    *,
    context: str | None = None,
) -> None:
    """
    Ensure that a DataFrame is not empty.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    context : str | None, optional
        Optional context string (e.g., function or module name) included
        in the error message to aid debugging.

    Raises
    ------
    DataFrameValidationError
        If the DataFrame has zero rows.

    Notes
    -----
    This check is commonly used after filtering or grouping operations
    to ensure downstream computations have at least one observation.
    """
    if len(df) == 0:
        prefix = f"[{context}] " if context is not None else ""
        raise DataFrameValidationError(f"{prefix}DataFrame is empty.")
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


class DataFrameValidationError(ValueError):
    """
    Error raised when an input pandas.DataFrame fails a validation check.

    This is a thin wrapper around ValueError so callers can catch a more
    specific exception type if they want to distinguish validation issues
    from other ValueErrors.
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
        The DataFrame to validate.

    required : sequence of str
        Column names that must be present in ``df``.

    context : str, optional
        Optional context string to include in the error message
        (e.g. the name of the calling function).

    Raises
    ------
    DataFrameValidationError
        If one or more required columns are missing.
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
        The DataFrame to validate.

    context : str, optional
        Optional context string to include in the error message
        (e.g. the name of the calling function).

    Raises
    ------
    DataFrameValidationError
        If the DataFrame has zero rows.
    """
    if not len(df):
        prefix = f"[{context}] " if context is not None else ""
        raise DataFrameValidationError(f"{prefix}DataFrame is empty.")
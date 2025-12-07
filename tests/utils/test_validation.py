from __future__ import annotations

import pandas as pd
import pytest

from eb_evaluation.utils import (
    DataFrameValidationError,
    ensure_columns_present,
    ensure_non_empty,
)


def test_ensure_columns_present_passes_when_all_columns_exist():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # Should not raise
    ensure_columns_present(df, ["a", "b"], context="test")


def test_ensure_columns_present_raises_with_missing_columns_and_context():
    df = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(DataFrameValidationError) as excinfo:
        ensure_columns_present(df, ["a", "b"], context="my_function")

    msg = str(excinfo.value)
    assert "my_function" in msg
    assert "missing required columns" in msg.lower()
    assert "['b']" in msg


def test_ensure_non_empty_passes_for_non_empty_dataframe():
    df = pd.DataFrame({"a": [1, 2]})

    # Should not raise
    ensure_non_empty(df, context="test")


def test_ensure_non_empty_raises_for_empty_dataframe_with_context():
    df = pd.DataFrame({"a": []})

    with pytest.raises(DataFrameValidationError) as excinfo:
        ensure_non_empty(df, context="empty_check")

    msg = str(excinfo.value)
    assert "empty_check" in msg
    assert "dataframe is empty" in msg.lower()


def test_dataframe_validation_error_is_subclass_of_valueerror():
    """Sanity check: callers can catch it as ValueError if desired."""
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        ensure_columns_present(df, ["a", "b"])
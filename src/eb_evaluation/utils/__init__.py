"""
Utility helpers for the eb-evaluation package.

This subpackage contains small, focused utilities that are shared across
the evaluation and model-selection layers.

Currently included
------------------
DataFrame validation utilities
    - DataFrameValidationError
    - ensure_columns_present
    - ensure_non_empty

These helpers are intentionally lightweight and explicit, providing
clear, fail-fast error messages without attempting schema inference
or data coercion.
"""

from .validation import (
    DataFrameValidationError,
    ensure_columns_present,
    ensure_non_empty,
)

__all__ = [
    "DataFrameValidationError",
    "ensure_columns_present",
    "ensure_non_empty",
]

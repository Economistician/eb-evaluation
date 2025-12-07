"""
Utility helpers for the eb-evaluation package.

Currently includes:
    - DataFrame validation utilities
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
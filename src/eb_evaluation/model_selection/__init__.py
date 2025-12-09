"""
Model-selection utilities for Electric Barometer.

This subpackage provides:

- compare_forecasts: multi-model comparison on a single series
- select_model_by_cwsl: holdout-based selection using CWSL
- select_model_by_cwsl_cv: K-fold CV selection using CWSL
"""

from .compare import (
    compare_forecasts,
    select_model_by_cwsl,
    select_model_by_cwsl_cv,
)

__all__ = [
    "compare_forecasts",
    "select_model_by_cwsl",
    "select_model_by_cwsl_cv",
]
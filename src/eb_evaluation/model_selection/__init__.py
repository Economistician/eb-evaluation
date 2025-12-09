"""
Model-selection utilities for Electric Barometer.

Exports:
- compare_forecasts: metric-rich comparison of multiple forecast series.
- select_model_by_cwsl: holdout-based model selection using CWSL.
- select_model_by_cwsl_cv: cross-validated model selection using CWSL.
- ElectricBarometer: model-selection engine (core orchestrator).
- AutoEngine: convenience factory that builds a configured ElectricBarometer
  with a curated model zoo for different speed presets.
"""

from .compare import (
    compare_forecasts,
    select_model_by_cwsl,
    select_model_by_cwsl_cv,
)
from .electric_barometer import ElectricBarometer
from .auto_engine import AutoEngine
from .cwsl_regressor import CWSLRegressor

__all__ = [
    "compare_forecasts",
    "select_model_by_cwsl",
    "select_model_by_cwsl_cv",
    "ElectricBarometer",
    "AutoEngine",
]
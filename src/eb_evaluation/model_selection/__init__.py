"""
Model-selection utilities for Electric Barometer.

This subpackage provides lightweight, sklearn-compatible helpers for selecting
forecasting models using **Cost-Weighted Service Loss (CWSL)** as the primary
objective, alongside reference diagnostics (e.g., RMSE, wMAPE).

Public API
----------
compare_forecasts
    Metric-rich comparison of multiple forecast series on a common target.
select_model_by_cwsl
    Holdout-based selection of the best model by minimum validation CWSL.
select_model_by_cwsl_cv
    Cross-validated selection of the best model by minimum mean CWSL.
ElectricBarometer
    Core selection orchestrator (holdout or CV).
AutoEngine
    Convenience factory that builds a configured ElectricBarometer with a curated
    model zoo for speed presets.
CWSLRegressor
    sklearn-style estimator that wraps ElectricBarometer (fit/predict/score).

Notes
-----
The Electric Barometer selection pattern is **cost-aware selection** (not
cost-aware training): candidate models are trained using their native objectives,
then chosen by CWSL.
"""

from .auto_engine import AutoEngine
from .compare import compare_forecasts, select_model_by_cwsl, select_model_by_cwsl_cv
from .cwsl_regressor import CWSLRegressor
from .electric_barometer import ElectricBarometer

__all__ = [
    "AutoEngine",
    "CWSLRegressor",
    "ElectricBarometer",
    "compare_forecasts",
    "select_model_by_cwsl",
    "select_model_by_cwsl_cv",
]

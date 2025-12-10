"""
Feature engineering utilities for Electric Barometer model preparation.

This module provides:

- FeatureEngineer: the main transformation class
- FeatureConfig: configuration object describing lag steps, rolling windows,
  calendar features, regressors, and static metadata.

These tools convert raw panel time-series data into model-ready
(X, y, feature_names) suitable for ElectricBarometer, AutoEngine,
CWSLRegressor, or any sklearn-compatible estimator.
"""

from .feature_engineer import FeatureEngineer, FeatureConfig

__all__ = ["FeatureEngineer", "FeatureConfig"]
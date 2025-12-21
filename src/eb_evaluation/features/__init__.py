"""
Feature engineering utilities for Electric Barometer.

This subpackage contains lightweight, frequency-agnostic tools for transforming
panel time-series data into model-ready inputs.

It provides:

- :class:`~eb_evaluation.features.feature_engineer.FeatureEngineer`:
  a stateless transformer that converts a long-form panel DataFrame into
  ``(X, y, feature_names)``.
- :class:`~eb_evaluation.features.feature_engineer.FeatureConfig`:
  a configuration object describing lag steps, rolling windows, calendar features,
  external regressors, and static metadata.

These utilities are designed to interoperate cleanly with Electric Barometer
components (e.g., CWSL-based evaluation and selection) as well as standard
scikit-learn–compatible estimators.
"""

from .feature_engineer import FeatureConfig, FeatureEngineer

__all__ = ["FeatureEngineer", "FeatureConfig"]
"""
Electric Barometer Evaluation Toolkit (EB Evaluation)

This package provides DataFrame-based forecasting evaluation utilities,
including group-level metrics, hierarchical evaluation, cost-sensitive
analysis, and entity-aware diagnostics.
"""

# Re-export core evaluation functions for convenient imports

from .dataframe import (
    compute_cwsl_df,
    evaluate_groups_df,
    evaluate_hierarchy_df,
    evaluate_panel_df,
    evaluate_panel_with_entity_R,
    compute_cwsl_sensitivity_df,
    cwsl_sensitivity_df,
    estimate_entity_R_from_balance,
)

__all__ = [
    "compute_cwsl_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",
    "compute_cwsl_sensitivity_df",
    "cwsl_sensitivity_df",
    "estimate_entity_R_from_balance",
    "compare_forecasts",
    "select_model_by_cwsl",
    "select_model_by_cwsl_cv",
]
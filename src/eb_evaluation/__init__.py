"""
Electric Barometer Evaluation Toolkit (EB Evaluation)

This package provides DataFrame-based forecasting evaluation utilities,
including group-level metrics, hierarchical evaluation, cost-sensitive
analysis, and entity-aware diagnostics.
"""

# Re-export core evaluation functions for convenient imports

from .dataframe.single import compute_cwsl_df
from .dataframe.group import evaluate_groups_df
from .dataframe.hierarchy import evaluate_hierarchy_df
from .dataframe.panel import evaluate_panel_df
from .dataframe.entity import evaluate_panel_with_entity_R

__all__ = [
    "compute_cwsl_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",
]
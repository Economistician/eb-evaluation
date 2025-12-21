from __future__ import annotations

"""
DataFrame utilities for Electric Barometer evaluation.

This subpackage contains pandas-oriented helpers that orchestrate grouping, reshaping,
and tabular outputs for forecast evaluation. Metric *definitions* live in
``ebmetrics.metrics``; this package focuses on DataFrame workflows such as:

- computing CWSL on a single DataFrame slice
- evaluating per-group metric tables
- evaluating multi-level hierarchies (overall / by-store / by-item / etc.)
- producing long-form (tidy) panels for plotting and reporting
- entity-aware evaluation using entity-specific cost ratios
- CWSL cost-ratio sensitivity curves for diagnostics

The primary public functions are re-exported here for a clean API surface.
"""

from .cost_ratio import estimate_entity_R_from_balance
from .entity import evaluate_panel_with_entity_R
from .group import evaluate_groups_df
from .hierarchy import evaluate_hierarchy_df
from .panel import evaluate_panel_df
from .sensitivity import compute_cwsl_sensitivity_df
from .single import compute_cwsl_df

# Backwards-compatible alias: older code/tests may still refer to this name.
cwsl_sensitivity_df = compute_cwsl_sensitivity_df

__all__ = [
    "compute_cwsl_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",
    "compute_cwsl_sensitivity_df",
    "cwsl_sensitivity_df",
    "estimate_entity_R_from_balance",
]
"""
DataFrame utilities for Electric Barometer evaluation.

This subpackage contains pandas-oriented helpers that orchestrate grouping, reshaping,
and tabular outputs for forecast evaluation. Metric definitions live in
``eb_metrics.metrics``; this package focuses on DataFrame workflows such as:

- computing CWSL on a single DataFrame slice
- evaluating per-group metric tables
- evaluating multi-level hierarchies (overall / by-store / by-item / etc.)
- producing long-form (tidy) panels for plotting and reporting
- entity-aware evaluation using externally supplied entity-specific cost ratios

Optimization, calibration, and sensitivity analysis utilities (e.g., cost-ratio
estimation, tau selection, sensitivity sweeps) live exclusively in ``eb-optimization``.
"""

from __future__ import annotations

from .entity import evaluate_panel_with_entity_R
from .group import evaluate_groups_df
from .hierarchy import evaluate_hierarchy_df
from .panel import evaluate_panel_df
from .single import compute_cwsl_df

__all__ = [
    "compute_cwsl_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",
]

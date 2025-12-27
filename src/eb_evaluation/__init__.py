"""
Electric Barometer Evaluation Toolkit (eb-evaluation).

This package provides **DataFrame-oriented evaluation, diagnostics, and
model-selection utilities** built around Cost-Weighted Service Loss (CWSL)
and related readiness metrics.

Scope
-----
The eb-evaluation package sits *above* ``eb_metrics`` in the Electric Barometer
architecture:

- ``eb_metrics`` defines **metric math** (CWSL, NSL, UD, HR@τ, FRS, etc.)
- ``eb-evaluation`` provides **tabular orchestration**, grouping logic,
  and model selection

Primary capabilities include:

DataFrame evaluation
    - Single-slice and grouped CWSL computation
    - Hierarchical evaluation across multiple aggregation levels
    - Long-form (tidy) panel outputs for plotting and reporting
    - Entity-aware evaluation using externally supplied entity-specific cost ratios

Model selection
    - Cost-aware comparison of forecast models
    - Holdout and cross-validated selection by minimum CWSL
    - sklearn-style wrappers for downstream pipelines

Design principles
-----------------
- **Separation of concerns**: metric definitions live in ``eb_metrics``,
  orchestration and evaluation live here.
- **Operational alignment**: selection and diagnostics are driven by cost
  and readiness, not symmetric error alone.
- **Deterministic & explicit**: no hidden heuristics; all behavior is
  controlled via parameters and documented outputs.

Optimization utilities such as tolerance (τ) calibration, cost-ratio estimation,
and sensitivity sweeps live exclusively in ``eb-optimization``.

This package is intended to be used alongside ``eb_metrics`` and
``eb-adapters`` as part of the broader Electric Barometer ecosystem.
"""

from .dataframe import (
    compute_cwsl_df,
    evaluate_groups_df,
    evaluate_hierarchy_df,
    evaluate_panel_df,
    evaluate_panel_with_entity_R,
)

__all__ = [
    "compute_cwsl_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",
]

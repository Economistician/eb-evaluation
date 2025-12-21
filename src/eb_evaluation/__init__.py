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
  sensitivity analysis, tolerance calibration, and model selection

Primary capabilities include:

DataFrame evaluation
    - Single-slice and grouped CWSL computation
    - Hierarchical evaluation across multiple aggregation levels
    - Long-form (tidy) panel outputs for plotting and reporting
    - Entity-aware evaluation with entity-specific cost ratios
    - Cost-ratio (R = cu / co) sensitivity analysis

Tolerance (τ) calibration
    - Data-driven τ estimation from historical residuals
    - Global and entity-level τ estimation with governance guards
    - HR@τ computation with automatically selected tolerances

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

This package is intended to be used alongside ``eb_metrics`` and
``eb-adapters`` as part of the broader Electric Barometer ecosystem.
"""

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

from .dataframe.tolerance import (
    hr_at_tau,
    estimate_tau,
    estimate_entity_tau,
    hr_auto_tau,
    TauEstimate,
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
    "hr_at_tau",
    "estimate_tau",
    "estimate_entity_tau",
    "hr_auto_tau",
    "TauEstimate",
]

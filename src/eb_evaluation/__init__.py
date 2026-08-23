"""DataFrame evaluation, diagnostics, and CWSL-aware model selection.

Sits above ``eb_metrics`` for tabular orchestration and selection. Tuning and
runtime policies live in ``eb-optimization``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .dataframe import (
    compute_cwsl_df,
    evaluate_governance_panel_df,
    evaluate_groups_df,
    evaluate_hierarchy_df,
    evaluate_panel_df,
    evaluate_panel_with_entity_R,
    run_governance_workflow_df,
    run_governance_workflow_df_dict,
)
from .diagnostics import FASClass
from .model_selection import compare_forecasts


def _resolve_version() -> str:
    """
    Resolve the installed package version.

    Returns
    -------
    str
        Installed version string. If the package is not installed (e.g., running
        from source without installation), returns ``"0.0.0"``.
    """
    try:
        return version("eb-evaluation")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()

__all__ = [
    "FASClass",
    "__version__",
    "compare_forecasts",
    "compute_cwsl_df",
    "evaluate_governance_panel_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",
    "run_governance_workflow_df",
    "run_governance_workflow_df_dict",
]

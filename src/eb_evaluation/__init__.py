"""DataFrame evaluation, diagnostics, and CWSL-aware model selection.

Sits above ``eb_metrics`` for tabular orchestration and selection. Tuning and
runtime policies live in ``eb-optimization``.

Core DQC, FPC, FAS, and governance diagnostics are re-exported from the
package root for a stable public import surface.
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
from .diagnostics import (
    DQCClass,
    DQCResult,
    DQCSignals,
    DQCThresholds,
    FASClass,
    FASSliceMode,
    FASThresholds,
    FPCClass,
    FPCResult,
    FPCSignals,
    FPCThresholds,
    GateResult,
    GovernanceDecision,
    GovernancePreset,
    GovernanceRALPolicy,
    GovernanceResult,
    GovernanceStatus,
    GovernanceTauPolicy,
    RALPolicy,
    TauPolicy,
    build_fas_surface,
    build_signals_from_series,
    classify_dqc,
    classify_fpc,
    compute_error_anatomy,
    decide_governance,
    dqc_to_dict,
    get_governance_preset,
    preset_policy,
    preset_thresholds,
    run_governance_gate,
    slice_keys,
    snap_to_grid,
    valid_slice_modes,
    validate_dqc,
    validate_fpc,
    validate_governance,
)
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
    "DQCClass",
    "DQCResult",
    "DQCSignals",
    "DQCThresholds",
    "FASClass",
    "FASSliceMode",
    "FASThresholds",
    "FPCClass",
    "FPCResult",
    "FPCSignals",
    "FPCThresholds",
    "GateResult",
    "GovernanceDecision",
    "GovernancePreset",
    "GovernanceRALPolicy",
    "GovernanceResult",
    "GovernanceStatus",
    "GovernanceTauPolicy",
    "RALPolicy",
    "TauPolicy",
    "__version__",
    "build_fas_surface",
    "build_signals_from_series",
    "classify_dqc",
    "classify_fpc",
    "compare_forecasts",
    "compute_cwsl_df",
    "compute_error_anatomy",
    "decide_governance",
    "dqc_to_dict",
    "evaluate_governance_panel_df",
    "evaluate_groups_df",
    "evaluate_hierarchy_df",
    "evaluate_panel_df",
    "evaluate_panel_with_entity_R",
    "get_governance_preset",
    "preset_policy",
    "preset_thresholds",
    "run_governance_gate",
    "run_governance_workflow_df",
    "run_governance_workflow_df_dict",
    "slice_keys",
    "snap_to_grid",
    "valid_slice_modes",
    "validate_dqc",
    "validate_fpc",
    "validate_governance",
]

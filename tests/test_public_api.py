"""Public root-export smoke tests for ``eb_evaluation``."""

from __future__ import annotations

import eb_evaluation as m


def test_root_reexports_core_diagnostics() -> None:
    names = (
        "DQCClass",
        "classify_dqc",
        "validate_dqc",
        "FPCClass",
        "classify_fpc",
        "validate_fpc",
        "FASClass",
        "build_fas_surface",
        "GovernanceDecision",
        "decide_governance",
        "validate_governance",
        "run_governance_gate",
        "RALPolicy",
        "TauPolicy",
        "GovernanceRALPolicy",
        "GovernanceTauPolicy",
        "DQCResult",
        "apply_ral",
        "__version__",
    )
    for name in names:
        assert name in m.__all__
        assert hasattr(m, name)
    assert isinstance(m.__version__, str) and m.__version__
    assert m.GovernanceRALPolicy is m.RALPolicy
    assert m.GovernanceTauPolicy is m.TauPolicy
    assert callable(m.apply_ral)

from __future__ import annotations

"""
Entity-level cost ratio estimation (DataFrame utilities).

This module is a thin compatibility wrapper.

Implementation ownership
------------------------
Entity-level cost ratio tuning is a search / calibration task and is implemented in:

    eb_optimization.tuning.cost_ratio

`eb-evaluation` keeps this wrapper to preserve a stable, evaluation-friendly import
path for downstream users:

    from eb_evaluation.dataframe.cost_ratio import estimate_entity_R_from_balance

If `eb-optimization` is not installed, importing this module will raise an ImportError.
"""

from typing import TYPE_CHECKING

try:
    from eb_optimization.tuning.cost_ratio import estimate_entity_R_from_balance
except Exception as e:
    raise ImportError(
        "eb_evaluation.dataframe.cost_ratio requires eb-optimization. "
        "Install eb-optimization to use estimate_entity_R_from_balance()."
    ) from e


__all__ = ["estimate_entity_R_from_balance"]

if TYPE_CHECKING:
    # For static analyzers only (keeps IDE type hints)
    from eb_optimization.tuning.cost_ratio import estimate_entity_R_from_balance as estimate_entity_R_from_balance
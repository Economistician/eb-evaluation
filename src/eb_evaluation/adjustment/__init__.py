"""
Readiness Adjustment Layer (RAL)

This module provides the post-processing stage of the Electric Barometer (EB)
forecasting pipeline. The ReadinessAdjustmentLayer learns uplift factors that
minimize CWSL and optionally segment these uplifts by entity, cluster, or any
grouping columns.

Public API:
    - ReadinessAdjustmentLayer
"""

from .readiness_adjustment import ReadinessAdjustmentLayer

__all__ = [
    "ReadinessAdjustmentLayer",
]
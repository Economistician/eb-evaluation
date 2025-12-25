"""
Adjustment utilities for Electric Barometer evaluation.

The `eb_evaluation.adjustment` package contains the **Readiness Adjustment Layer (RAL)**,
a lightweight post-processing component that converts a baseline statistical forecast into an
operationally conservative *readiness forecast* via a learned multiplicative uplift.

Key ideas
---------
- **Metrics live in** `eb_metrics.metrics` (definitions only).
- **Adjustments live here** (evaluation / selection utilities that *consume* metrics).
- RAL learns an uplift by grid-searching multipliers and selecting the value that minimizes
  **Cost-Weighted Service Loss (CWSL)** on historical data.
- Uplifts can be learned globally or per-segment (with a global fallback for unseen segments).

Public API
----------
- `ReadinessAdjustmentLayer`
"""

from .ral import ReadinessAdjustmentLayer

__all__ = [
    "ReadinessAdjustmentLayer",
]

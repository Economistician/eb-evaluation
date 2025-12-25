from __future__ import annotations

"""
Compatibility wrappers for τ (tolerance) selection utilities.

The Electric Barometer project locates optimization / parameter selection logic in
`eb-optimization`. This module re-exports the public τ-selection API to preserve
backwards compatibility for `eb-evaluation` users.

Source of truth:
- `eb_optimization.tuning.tau`
"""

from eb_optimization.tuning.tau import (
    TauEstimate,
    TauMethod,
    estimate_entity_tau,
    estimate_tau,
    hr_at_tau,
    hr_auto_tau,
)
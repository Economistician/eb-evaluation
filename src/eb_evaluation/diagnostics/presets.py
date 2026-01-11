"""
Governance presets for Electric Barometer diagnostics.

This module defines small, named bundles of governance thresholds that represent
policy stances (not model tuning). Presets are intended to be:

- stable and versionable (e.g., referenced in notebooks, configs, and reports),
- auditable (explicit thresholds, no hidden behavior),
- lightweight (pure configuration; no computation).

Presets are consumed by stable entrypoints such as validate_governance to
standardize downstream behavior without requiring callers to hand-wire
thresholds on every invocation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from .dqc import DQCThresholds
from .fpc import FPCThresholds


@dataclass(frozen=True)
class GovernancePreset:
    """
    Named bundle of governance thresholds.

    Parameters
    ----------
    name:
        Stable preset identifier (e.g., "conservative", "balanced", "aggressive").
    description:
        Short human-readable summary of the policy stance.
    dqc:
        Thresholds governing Demand Quantization Compatibility (DQC).
    fpc:
        Thresholds governing Forecast Primitive Compatibility (FPC).
    """

    name: str
    description: str
    dqc: DQCThresholds
    fpc: FPCThresholds


# ---------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------
#
# Philosophy:
# - DQC thresholds govern *when snapping is required*.
# - FPC thresholds govern *how much evidence is needed to allow RAL*.
#
# The defaults in DQC/FPC are typically "balanced". Here we define explicit
# named presets so callers can reference policy by name instead of by values.
# ---------------------------------------------------------------------


CONSERVATIVE: Final[GovernancePreset] = GovernancePreset(
    name="conservative",
    description=(
        "Strict governance: require stronger quantization evidence before snapping "
        "and require larger NSL gains before declaring compatibility."
    ),
    dqc=DQCThresholds(
        # Must be at least as strict as the default (balanced) thresholds.
        multiple_rate_quantized=0.90,
        multiple_rate_packed=0.95,
        offgrid_mad_ratio_max=0.10,
        min_nonzero_obs=50,
    ),
    fpc=FPCThresholds(
        nsl_very_low=0.05,
        delta_nsl_tiny=0.03,
        hr_very_low=0.07,
        delta_hr_large_drop=-0.15,
        ud_high=12.0,
        delta_cwsl_high=0.50,
    ),
)

BALANCED: Final[GovernancePreset] = GovernancePreset(
    name="balanced",
    description=(
        "Default governance: pragmatic thresholds for typical operational usage. "
        "This is the recommended starting point."
    ),
    dqc=DQCThresholds(),
    fpc=FPCThresholds(),
)

AGGRESSIVE: Final[GovernancePreset] = GovernancePreset(
    name="aggressive",
    description=(
        "Permissive governance: snap earlier on weaker quantization evidence and "
        "declare compatibility with smaller NSL improvements."
    ),
    dqc=DQCThresholds(
        multiple_rate_quantized=0.65,
        multiple_rate_packed=0.80,
        offgrid_mad_ratio_max=0.25,
        min_nonzero_obs=20,
    ),
    fpc=FPCThresholds(
        nsl_very_low=0.02,
        delta_nsl_tiny=0.015,
        hr_very_low=0.04,
        delta_hr_large_drop=-0.10,
        ud_high=15.0,
        delta_cwsl_high=0.90,
    ),
)

# Public mapping for lookup by name.
GOVERNANCE_PRESETS: Final[Mapping[str, GovernancePreset]] = {
    CONSERVATIVE.name: CONSERVATIVE,
    BALANCED.name: BALANCED,
    AGGRESSIVE.name: AGGRESSIVE,
}

# Stable list of preset names (useful for UIs / notebooks / error messages).
GOVERNANCE_PRESET_NAMES: Final[Sequence[str]] = tuple(sorted(GOVERNANCE_PRESETS.keys()))


def list_governance_presets() -> tuple[GovernancePreset, ...]:
    """
    List all available governance presets in stable (name-sorted) order.

    Returns
    -------
    tuple[GovernancePreset, ...]
        Presets sorted by their `.name`.
    """
    return tuple(GOVERNANCE_PRESETS[name] for name in GOVERNANCE_PRESET_NAMES)


def get_governance_preset(name: str) -> GovernancePreset:
    """
    Retrieve a governance preset by name.

    Parameters
    ----------
    name:
        Preset name. One of {"conservative", "balanced", "aggressive"}.

    Returns
    -------
    GovernancePreset
        The corresponding preset.

    Raises
    ------
    KeyError
        If the preset name is unknown.
    """
    key = name.strip().lower()
    try:
        return GOVERNANCE_PRESETS[key]
    except KeyError as e:
        valid = ", ".join(GOVERNANCE_PRESET_NAMES)
        raise KeyError(f"Unknown governance preset '{name}'. Valid presets: {valid}.") from e


def preset_thresholds(preset: str | GovernancePreset) -> tuple[DQCThresholds, FPCThresholds]:
    """
    Resolve a preset into (DQCThresholds, FPCThresholds).

    Parameters
    ----------
    preset:
        Either a preset name ("conservative" | "balanced" | "aggressive") or an
        explicit GovernancePreset instance.

    Returns
    -------
    (DQCThresholds, FPCThresholds)
        Threshold objects suitable for passing to validate_governance /
        decide_governance.

    Raises
    ------
    TypeError
        If `preset` is not a `str` or `GovernancePreset`.
    ValueError
        If `preset` is a `str` but not a known preset name.
    """
    if isinstance(preset, GovernancePreset):
        p = preset
    elif isinstance(preset, str):
        try:
            p = get_governance_preset(preset)
        except KeyError as e:
            raise ValueError(str(e)) from e
    else:
        raise TypeError(f"`preset` must be a str or GovernancePreset, got {type(preset).__name__}.")

    return p.dqc, p.fpc

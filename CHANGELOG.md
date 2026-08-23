# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `FASClass` is a public ALLOWED / CONDITIONAL / BLOCKED enum and an optional upstream binding input to `decide_governance` (`BLOCKED` short-circuits; `CONDITIONAL` downgrades permissive RAL).

### Fixed

- Group-level UD averages shortfall over shortfall intervals only, matching `eb_metrics.ud`.
- FAS spike rate uses strict inequality (`abs_error > spike_ge`) to match the technical note.
- FAS preserves slices with zero valid observations and classifies them as `CONDITIONAL`.
- FAS threshold fingerprints always include `spike_ge` and use a 16-character SHA-256 prefix.
- Default governance gate applies the `balanced` nonnegativity policy (`clip_zero`) when `preset` is omitted, so audit reasons and scored forecasts cannot disagree.
- `snap_to_grid` raises on non-positive or NaN units instead of passing forecasts through.
- FPC treats missing/NaN core signals as `INCOMPATIBLE` and lets metric domain errors raise.

### Changed

- Diagnostic policy enums (`DQCClass`, `FPCClass`, `GovernanceStatus`, `TauPolicy`, `RALPolicy`) inherit `StrEnum`.
- Polished module and public API docstrings to remove prompt-spec essay prose.
- Tightened README Overview; removed cloned Role section.

## [0.2.x] - 2026-08-22

### Breaking Changes

- Removed default cost parameters (`cu=2.0`, `co=1.0`) in `evaluate_groups_df` to enforce explicit operational costs ("no hidden heuristics").

### Performance

- Vectorized group processing, reducing panel evaluation from ~373 ms to <1 ms on multi-million row panels.

### Added

- Exposed `compare_forecasts` and `__version__` on root `__all__`.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `slice_keys` default interval column is `INTERVAL_INDEX`.

### Performance

- `compute_error_anatomy` aggregates FAS slice stats with `np.bincount` / `np.add.at` and segmented quantiles instead of Python lambdas in `groupby.agg`.
- `build_signals_from_series` validates each `(y, yhat)` pair once and reuses the arrays in NSL / HR / UD / CWSL.
- `apply_ral` broadcasts `uint8` nonneg/snap codes internally and writes the existing string audit columns on export.
- `ReadinessAdjustmentLayer._best_uplift` validates `(y, yhat)` once and scores the uplift grid with `_cwsl_from_validated`.
- `evaluate_governance_panel_df` slices pre-grouped 1D forecast arrays into `run_governance_gate` instead of copying a per-stream DataFrame of `y` / `yhat`.

## [0.2.9] - 2026-08-23

### Changed

- `apply_ral`, `run_governance_workflow_df`, and `run_governance_workflow_df_dict` default `infer_policy_from_recommendations` to `False`, so snap mode stays `ceil` unless an operator opts in.
- `ReadinessAdjustmentLayer.transform` requires a valid approved governance decisions table. `apply_mask` cannot authorize writes without `decisions`.
- Development Status classifier is `5 - Production/Stable`.
- Unknown or typo FAS tokens fail-close that stream as `BLOCKED` / `RED` / `DISALLOW` instead of raising, so sibling streams can continue.

### Fixed

- `evaluate_governance_panel_df` and `run_governance_workflow_df` require FAS review. Omitted or null FAS fail-closes as `fas_class=BLOCKED`, `status=red`, `ral_policy=disallow`.
- `classify_dqc` and `run_governance_gate` fail-close on NaN or negative demand cells instead of classifying a cleaned remainder (`dqc_class=UNKNOWN`, `status=red`).
- `apply_ral` treats NA, null, or empty `fas_class` cells as `BLOCKED` and copies the baseline forecast.
- `decide_governance` and `run_governance_gate` require FAS review. Omitted or blank `fas_class` fail-closes as `BLOCKED` / `RED` / `DISALLOW`.

## [0.2.8] - 2026-08-23

### Added

- `FASClass` is a public ALLOWED / CONDITIONAL / BLOCKED enum and an optional upstream binding input to `decide_governance` (`BLOCKED` short-circuits; `CONDITIONAL` downgrades permissive RAL).
- Re-exported core DQC, FPC, FAS, and governance diagnostics from the package root.
- Public aliases `GovernanceRALPolicy` and `GovernanceTauPolicy` for the governance enums, distinct from `eb-optimization` policy artifacts.

### Fixed

- Group-level UD averages shortfall over shortfall intervals only, matching `eb_metrics.ud`.
- FAS spike rate uses strict inequality (`abs_error > spike_ge`) to match the technical note.
- FAS preserves slices with zero valid observations and classifies them as `CONDITIONAL`.
- FAS threshold fingerprints always include `spike_ge` and use a 16-character SHA-256 prefix.
- Default governance gate applies the `balanced` nonnegativity policy (`clip_zero`) when `preset` is omitted, so audit reasons and scored forecasts cannot disagree.
- `snap_to_grid` raises on non-positive or NaN units instead of passing forecasts through.
- FPC treats missing/NaN core signals as `INCOMPATIBLE` and lets metric domain errors raise.
- `decide_governance` and `apply_ral` fail closed when snapping is required without a finite positive `snap_unit`.
- `apply_ral(nonneg_mode="none")` and missing recommendation payloads follow the balanced `clip_zero` policy, matching the governance gate.
- `apply_ral` parses comma-joined recommendation strings and applies snap/nonneg policy per row instead of the first panel row.
- `apply_ral` fails closed when DQC class implies snapping even if `snap_required` is omitted.
- `apply_ral` copies governed baseline into `yhat_ral_governed` when `ral_policy` is DISALLOW, `status` is RED, `fas_class` is BLOCKED, or `dqc_class` is UNKNOWN.
- Panel recommendation strings use comma delimiters so `apply_ral` can parse per-row snap/nonneg tokens.
- `evaluate_governance_panel_df`, `run_governance_panel_df`, and the governance workflow accept per-stream `fas_class_col` / row-aligned `fas_class` Series.
- DQC `UNKNOWN` is fail-closed (`DISALLOW` / `RED`) and is never treated as continuous-like.
- `decide_governance` marks snapped FPC `INCOMPATIBLE` when snapping is required but snapped signals are omitted, instead of reusing raw FPC.
- `run_governance_panel_df` treats empty post-dropna streams as `RED` / `DISALLOW`.
- `evaluate_governance_panel_df` and `run_governance_workflow_df` fail closed on empty or non-finite `y` / `yhat` streams instead of raising from `eb-metrics`.
- `ReadinessAdjustmentLayer.transform` requires a joined decisions table or `apply_mask`; ungated calls raise so `apply_ral` remains the sole fail-closed apply path.
- Re-exported `apply_ral` from the package root.
- `snap_to_grid` raises on `None` or unparseable non-numeric values instead of passing them through.
- `apply_ral` requires a decisions table with `ral_policy`, `status`, `fas_class`, `dqc_class`, and `snap_required`; ungated calls raise.
- Injected `decisions_df` overrides in `run_governance_workflow_df` fail closed to `DISALLOW` / `RED` when required control columns are missing or NA.
- Injected `decisions_df` overrides are reconciled against the live governance gate so forged `allow` / `green` / `continuous_like` rows cannot disable required snap or flagging.
- Injected `ral_policy` uses the total order `disallow > caution_* > allow_*` and cannot upgrade a gate `caution_after_snap` to `allow`.
- `evaluate_governance_panel_df` and the governance workflow fail closed when more than 20% of a stream is non-finite or fewer than 8 finite aligned rows remain, instead of silently governing the remainder.
- Re-exported `results_to_dict` from the package root.
- `run_governance_panel_df` drops `±inf` with NaN and fail-closes non-finite streams.
- `snap_to_grid` raises on NaN or inf forecast cells instead of passing them through unsnapped.
- `_clean_nonneg` no longer swallows arbitrary exceptions; unparseable demand values fail-close DQC as `UNKNOWN`.

### Changed

- Diagnostic policy enums (`DQCClass`, `FPCClass`, `GovernanceStatus`, `TauPolicy`, `RALPolicy`) inherit `StrEnum`.
- Polished module and public API docstrings to remove prompt-spec essay prose.
- Tightened README Overview; removed cloned Role section.
- Changelog version header now matches `pyproject.toml` (`0.2.8`).
- Pinned sibling Electric Barometer packages to exact System Release 0.2.9 versions.

## [0.2.7] - 2026-08-22

### Breaking Changes

- Removed default cost parameters (`cu=2.0`, `co=1.0`) in `evaluate_groups_df` to enforce explicit operational costs ("no hidden heuristics").

### Performance

- Vectorized group processing, reducing panel evaluation from ~373 ms to <1 ms on multi-million row panels.

### Added

- Exposed `compare_forecasts` and `__version__` on root `__all__`.

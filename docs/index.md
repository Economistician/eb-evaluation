# eb-evaluation

**eb-evaluation** is the evaluation, diagnostics, and governance toolkit of the
**Electric Barometer** ecosystem.

It provides **DataFrame-oriented evaluation workflows**, **structural diagnostics**,
and **cost-aware model selection utilities** built on top of the core metrics
defined in `eb-metrics`.

While `eb-metrics` defines *how* forecast error, service loss, and readiness are
measured, **eb-evaluation** defines *how those measurements are interpreted,
validated, and governed* across entities, hierarchies, and competing models in
real operational environments.

This package is designed for **applied forecasting and decision systems** where
evaluation must be:

- cost-aware rather than purely symmetric,
- structurally valid rather than assumption-blind,
- operationally interpretable rather than abstract,
- scalable across entities, hierarchies, and time.

---

## Naming convention

Electric Barometer packages follow a consistent naming convention:

- **Distribution names** (used with `pip install`) use hyphens  
  e.g. `pip install eb-evaluation`
- **Python import paths** use underscores  
  e.g. `import eb_evaluation`

This follows standard Python packaging practices and avoids ambiguity between
package names and module imports.

---

## What this package provides

### DataFrame-based forecast evaluation
High-level utilities for applying Electric Barometer metrics to tabular data.

- Single-slice CWSL evaluation from pandas DataFrames
- Group-level evaluation (per store, per item, per entity, etc.)
- Hierarchical evaluation across multiple aggregation levels
- Long-form (tidy) panel outputs for plotting and reporting
- Entity-aware evaluation using entity-specific cost ratios

See: `docs/api/dataframe.md`

---

### Diagnostics (structural compatibility & validity)
Governance-oriented diagnostics that assess whether forecasts and evaluation
controls are **structurally compatible** with observed demand processes.

- Demand Quantization Compatibility (DQC)
- Forecast Primitive Compatibility (FPC)
- Grid detection and snap-to-grid diagnostics
- Diagnostic rationales suitable for audit and review

See: `docs/api/diagnostics.md`

---

### Governance & decision contracts
An authoritative governance layer that composes diagnostics into **explicit,
machine-readable policy decisions**.

- Snap requirements (raw units vs grid units)
- τ interpretation policy (raw vs grid units)
- Readiness Adjustment Layer (RAL) allow / caution / disallow decisions
- Traffic-light governance status (green / yellow / red)
- Stable decision contracts for downstream systems

See:
- `docs/api/governance.md`
- `docs/api/decision_lifecycle.md`

---

### Cost-ratio and sensitivity analysis
Tools for understanding how asymmetric costs influence evaluation outcomes.

- Cost-ratio sensitivity curves for CWSL
- Data-driven estimation of entity-specific cost ratios
- Diagnostics to support governance and operational review

See: `docs/api/adjustment.md`

---

### Tolerance (τ) calibration for service metrics
Data-driven selection of tolerance thresholds used in service-level metrics
such as HR@τ.

- Global τ estimation from historical residuals
- Entity-level τ estimation with minimum sample guards
- Optional global caps to prevent tolerance inflation
- Automatic HR@τ computation using estimated tolerances

---

### Cost-aware model comparison and selection
Utilities for selecting forecasting models based on **operational cost**, not
just symmetric error.

- Metric-rich comparison of multiple forecast series
- Holdout-based model selection using minimum CWSL
- Cross-validated model selection by mean CWSL
- sklearn-style wrapper (`CWSLRegressor`) for pipelines and production use
- Convenience factories for building curated model zoos

See: `docs/api/model_selection.md`

---

### Feature engineering utilities
Lightweight, frequency-agnostic feature engineering for panel time series.

- Lag and rolling-window features
- Calendar and cyclical time features
- Static metadata and external regressors
- Clean transformation into `(X, y, feature_names)` for modeling

See: `docs/api/utils.md`

---

## Documentation structure

- **API Reference**  
  Automatically generated from NumPy-style docstrings using `mkdocstrings`.

- **Conceptual & governance documentation**  
  Explains *why* diagnostics and decisions exist, not just how to call them.

Formal theory, mathematical definitions, and research context live in the
companion repository **eb-papers**.

---

## Intended audience

This package is intended for:

- data scientists and applied ML practitioners
- forecasting and demand-planning teams
- platform and data engineers integrating forecasts into operations
- operations, service, and supply-chain analytics leaders
- researchers working in cost-sensitive or service-constrained environments

The emphasis is on **decision-relevant evaluation and governance**, not isolated
metrics.

---

## Relationship to the Electric Barometer ecosystem

`eb-evaluation` provides the **evaluation, diagnostics, and governance layer**
of the Electric Barometer ecosystem. It is designed to be used alongside:

- **eb-metrics** — core metric definitions and mathematical implementations
- **eb-adapters** — integrations with external forecasting systems and models
- **eb-papers** — formal theory, definitions, and technical notes

Together, these components support a unified framework for measuring and
governing **forecast readiness**, not just forecast accuracy.

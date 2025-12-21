# eb-evaluation

**eb-evaluation** is the evaluation and model-selection toolkit of the
**Electric Barometer** ecosystem.

It provides **DataFrame-oriented workflows**, **diagnostic orchestration**, and
**cost-aware model selection utilities** built on top of the core metrics
defined in `eb-metrics`.

While `eb-metrics` defines *how* forecast error and readiness are measured,
`eb-evaluation` defines *how those measurements are applied* across groups,
hierarchies, entities, and competing models in real operational settings.

This package is designed for **applied forecasting environments** where
evaluation must be:

- cost-aware rather than purely symmetric,
- operationally interpretable rather than abstract,
- scalable across entities, hierarchies, and time.

---

## What this package provides

### DataFrame-based forecast evaluation
High-level utilities for applying Electric Barometer metrics to tabular data.

- Single-slice CWSL evaluation from pandas DataFrames
- Group-level evaluation (per store, per item, per entity, etc.)
- Hierarchical evaluation across multiple aggregation levels
- Long-form (tidy) panel outputs for plotting and reporting
- Entity-aware evaluation using entity-specific cost ratios

---

### Cost-ratio and sensitivity diagnostics
Tools for understanding how asymmetric costs influence evaluation outcomes.

- Cost-ratio sensitivity curves for CWSL
- Data-driven estimation of entity-specific cost ratios
- Diagnostics to support governance and operational review

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

---

### Feature engineering utilities
Lightweight, frequency-agnostic feature engineering for panel time series.

- Lag and rolling-window features
- Calendar and cyclical time features
- Static metadata and external regressors
- Clean transformation into `(X, y, feature_names)` for modeling

---

## Documentation structure

- **API Reference**  
  All documentation is generated automatically from NumPy-style docstrings in
  the source code using `mkdocstrings`.

Conceptual motivation, mathematical definitions, and theoretical grounding for
the Electric Barometer framework are documented in the companion research
repository **eb-papers**.

---

## Intended audience

This package is intended for:

- data scientists and applied ML practitioners
- forecasting and demand-planning teams
- operations and service analytics leaders
- researchers working in cost-sensitive or service-constrained environments

The emphasis is on **decision-relevant evaluation workflows**, not isolated
metrics.

---

## Relationship to the Electric Barometer ecosystem

`eb-evaluation` provides the **evaluation and orchestration layer** of the
Electric Barometer ecosystem. It is designed to be used alongside:

- **eb-metrics** — core metric definitions and mathematical implementations
- **eb-adapters** — integrations with external forecasting systems and models
- **eb-papers** — formal theory, definitions, and technical notes

Together, these components support a unified framework for measuring
**forecast readiness**, not just forecast error.
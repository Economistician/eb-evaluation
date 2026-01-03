# Model Selection — Electric Barometer

This document describes the **model selection layer** of the Electric Barometer (EB)
ecosystem. Model selection in EB is **cost-aware, governed, and auditable** by design.

Unlike traditional model selection frameworks that optimize symmetric error metrics
(e.g., MSE), Electric Barometer selects models based on **operational cost**, using
Cost-Weighted Service Loss (CWSL) or other explicitly configured objectives.

---

## Design Philosophy

Electric Barometer model selection is built around five core principles:

1. **Cost-aware selection, not cost-aware training**
2. **Explicit governance over candidate models**
3. **Deterministic, auditable execution**
4. **Failure isolation and bounded runtime**
5. **Minimal abstraction, maximal clarity**

The goal is not to be a general AutoML system, but a **decision-aligned selector**
suitable for production forecasting systems.

---

## Core Components

### ElectricBarometer

`ElectricBarometer` is the core selection orchestrator.

It:
- evaluates multiple candidate regressors,
- compares them using a chosen selection metric,
- selects a winner under asymmetric cost preferences,
- exposes full audit artifacts.

Supported selection modes:
- **holdout**: explicit train/validation split
- **cv**: K-fold cross-validation

Supported metrics:
- `cwsl` (default, cost-aware)
- `rmse`
- `wmape`

The selector always computes all metrics; the `metric` parameter controls **which
column is optimized**, not which diagnostics are reported.

---

### AutoEngine

`AutoEngine` is a **model-zoo factory** that builds a configured
`ElectricBarometer` with a curated set of candidate models.

It provides:
- speed presets (`fast`, `balanced`, `slow`)
- optional inclusion of boosted/tree models when dependencies are installed
- explicit include/exclude filtering
- deterministic model naming

AutoEngine is intentionally **decoupled** from selection logic:
- it decides *what can run*
- ElectricBarometer decides *what should win*

This separation keeps governance clean and extensible.

---

## Include / Exclude Governance

Both AutoEngine and ElectricBarometer support explicit model filtering:

```python
include={"linear", "ridge"}
exclude={"xgb"}
```

Properties:
- strict validation (unknown names raise)
- deterministic ordering
- enforced before any model execution

This prevents:
- accidental execution of expensive models
- silent changes in behavior as libraries are added
- wasted compute on infeasible candidates

---

## Error Handling Policy

Model failures are inevitable in real systems.

Electric Barometer makes failure behavior **explicit** via `error_policy`:

- `raise` — fail fast
- `skip` — silently skip failed models
- `warn_skip` (default) — warn and skip

All failures are recorded in:

```python
eb.failures_
```

This ensures failures are **auditable**, not hidden.

---

## Time Budgeting

Selection runs can be bounded using:

- `time_budget_s` — global wall-clock budget
- `per_model_time_budget_s` — per-candidate budget

Budgets:
- gate the *start* of model evaluation
- mark timed-out models as failures
- cannot forcibly interrupt an in-flight fit

This prevents runaway jobs while preserving correctness.

---

## Audit & Introspection Fields

After each `fit()` call, Electric Barometer exposes:

- `candidate_names_` — models considered after filtering
- `evaluated_names_` — models actually attempted
- `results_` — per-model score table
- `failures_` — failure reasons
- `stopped_early_` / `stop_reason_` — budget enforcement
- `best_name_`, `best_model_` — final selection

These fields are reset on each run.

This makes EB suitable for:
- debugging
- reporting
- governance review
- reproducibility

---

## Selection Semantics

Electric Barometer guarantees:

- deterministic selection given fixed inputs
- explicit optimization objective
- stable result schema
- no silent fallback behavior

Tie-breaking and input-validation policies are intentionally conservative and
explicit to avoid accidental decisions.

---

## What Electric Barometer Is Not

Electric Barometer intentionally avoids:

- AutoML search
- hyperparameter optimization
- ensembling
- opaque scoring logic

Those concerns belong **outside** the selection layer.

EB focuses on **choosing**, not **inventing**, models.

---

## Typical Usage

```python
engine = AutoEngine(cu=2.0, co=1.0, speed="balanced")

eb = engine.build_selector(
    X_train,
    y_train,
    include={"linear", "ridge", "gbr"},
)

eb.fit(X_train, y_train, X_val, y_val)

print(eb.best_name_)
print(eb.results_)
```

---

## Summary

Electric Barometer model selection is:

- cost-aligned
- governed
- auditable
- production-oriented

It is designed to support **operational decision systems**, not leaderboard chasing.

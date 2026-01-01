
# Diagnostics API

The Diagnostics layer of Electric Barometer provides **structural and behavioral
classification of forecast inputs and realized demand**. Diagnostics do not
optimize models or produce forecasts. They exist to answer one question:

> *Is a given forecasting primitive structurally valid for this demand process
> at this resolution?*

Diagnostics are **governance-aware**, deterministic, and auditable.

---

## Diagnostic Families

Electric Barometer defines two core diagnostic families:

| Diagnostic | Purpose |
|-----------|---------|
| DQC | Demand Quantization Compatibility |
| FPC | Forecast Primitive Compatibility |

These diagnostics are composed by the governance layer but can also be consumed
independently for analysis, reporting, or research.

---

## Demand Quantization Compatibility (DQC)

### What DQC Answers

DQC determines whether realized demand behaves as:

- **Continuous-like**
- **Quantized**
- **Piecewise packed**

This distinction is critical because tolerance-based diagnostics (τ, HR@τ) and
scale-based controls (RAL) are **invalid when applied at the wrong resolution**.

---

### DQC Signals

DQC derives the following observable signals from realized demand:

- `granularity`
- `multiple_rate`
- `offgrid_mad_ratio`
- `support_size`
- `zero_mass`

These signals are packaged into a `DQCSignals` structure and classified
deterministically.

---

### DQC Classes

| Class | Meaning |
|------|--------|
| continuous_like | Demand behaves approximately continuous |
| quantized | Strong grid structure with fixed unit |
| piecewise_packed | Multiple discrete pack sizes detected |

---

### Public API

```python
from eb_evaluation.diagnostics import validate_dqc

result = validate_dqc(y=demand_series)
```

Returned object: `DQCResult`

---

## Forecast Primitive Compatibility (FPC)

### What FPC Answers

FPC evaluates whether a **scale-based forecast adjustment** (e.g., RAL) is a
structurally valid control lever for a demand process.

It explicitly does **not** measure forecast accuracy.

---

### FPC Signals

FPC consumes precomputed observable signals:

- NSL (baseline & RAL)
- HR@τ (baseline & RAL)
- UD
- Optional CWSL

Signals must be computed upstream and passed explicitly.

---

### FPC Classes

| Class | Meaning |
|------|--------|
| compatible | Adjustment is structurally valid |
| marginal | Weak or ambiguous validity |
| incompatible | Adjustment is invalid |

---

### Public API

```python
from eb_evaluation.diagnostics import validate_fpc

result = validate_fpc(signals=fpc_signals)
```

Returned object: `FPCResult`

---

## Design Principles

### 1. Diagnostics Are Not Metrics

Diagnostics classify **structure**, not performance.

### 2. No Hidden State

All signals are passed explicitly. No implicit model calls.

### 3. Deterministic & Auditable

Given the same inputs and thresholds, results are identical.

### 4. Governance-Oriented

Diagnostics exist to support **downstream policy decisions**, not model tuning.

---

## Relationship to Governance

Diagnostics feed the governance layer but do not enforce policy themselves.

```
Demand + Signals
      ↓
   DQC / FPC
      ↓
 Governance Decision
      ↓
 Downstream Control
```

---

## When to Use Diagnostics Directly

- Exploratory analysis
- Model research
- Reporting / dashboards
- Validating new forecast primitives
- Academic or whitepaper work

For production gating, prefer the **governance API**.

---

## Summary

Diagnostics are the **structural truth layer** of Electric Barometer.

They answer *what is valid* — not *what is best*.

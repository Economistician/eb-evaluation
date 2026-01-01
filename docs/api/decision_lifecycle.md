# Decision Lifecycle (Electric Barometer)

This document describes the **end-to-end decision lifecycle** in the Electric Barometer
ecosystem—from raw data to authoritative governance decisions and downstream action.
It clarifies *when* each component is used, *what* it is responsible for, and *why*
the separation of concerns is intentional.

The lifecycle is designed to ensure:
- Auditability
- Determinism
- Governance safety
- Clear handoff between analytics and operations

---

## 1. Data Ingestion & Canonicalization

**Responsibility:** Adapters / Data Engineering

**Artifacts:**
- Raw demand / usage series
- Forecast outputs (baseline, variants)
- Metadata (entity IDs, resolution, timestamps)

**Key Properties:**
- Time-aligned
- No lookahead leakage
- Canonical units and schema

At this stage, *no evaluation or decision-making occurs*.
The system is purely observational.

---

## 2. Forecast Generation (Upstream)

**Responsibility:** Forecasting systems (outside EB)

**Artifacts:**
- Baseline forecasts
- Optional adjusted forecasts (e.g., RAL-applied)
- Confidence intervals or quantiles (optional)

Electric Barometer does **not** prescribe how forecasts are produced.
It only consumes their outputs.

---

## 3. Diagnostic Signal Construction

**Responsibility:** Evaluation layer (`eb-evaluation`)

**Artifacts:**
- FPCSignals
- DQCSignals

This step transforms raw series into **interpretable diagnostic signals**
without applying any policy judgment.

Examples:
- Non-service level (NSL)
- Under-forecast duration (UD)
- Hit rate at tolerance τ
- Demand granularity / packing signatures

No classifications are applied yet.

---

## 4. Diagnostic Classification

**Responsibility:** Diagnostics layer

### 4.1 Demand Quantization Compatibility (DQC)

Answers:
> *Does this demand behave like a continuous process or a quantized/packed one?*

Outputs:
- DQCClass
- Granularity estimate
- Supporting signals

### 4.2 Forecast Primitive Compatibility (FPC)

Answers:
> *Is a scale-based readiness adjustment structurally valid here?*

Outputs:
- FPCClass
- Rationale and supporting signals

Diagnostics are:
- Deterministic
- Threshold-governed
- Non-prescriptive

They do **not** allow or disallow actions.

---

## 5. Governance Decision

**Responsibility:** Governance layer (`decide_governance`)

This is the **authoritative decision point**.

Inputs:
- DQCResult
- FPCResult (raw)
- FPCResult (snapped, if applicable)

Outputs (GovernanceDecision):
- snap_required
- snap_unit
- τ interpretation policy
- RAL policy
- Traffic-light status (green / yellow / red)

This step answers:
> *What is allowed downstream, and under what conditions?*

Governance decisions are:
- Explicit
- Auditable
- Stable across implementations

---

## 6. Policy Enforcement (Downstream)

**Responsibility:** Adapters / Operational systems

Examples:
- Enforce snapping before applying RAL
- Block RAL application when disallowed
- Switch τ units automatically
- Gate model selection or deployment

Electric Barometer **does not execute** actions.
It provides the contract that downstream systems must respect.

---

## 7. Reporting & Oversight

**Responsibility:** Analytics / Governance

Governance artifacts can be used for:
- Dashboards
- Audits
- Change reviews
- Policy evolution

Because all decisions are deterministic, historical replay is possible.

---

## 8. Feedback & Policy Evolution

**Responsibility:** Humans

Thresholds, presets, and policies may evolve over time.
The lifecycle ensures such changes are:
- Localized
- Explicit
- Versionable

No silent behavior changes are allowed.

---

## Summary Flow

```
Raw Data
   ↓
Forecasts
   ↓
Signals
   ↓
Diagnostics (DQC / FPC)
   ↓
Governance Decision (authoritative)
   ↓
Downstream Enforcement
```

---

## Design Principle

> **Forecast accuracy is not the objective.
> Cost-aware, structurally valid readiness is.**

The decision lifecycle enforces this principle end to end.

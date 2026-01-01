# Governance

## What Governance Is (and Is Not)

Electric Barometer governance is a **deterministic decision contract** that
produces authoritative policy outcomes for downstream systems.

Governance **is**:
- A policy gate for readiness adjustment (RAL)
- A structural compatibility check between forecasts and demand processes
- An auditable, reproducible decision surface

Governance **is not**:
- A performance metric
- An optimization objective
- A forecasting model

Its sole purpose is to answer:
> *Is a scale-based readiness adjustment structurally valid here, and under what conditions?*

---

## Inputs

Governance composes the following inputs:

### Realized Demand
- A realized demand series (`y`)
- Used exclusively for Demand Quantization Compatibility (DQC)

### Forecast Primitive Compatibility (FPC) Signals
- **Raw FPC signals** (computed in raw units)
- **Snapped FPC signals** (computed after snapping forecasts to a detected demand grid)

Snapped signals may be omitted when snapping is not required.

### Optional Threshold Overrides
- `DQCThresholds`
- `FPCThresholds`

Thresholds allow governance behavior to be tuned without changing code.

---

## Outputs: GovernanceDecision

Governance produces a single, authoritative artifact:

### GovernanceDecision Fields

- `dqc`
- `fpc_raw`
- `fpc_snapped`
- `snap_required`
- `snap_unit`
- `tau_policy`
- `ral_policy`
- `status`
- `reasons`

Downstream systems should **consume these fields directly** and must not
re-implement governance logic.

---

## Decision Logic (Narrative)

### 1. Demand Structure Assessment (DQC)

Governance first determines whether realized demand is:
- Continuous-like
- Quantized
- Piecewise-packed

If demand is quantized or packed:
- Snapping is required
- Tolerances must be interpreted in grid units

### 2. Forecast Primitive Assessment (FPC)

Governance evaluates whether a scale-based adjustment (e.g., RAL):
- Produces material improvement
- Responds meaningfully to adjustment
- Operates within structurally valid limits

FPC is evaluated:
- In raw units
- In snapped units (if snapping is required)

### 3. Policy Selection

- If snapping is required → **snapped FPC governs allowability**
- If demand is continuous-like → **raw FPC governs allowability**

No mixing is permitted.

---

## Policy Semantics

### RALPolicy

- `allow`  
  Readiness adjustment is permitted in raw units.

- `allow_after_snap`  
  Adjustment is permitted only after snapping forecasts to the demand grid.

- `caution_after_snap`  
  Adjustment is structurally valid but marginal; proceed with caution.

- `disallow`  
  Adjustment is structurally invalid and must not be applied.

### TauPolicy

- `raw_units`  
  Interpret tolerance τ in the original forecast units.

- `grid_units`  
  Interpret tolerance τ in snapped (quantized) units.

### GovernanceStatus

- `green`  
  Structurally valid and safe.

- `yellow`  
  Marginal; monitor and constrain usage.

- `red`  
  Structurally invalid.

---

## Stability Guarantees

Governance provides the following guarantees:

- Deterministic outputs
- Stable field names
- Explicit reasoning via `reasons`
- Backward-compatible behavior under threshold tuning

Governance behavior may evolve **only** through:
- Threshold changes
- Explicit versioned extensions

---

## Design Principle

> Forecast accuracy is not the objective.  
> Cost-aware readiness under structural validity is.

Governance encodes this principle directly into software.

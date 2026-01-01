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

Governance composes the following inputs.

### Realized Demand

- A realized demand series (`y`)
- Used exclusively for **Demand Quantization Compatibility (DQC)**

The realized series is never pre-snapped; governance determines whether
snapping is required.

---

### Forecast Primitive Compatibility (FPC) Signals

Governance consumes **precomputed diagnostic signals**, not raw forecasts.

- **Raw FPC signals**
  - Computed in original forecast units
  - Always required

- **Snapped FPC signals**
  - Computed after snapping forecasts to a detected demand grid
  - Required *only if* governance determines snapping is necessary

Snapped signals **must not** be used unless explicitly required by DQC.

---

### Threshold Configuration

Governance behavior is controlled entirely by thresholds, never by hidden logic.

Thresholds may be supplied via:

- Explicit overrides:
  - `DQCThresholds`
  - `FPCThresholds`

- Or a **governance preset**:
  - `"conservative"`
  - `"balanced"` (default)
  - `"aggressive"`

Presets are named, versionable bundles of thresholds representing policy stance,
not model tuning.

Callers must choose **either**:
- a preset, **or**
- explicit thresholds

Mixing is intentionally disallowed.

---

## Outputs: `GovernanceDecision`

Governance produces a single, authoritative artifact:

### `GovernanceDecision` Fields

- `dqc` — Demand quantization classification and signals
- `fpc_raw` — FPC result evaluated in raw units
- `fpc_snapped` — FPC result evaluated in snapped units (if applicable)
- `snap_required` — Whether snapping to a grid is mandatory
- `snap_unit` — The detected grid unit (if snapping is required)
- `tau_policy` — How tolerance τ must be interpreted
- `ral_policy` — Whether and how readiness adjustment is allowed
- `status` — Traffic-light governance status
- `reasons` — Explicit, human-readable rationale

Downstream systems must **consume these fields directly** and must not
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
- Snapped FPC becomes authoritative

---

### 2. Forecast Primitive Assessment (FPC)

Governance evaluates whether a scale-based adjustment (e.g., RAL):

- Produces material improvement
- Responds meaningfully to adjustment
- Operates within structurally valid limits

FPC is evaluated:

- In raw units
- In snapped units (if and only if snapping is required)

---

### 3. Policy Selection

Governance enforces a strict selection rule:

- If snapping is required → **snapped FPC governs allowability**
- If demand is continuous-like → **raw FPC governs allowability**

No blending, averaging, or fallback behavior is permitted.

---

## Policy Semantics

### `RALPolicy`

- `allow`
  Readiness adjustment is permitted in raw units.

- `allow_after_snap`
  Adjustment is permitted only after snapping forecasts to the demand grid.

- `caution_after_snap`
  Adjustment is structurally valid but marginal; apply conservatively.

- `disallow`
  Adjustment is structurally invalid and must not be applied.

---

### `TauPolicy`

- `raw_units`
  Interpret tolerance τ in original forecast units.

- `grid_units`
  Interpret tolerance τ in snapped (quantized) units.

---

### `GovernanceStatus`

- `green`
  Structurally valid and safe.

- `yellow`
  Marginal; constrain usage and monitor outcomes.

- `red`
  Structurally invalid.

---

## Stability Guarantees

Governance provides the following guarantees:

- Deterministic outputs
- Stable field names and semantics
- Explicit reasoning via `reasons`
- Backward-compatible behavior under threshold or preset tuning

Governance behavior may evolve **only** through:

- Threshold changes
- Explicit, versioned extensions

---

## Design Principle

> Forecast accuracy is not the objective.
> Cost-aware readiness under structural validity is.

Governance encodes this principle directly into software.

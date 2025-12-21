# Electric Barometer Evaluation (`eb-evaluation`)

![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Docs](https://img.shields.io/badge/docs-electric--barometer-blue)](https://economistician.github.io/eb-docs/)
![Project Status](https://img.shields.io/badge/Status-Alpha-yellow)

This repository contains the **evaluation and orchestration layer** of the
*Electric Barometer* ecosystem.

`eb-evaluation` sits above core metric implementations (`eb-metrics`) and
provides structured tools for applying Electric Barometer concepts to
real-world forecasting workflows, including readiness adjustment, model
comparison, sensitivity analysis, and dataframe-based evaluation.

Conceptual definitions and theoretical framing for the evaluation logic are
maintained in the companion research repository:
**`eb-papers`**.

---

## Role Within Electric Barometer

Within the Electric Barometer ecosystem:

- **`eb-papers`** defines *concepts, frameworks, and meaning*
- **`eb-metrics`** implements *individual metrics*
- **`eb-evaluation`** orchestrates *how metrics are applied, combined, and interpreted*

This repository focuses on *evaluation logic*, not raw metric computation.

---

## What This Library Provides

- **Readiness adjustment logic** for modifying evaluation outputs based on
  operational readiness signals
- **Model selection and comparison utilities** grounded in asymmetric loss and
  readiness-aware metrics
- **Sensitivity and tolerance analysis** for cost ratios and service thresholds
- **DataFrame-oriented evaluation tools** for entity-level and time-based analysis
- **Feature engineering utilities** to support evaluation pipelines

---

## Scope

This repository focuses on **evaluation workflows and orchestration**, not
low-level metric definitions.

**In scope:**
- Applying EB metrics to datasets and model outputs
- Combining metrics into readiness-aware evaluation artifacts
- Model comparison and selection logic
- Sensitivity analysis and tolerance handling

**Out of scope:**
- Metric definitions and loss formulations (see `eb-metrics`)
- Conceptual frameworks and theory (see `eb-papers`)
- Model training or forecasting algorithms

---

## Installation

Once published, the package will be installable via PyPI:

```bash
pip install eb-evaluation
```

For development or local use:

```bash
pip install -e .
```

---

## Package Structure

The repository follows a modern Python package layout:

```text
eb-evaluation/
├── src/eb_evaluation/
│   ├── adjustment/        # Readiness and evaluation adjustments
│   ├── dataframe/         # DataFrame-based evaluation utilities
│   ├── features/          # Feature engineering helpers
│   ├── model_selection/   # Model comparison and selection logic
│   └── utils/              # Shared validation and helpers
│
├── tests/                  # Unit tests mirroring package structure
├── pyproject.toml          # Build and dependency configuration
├── README.md               # Project documentation
└── LICENSE                 # BSD-3-Clause license
```

---

## Relationship to Other EB Repositories

- `eb-papers`  
  Source of truth for conceptual definitions and evaluation philosophy.

- `eb-metrics`  
  Provides the metric implementations used during evaluation.

- `eb-evaluation`  
  Orchestrates evaluation workflows using adapted models.

- `eb-adapters`  
  Ensures heterogeneous models can be evaluated consistently.

When discrepancies arise, conceptual intent in `eb-papers` should be treated as authoritative.

---

## Development and Testing

Tests are located under the `tests/` directory and mirror the package structure.

To run the test suite:

```bash
pytest
```

---

## Status

This package is under active development.
Public APIs may evolve prior to the first stable release.
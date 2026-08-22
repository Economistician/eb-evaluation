# Electric Barometer · Evaluation (`eb-evaluation`)

[![CI](https://github.com/Economistician/eb-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-evaluation/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-evaluation)
![PyPI](https://img.shields.io/pypi/v/eb-evaluation)

Evaluation and model selection utilities for applying Electric Barometer metrics across entities, groups, and operational contexts.

---

## Overview

`eb-evaluation` provides the evaluation and model selection layer of the Electric Barometer ecosystem. It applies metric primitives to forecasts and observations across entities, groups, and hierarchical structures, enabling consistent assessment of forecasting performance in operational settings.

The package focuses on DataFrame-first evaluation workflows, including cost-sensitive comparison, tolerance-aware scoring given explicit thresholds, and readiness-oriented adjustment logic. It does not define feature construction or model interfaces; instead, it consumes standardized inputs from upstream layers and produces evaluation outputs that can be used for model selection, reporting, and decision support.

---

## Role in the Electric Barometer Ecosystem

`eb-evaluation` defines the evaluation and model selection layer used throughout the Electric Barometer ecosystem. It is responsible for applying metric primitives to forecasts and observations across entities, groups, and hierarchies, enabling consistent comparison of forecasting performance in operational contexts.

This package focuses exclusively on evaluation logic, aggregation semantics, and selection workflows. It does not perform feature construction, model training, or metric definition. Those responsibilities are handled by adjacent layers that generate inputs, adapt model interfaces, or define metric behavior.

By separating evaluation orchestration from metric semantics and model implementation details, `eb-evaluation` provides a stable, DataFrame-first foundation for decision-aligned model comparison and readiness assessment across heterogeneous forecasting pipelines.

---

## Installation

`eb-evaluation` is distributed as a standard Python package.

```bash
pip install eb-evaluation
```

The package supports Python 3.11 and later.

---

## Core Concepts

- **DataFrame-first evaluation** — Evaluation logic operates directly on tabular forecast and observation data, enabling transparent aggregation, grouping, and comparison across entities and hierarchies.
- **Cost- and tolerance-aware scoring** — Forecast performance is assessed using metrics that reflect asymmetric cost and explicitly supplied deviation thresholds, rather than purely symmetric statistical error.
- **Hierarchical and panel semantics** — Evaluation respects entity boundaries, grouping structure, and temporal alignment, ensuring correctness in multi-level forecasting environments.
- **Model comparability** — Forecasts produced by heterogeneous models can be evaluated and compared using a consistent set of metrics and aggregation rules.
- **Readiness-oriented selection** — Model selection emphasizes execution feasibility and operational adequacy as reflected in evaluation metrics, not just aggregate accuracy, supporting decision-aligned forecasting workflows.

---

## Minimal Example

The example below shows how forecast accuracy can be evaluated across entities
using Electric Barometer metrics in a DataFrame-first workflow.

```python
import pandas as pd
from eb_evaluation import compute_cwsl_df, evaluate_groups_df

df = pd.DataFrame({
    "entity_id": ["A", "A", "B", "B"],
    "actual": [10, 12, 7, 9],
    "prediction": [9, 11, 8, 10],
})

# Single-slice CWSL (scalar)
loss = compute_cwsl_df(
    df,
    y_true_col="actual",
    y_pred_col="prediction",
    cu=2.0,
    co=1.0,
)

# Per-entity metrics, including FRS (requires cwsl_max)
results = evaluate_groups_df(
    df,
    group_cols=["entity_id"],
    actual_col="actual",
    forecast_col="prediction",
    cu=2.0,
    co=1.0,
    cwsl_max=0.30,
)

print(loss)
print(results)
```

---

## License

BSD 3-Clause License.
© 2026 Kyle Corrie.

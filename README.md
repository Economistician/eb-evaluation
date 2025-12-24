# Electric Barometer · Evaluation (`eb-evaluation`)

Evaluation and model selection utilities for applying Electric Barometer metrics across entities, groups, and operational contexts.

---

## Overview

`eb-evaluation` provides the evaluation and model selection layer of the Electric Barometer ecosystem. It applies metric primitives to forecasts and observations across entities, groups, and hierarchical structures, enabling consistent assessment of forecasting performance in operational settings.

The package focuses on DataFrame-first evaluation workflows, including tolerance-based scoring, cost-sensitive comparison, and readiness-oriented adjustment logic. It does not define feature construction or model interfaces; instead, it consumes standardized inputs from upstream layers and produces evaluation outputs that can be used for model selection, reporting, and decision support.

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

The package supports Python 3.10 and later.

---

## Core Concepts

- **DataFrame-first evaluation** — Evaluation logic operates directly on tabular forecast and observation data, enabling transparent aggregation, grouping, and comparison across entities and hierarchies.
- **Cost- and tolerance-aware scoring** — Forecast performance is assessed using metrics that reflect asymmetric cost, acceptable deviation thresholds, and operational risk rather than purely symmetric statistical error.
- **Hierarchical and panel semantics** — Evaluation respects entity boundaries, grouping structure, and temporal alignment, ensuring correctness in multi-level forecasting environments.
- **Model comparability** — Forecasts produced by heterogeneous models can be evaluated and compared using a consistent set of metrics and aggregation rules.
- **Readiness-oriented selection** — Model selection emphasizes execution feasibility and operational adequacy, not just aggregate accuracy, supporting decision-aligned forecasting workflows.

---

## Minimal Example

The example below shows how forecasts and observations can be evaluated and compared across entities using Electric Barometer metrics in a DataFrame-first workflow.

```python
import pandas as pd
from eb_evaluation.dataframe.compare import compare_models

# Example evaluation data
df = pd.DataFrame({
    "entity_id": ["A", "A", "B", "B"],
    "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
    "actual": [10, 12, 7, 9],
    "model_a": [9, 11, 8, 10],
    "model_b": [11, 13, 6, 8],
})

# Compare models using a common evaluation contract
results = compare_models(
    df,
    actual_col="actual",
    prediction_cols=["model_a", "model_b"],
    entity_col="entity_id",
    time_col="date",
)

print(results)
```

---

## License

BSD 3-Clause License.  
© 2025 Kyle Corrie.
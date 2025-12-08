# EB Evaluation  
**Electric Barometer: Forecast Evaluation Toolkit**

`eb-evaluation` provides a unified, DataFrame-first interface for evaluating forecast performance across groups, hierarchies, and entities — all aligned with the Electric Barometer framework and cost-asymmetric forecasting philosophy.

It is designed to work seamlessly with the `eb-metrics` library, extending raw metric functions (CWSL, NSL, UD, wMAPE, FRS, etc.) into higher-level evaluators used in production forecasting workflows.

---

## Key Features

### DataFrame-based evaluation
Simple, explicit APIs that operate directly on pandas DataFrames.

### Group-level metrics
Evaluate CWSL and related diagnostics across any grouping dimensions.

### Hierarchical evaluation
Multi-level performance summaries (e.g., overall → store → item → store×item).

### Entity-aware evaluation using learned cost ratios
Combine interval-level data with entity-level `R` (cost ratio) estimates to compute cost-aligned metrics.

### Compatible with Electric Barometer & asymmetric forecasting
Directly consumes metrics from `eb-metrics` for CWSL-based optimization.

### Production-ready
- `src/` layout  
- Full test suite  
- Type-annotated API  
- Minimal, dependency-safe design  

---

## Installation

### Editable (development) install

```bash
pip install -e .
```

If developing alongside eb-metrics:

```bash
pip install -e ../eb-metrics
```

---

## Quick Start

1. Group-level evaluation

```python
import pandas as pd
from eb_evaluation import evaluate_groups_df

df = pd.DataFrame({
    "store_id": [1,1,2,2],
    "item_id": ["A","A","A","A"],
    "actual_qty": [10, 12, 9, 11],
    "forecast_qty": [11, 13, 8, 10],
})

summary = evaluate_groups_df(
    df=df,
    group_cols=["store_id", "item_id"],
)

print(summary)
```

2. Hierarchical evaluation

```python
from eb_evaluation import evaluate_hierarchy_df

levels = {
    "overall": [],
    "by_store": ["store_id"],
    "by_item": ["item_id"],
}

hier = evaluate_hierarchy_df(
    df,
    levels=levels,
    actual_col="actual_qty",
    forecast_col="forecast_qty",
    cu=2.0,
    co=1.0,
)
```

3. Entity-level cost-ratio evaluation

```python
from eb_evaluation import evaluate_panel_with_entity_R

entity_R = pd.DataFrame({
    "entity": ["A", "B"],
    "R": [2.0, 3.0],
    "co": [1.0, 1.0],
})

result = evaluate_panel_with_entity_R(
    df=df,
    entity_R=entity_R,
    entity_col="entity",
)
```

---

## License

Released under the **BSD-3 License**, consistent with the rest of the Electric Barometer ecosystem.
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

### Cost Sensitivity Analysis (CWSL robustness evaluation)
Evaluate how CWSL behaves across multiple candidate cost ratios  
\( R = rac{c_u}{c_o} \), enabling:
- robustness checks against cost assumptions  
- identification of stable vs. sensitive forecast behaviors  
- tuning of asymmetric cost parameters prior to model deployment  

This is powered by `compute_cwsl_sensitivity_df`, which computes CWSL over any
set of proposed R values and returns a tidy, analysis-ready DataFrame.

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

4. Cost-Ratio Estimation (R = cu/co) Utilities

Electric Barometer workflows often require setting or tuning the cost ratio R=cu/co, which determines the asymmetry between shortfall cost and overbuild cost in CWSL.

The evaluation package now includes **data-driven estimators** for R, based on cost balance:

- **Global R estimation** for a single forecast series
- **Entity-level R estimation** across a panel (e.g., item, store, SKU)

**Global Cost-Ratio Estimation**

```python
from eb_evaluation.dataframe import estimate_R_from_cost_balance

R_star = estimate_R_from_cost_balance(
    y_true=y_true,
    y_pred=y_pred,
    ratios=[0.5, 1.0, 2.0, 4.0],
    co=1.0,
)
```

`R_star` is the value in the grid that minimizes the absolute imbalance: | underbuild cost (R) - overbuild cost (R) |

This is useful for:

- selecting a reasonable global R for CWSL
- probing how sensitive model selection is to cost assumptions

**Entity-Level Cost-Ratio Estimation**

```python
from eb_evaluation.dataframe import estimate_entity_R_from_balance

result = estimate_entity_R_from_balance(
    df=df,
    entity_col="item_id",
    y_true_col="actual_qty",
    y_pred_col="forecast_qty",
    ratios=[0.5, 1.0, 2.0, 4.0],
    co=1.0,
)
```

Output columns include:

| Column        | Meaning                                 |
| ------------- | --------------------------------------- |
| `entity`      | Entity identifier                       |
| `R`           | Chosen cost ratio                       |
| `cu`          | Computed shortfall cost (cu = R * co)   |
| `co`          | Overbuild cost (input)                  |
| `under_cost`  | Total ∑ cu * shortfall                  |
| `over_cost`   | Total ∑ co * overbuild                  |
| `diff`        | Absolute imbalance at chosen R          |

This enables:

- entity-specific tuning of asymmetry
- understanding which products/stores are shortfall-leaning or overbuild-leaning
- advanced operational diagnostics inside EB

---

# Model Selection Utilities (CWSL-Driven)

`eb-evaluation` also provides a complete **model selection engine powered by CWSL**, allowing you to compare multiple forecasting models under asymmetric costs — the correct way to select models in operational environments such as QSR forecasting.

---

## 1. Compare Multiple Forecast Outputs

```python
from eb_evaluation.model_selection import compare_forecasts

df = compare_forecasts(
    y_true=y_true,
    forecasts={
        "model_a": y_pred_a,
        "model_b": y_pred_b,
    },
    cu=2.0,
    co=1.0,
)
```

Returns a tidy leaderboard DataFrame with:

- CWSL  
- NSL  
- UD  
- wMAPE  
- HR@τ  
- FRS  
- MAE  
- RMSE  
- MAPE  

---

## 2. Select Best Model on a Validation Set (CWSL-optimized)

```python
from eb_evaluation.model_selection import select_model_by_cwsl

best_name, best_model, results = select_model_by_cwsl(
    models={
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "mean": MeanModel(),
    },
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    cu=2.0,
    co=1.0,
)
```

This fits models normally but **selects** based on **minimizing CWSL**, ensuring decisions reflect real shortfall/overbuild economics.

## Model Selection (ElectricBarometer & AutoEngine)

`eb-evaluation` includes a full CWSL-driven model selection engine for operational forecasting.

### ElectricBarometer (CWSL Model Selector)
A high-level selector that:
- trains multiple candidate models  
- evaluates them with CWSL, RMSE, and wMAPE  
- selects the best model based on **minimum CWSL**  
- optionally refits the winner on full data  

Example:
```python
from eb_evaluation.model_selection import ElectricBarometer

eb = ElectricBarometer(
    models={"lr": LinearRegression(), "rf": RandomForestRegressor()},
    cu=2.0,
    co=1.0,
    selection_mode="holdout",
)

eb.fit(X_train, y_train, X_val, y_val)
pred = eb.predict(X_test)
```

### AutoEngine (Preset Model Zoo)

Convenience factory that assembles an ElectricBarometer with a curated model zoo.

```python
from eb_evaluation.model_selection import AutoEngine

engine = AutoEngine(cu=2.0, co=1.0, speed="balanced")
eb = engine.build_selector(X_train, y_train)
eb.fit(X_train, y_train, X_val, y_val)
```

---

## 3. Cross-Validated Model Selection (CWSL-CV)

```python
from eb_evaluation.model_selection import select_model_by_cwsl_cv

best_name, best_model, results = select_model_by_cwsl_cv(
    models={
        "xgboost": xgb_model,
        "zero": ZeroEstimator(),
    },
    X=X,
    y=y,
    cu=2.0,
    co=1.0,
    cv=5,
)
```

Returns:

- mean CWSL, RMSE, wMAPE  
- standard deviations  
- final model refit on full data  

This is the recommended approach for **stable, robust, production-grade model selection**.

---

## License

Released under the **BSD-3 License**, consistent with the rest of the Electric Barometer ecosystem.
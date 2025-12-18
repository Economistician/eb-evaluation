# eb-evaluation
**Electric Barometer — Evaluation & Calibration Toolkit**

`eb-evaluation` is the **evaluation and calibration layer** of the Electric Barometer ecosystem.
It extends the pure metric definitions in `ebmetrics` into **DataFrame-first, production-ready
workflows** for forecast evaluation, governance, and readiness calibration.

This package answers questions such as:
- *How good is this forecast under asymmetric costs?*
- *How should cost ratios (R) or tolerance bands (τ) be calibrated from data?*
- *How do metrics behave across entities, hierarchies, and segments?*
- *Which model should be selected when shortfalls are more expensive than overbuild?*

---

## Design Philosophy

Electric Barometer is intentionally layered:

| Layer | Responsibility |
|------|----------------|
| **ebmetrics** | Canonical metric definitions (CWSL, NSL, UD, HR@τ, FRS) |
| **eb-evaluation** | Calibration, governance, DataFrame workflows |
| **Applications / Papers** | Forecast Readiness Framework (FRF), EB deployments |

`eb-evaluation` **does not redefine metrics**.  
Instead, it *calibrates, applies, and governs* them.

---

## Key Capabilities

### DataFrame-first evaluation
All APIs operate directly on pandas DataFrames and return tidy, analysis-ready outputs.

### Group & hierarchical evaluation
Evaluate metrics across:
- arbitrary groupings
- multi-level hierarchies (overall → store → item → store×item)

### Cost-aware evaluation (CWSL-driven)
All evaluations support asymmetric cost structures using **Cost-Weighted Service Loss (CWSL)**.

### Cost-ratio (R = cu / co) calibration
Data-driven estimation of asymmetric cost ratios:
- global R calibration
- entity-level R calibration with diagnostics and safeguards

### Tolerance (τ) calibration for HR@τ
Automatic, data-driven selection of tolerance bands for **Hit Rate within Tolerance (HR@τ)**:
- target hit-rate quantiles
- knee-point detection
- utility-based tradeoffs

HR@τ itself is defined in `ebmetrics`; this package focuses on **choosing τ responsibly**.

### Feature engineering for forecasting
A modular, entity-aware feature engineering system for panel time-series data:
- lags, rolling stats, calendar features
- cyclical encodings
- static metadata
- external regressors

### Readiness Adjustment Layer (RAL)
A post-model calibration layer that learns **multiplicative uplifts** to minimize CWSL
and improve operational readiness.

### CWSL-driven model selection
Model comparison and selection utilities that choose models based on **economic impact**,
not symmetric error metrics.

---

## Installation

```bash
pip install eb-evaluation
```

For development:

```bash
pip install -e .
pip install -e ../eb-metrics
```

---

## Quick Examples

### Group-level evaluation

```python
from eb_evaluation import evaluate_groups_df

summary = evaluate_groups_df(
    df=df,
    group_cols=["store_id", "item_id"],
)
```

### Hierarchical evaluation

```python
from eb_evaluation import evaluate_hierarchy_df

levels = {
    "overall": [],
    "by_store": ["store_id"],
    "by_item": ["item_id"],
}

out = evaluate_hierarchy_df(
    df,
    levels=levels,
    actual_col="actual_qty",
    forecast_col="forecast_qty",
    cu=2.0,
    co=1.0,
)
```

### Entity-level cost ratio calibration

```python
from eb_evaluation.dataframe import estimate_entity_R_from_balance

entity_R = estimate_entity_R_from_balance(
    df=df,
    entity_col="item_id",
    y_true_col="actual_qty",
    y_pred_col="forecast_qty",
    ratios=[0.5, 1.0, 2.0, 4.0],
    co=1.0,
)
```

### Automatic τ calibration for HR@τ

```python
from eb_evaluation.dataframe.tolerance import hr_auto_tau

hr, tau, diagnostics = hr_auto_tau(
    y=y_true,
    yhat=y_pred,
    method="target_hit_rate",
    target_hit_rate=0.9,
)
```

---

## Feature Engineering

```python
from eb_evaluation.features import FeatureEngineer, FeatureConfig

fe = FeatureEngineer(
    entity_col="entity_id",
    timestamp_col="timestamp",
    target_col="target",
)

config = FeatureConfig(
    lag_steps=[1, 2, 24],
    rolling_windows=[3],
    rolling_stats=["mean", "max"],
    calendar_features=["hour", "dow", "month", "is_weekend"],
    use_cyclical_time=True,
    regressor_cols=["promo"],
    static_cols=["store_type"],
)

X, y, feature_names = fe.transform(df, config)
```

Guarantees:
- strict time ordering per entity
- no negative targets
- deterministic outputs
- sklearn-compatible arrays

---

## Readiness Adjustment Layer (RAL)

```python
from eb_evaluation.adjustment import ReadinessAdjustmentLayer

ral = ReadinessAdjustmentLayer(cu=2.0, co=1.0)

ral.fit(
    df,
    forecast_col="forecast",
    actual_col="actual",
    segment_cols=["cluster"],
)

df_future = ral.transform(
    df_future,
    forecast_col="forecast",
    output_col="readiness_forecast",
)
```

RAL is the **final mile** between statistical forecasts and operations.

---

## Model Selection (CWSL-driven)

```python
from eb_evaluation.model_selection import compare_forecasts

leaderboard = compare_forecasts(
    y_true=y_true,
    forecasts={
        "model_a": y_pred_a,
        "model_b": y_pred_b,
    },
    cu=2.0,
    co=1.0,
)
```

Cross-validated selection:

```python
from eb_evaluation.model_selection import select_model_by_cwsl_cv
```

All selection is performed by **minimizing CWSL**.

---

## Status & Scope

- Production-oriented
- Deterministic and test-covered
- No model training assumptions
- Designed for QSR, retail, and operations-heavy forecasting

---

## License

BSD-3-Clause © Economistician / Electric Barometer
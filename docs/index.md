# eb-evaluation

`eb-evaluation` provides evaluation, diagnostics, and model-selection utilities used to assess forecast quality, readiness, and decision impact within the Electric Barometer ecosystem.

This package focuses on **post-forecast analysis**: measuring performance, identifying risk, validating assumptions, and supporting model comparison and selection.

## Scope

This package is responsible for:

- Evaluating forecast outputs using cost-aware and readiness-oriented metrics
- Producing diagnostic signals for data quality, service risk, and governance
- Supporting model comparison and automated model selection
- Structuring evaluation results for downstream reporting and decision workflows

It intentionally avoids model training, optimization policy definition, or data contract enforcement.

## Contents

- **Evaluation utilities**
  Functions and helpers for computing evaluation metrics and summaries

- **Diagnostics**
  Tools for assessing data quality, forecast plausibility, and governance signals

- **Model selection**
  Utilities for comparing models and selecting candidates based on evaluation criteria

## API reference

- [Evaluation APIs](api/)

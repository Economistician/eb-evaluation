from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ebopt = pytest.importorskip(
    "eb_optimization", reason="eb-optimization is required for estimate_entity_R_from_balance()"
)

from eb_evaluation.dataframe import estimate_entity_R_from_balance  # noqa: E402


def _build_simple_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for y_true, y_pred in [(10, 12), (12, 15), (15, 18)]:
        rows.append({"entity": "A", "actual_qty": y_true, "forecast_qty": y_pred})

    for y_true, y_pred in [(10, 8), (12, 9), (15, 13)]:
        rows.append({"entity": "B", "actual_qty": y_true, "forecast_qty": y_pred})

    return pd.DataFrame(rows)


def test_estimate_entity_R_basic_structure():
    df = _build_simple_panel()

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    assert set(result["entity"]) == {"A", "B"}
    assert len(result) == 2

    expected_cols = {"entity", "R", "cu", "co", "under_cost", "over_cost", "diff"}
    assert expected_cols.issubset(result.columns)

    assert np.all(np.isfinite(result["R"]))
    assert np.all(result["R"] > 0)
    assert np.all(result["cu"] > 0)
    assert np.all(result["co"] > 0)


def test_estimate_entity_R_behavior_shortfall_vs_overbuild():
    df = _build_simple_panel()

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=(0.5, 1.0, 2.0, 4.0),
        co=1.0,
    )

    R_A = float(result.loc[result["entity"] == "A", "R"].iloc[0])
    R_B = float(result.loc[result["entity"] == "B", "R"].iloc[0])
    assert R_B >= R_A


def test_estimate_entity_R_zero_error_picks_R_near_one():
    df = pd.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "actual_qty": [10.0, 12.0, 5.0, 7.0],
            "forecast_qty": [10.0, 12.0, 5.0, 7.0],
        }
    )

    ratios = (0.4, 0.9, 1.3, 3.0)
    expected_R = 0.9

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=2.0,
    )

    for _, row in result.iterrows():
        assert np.isclose(row["R"], expected_R)
        assert np.isclose(row["cu"], expected_R * 2.0)
        assert np.isclose(row["under_cost"], 0.0)
        assert np.isclose(row["over_cost"], 0.0)
        assert np.isclose(row["diff"], 0.0)


def test_estimate_entity_R_respects_sample_weights():
    df = pd.DataFrame(
        {
            "entity": ["X", "X"],
            "actual_qty": [10.0, 10.0],
            "forecast_qty": [8.0, 12.0],
            "w_balanced": [1.0, 1.0],
            "w_shortfall_heavy": [3.0, 1.0],
        }
    )

    ratios = (0.5, 1.0, 2.0, 4.0)

    res_balanced = estimate_entity_R_from_balance(
        df=df.rename(columns={"w_balanced": "weight"}),
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        sample_weight_col="weight",
    )

    res_shortfall_heavy = estimate_entity_R_from_balance(
        df=df.rename(columns={"w_shortfall_heavy": "weight"}),
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        sample_weight_col="weight",
    )

    R_balanced = float(res_balanced.loc[res_balanced["entity"] == "X", "R"].iloc[0])
    R_shortfall_heavy = float(
        res_shortfall_heavy.loc[res_shortfall_heavy["entity"] == "X", "R"].iloc[0]
    )

    assert R_shortfall_heavy <= R_balanced


def test_estimate_entity_R_invalid_inputs():
    df = pd.DataFrame(
        {"entity": ["A"], "actual_qty": [10.0], "forecast_qty": [9.0]}
    )

    with pytest.raises(KeyError):
        estimate_entity_R_from_balance(
            df=df.drop(columns=["entity"]),
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
        )

    with pytest.raises(ValueError):
        estimate_entity_R_from_balance(
            df=df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            ratios=(0.0, -1.0),
        )

    with pytest.raises(ValueError):
        estimate_entity_R_from_balance(
            df=df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            ratios=(0.5, 1.0),
            co=0.0,
        )
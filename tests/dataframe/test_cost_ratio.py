import numpy as np
import pandas as pd
import pytest

from eb_evaluation.dataframe import estimate_entity_R_from_balance


def _build_simple_panel() -> pd.DataFrame:
    """
    Helper to construct a small panel with two entities:

    - Entity A: mostly overbuild (y_pred > y_true)
    - Entity B: mostly shortfall (y_pred < y_true)

    This lets us check that the estimated R_B (shortfall-heavy)
    ends up larger than R_A (overbuild-heavy) on a reasonable grid.
    """
    rows = []

    # Entity A: overbuild-dominant
    for y_true, y_pred in [(10, 12), (12, 15), (15, 18)]:
        rows.append(
            {
                "entity": "A",
                "actual_qty": y_true,
                "forecast_qty": y_pred,
            }
        )

    # Entity B: shortfall-dominant
    for y_true, y_pred in [(10, 8), (12, 9), (15, 13)]:
        rows.append(
            {
                "entity": "B",
                "actual_qty": y_true,
                "forecast_qty": y_pred,
            }
        )

    return pd.DataFrame(rows)


def test_estimate_entity_R_basic_structure():
    """
    Structural test:
    - one row per entity
    - expected columns present
    - R, cu, co are finite and positive
    """
    df = _build_simple_panel()

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    # Two entities → two rows
    assert set(result["entity"]) == {"A", "B"}
    assert len(result) == 2

    # Expected columns
    expected_cols = {
        "entity",
        "R",
        "cu",
        "co",
        "under_cost",
        "over_cost",
        "diff",
    }
    assert expected_cols.issubset(result.columns)

    # Basic sanity on values
    assert np.all(np.isfinite(result["R"]))
    assert np.all(result["R"] > 0)
    assert np.all(result["cu"] > 0)
    assert np.all(result["co"] > 0)


def test_estimate_entity_R_behavior_shortfall_vs_overbuild():
    """
    Behavioral test:

    - Entity A is mostly overbuild.
    - Entity B is mostly shortfall.

    On a symmetric-ish grid of R values, the shortfall-heavy entity
    should prefer a higher R (shortfalls more expensive), while
    the overbuild-heavy entity should prefer a lower R.
    """
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

    # Shortfall-dominant entity B should land at an equal or higher R
    # than overbuild-dominant entity A.
    assert R_B >= R_A


def test_estimate_entity_R_zero_error_picks_R_near_one():
    """
    Degenerate case: entity with no error (y_true == y_pred) everywhere.

    We expect:
    - chosen R to be the grid value closest to 1.0
    - under_cost == over_cost == 0
    - diff == 0
    """
    df = pd.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "actual_qty": [10.0, 12.0, 5.0, 7.0],
            "forecast_qty": [10.0, 12.0, 5.0, 7.0],
        }
    )

    # Asymmetric grid so there is a unique "closest to 1.0"
    ratios = (0.4, 0.9, 1.3, 3.0)

    result = estimate_entity_R_from_balance(
        df=df,
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=2.0,
    )

    # The closest to 1.0 in ratios is 0.9
    expected_R = 0.9

    for _, row in result.iterrows():
        assert np.isclose(row["R"], expected_R)
        assert np.isclose(row["cu"], expected_R * 2.0)  # cu = R * co
        assert np.isclose(row["under_cost"], 0.0)
        assert np.isclose(row["over_cost"], 0.0)
        assert np.isclose(row["diff"], 0.0)


def test_estimate_entity_R_respects_sample_weights():
    """
    Check that sample weights are accepted and do not raise,
    and that changing weights can change the chosen R.

    We construct one entity with one shortfall and one overbuild
    of equal magnitude. Under the cost-balance rule, overweighting
    the shortfall interval pushes the chosen R downward on a simple
    grid, because we reduce the per-unit shortfall cost (cu) needed
    to keep total under/over cost in balance.
    """
    df = pd.DataFrame(
        {
            "entity": ["X", "X"],
            "actual_qty": [10.0, 10.0],
            "forecast_qty": [8.0, 12.0],  # shortfall of 2, overbuild of 2
            "w_balanced": [1.0, 1.0],
            "w_shortfall_heavy": [3.0, 1.0],
        }
    )

    ratios = (0.5, 1.0, 2.0, 4.0)

    # Balanced weights
    res_balanced = estimate_entity_R_from_balance(
        df=df.rename(columns={"w_balanced": "weight"}),
        entity_col="entity",
        y_true_col="actual_qty",
        y_pred_col="forecast_qty",
        ratios=ratios,
        co=1.0,
        sample_weight_col="weight",
    )

    # Shortfall-heavy weights
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

    # Under cost-balance, overweighting shortfalls moves R downward
    # relative to the balanced case.
    assert R_shortfall_heavy <= R_balanced


def test_estimate_entity_R_invalid_inputs():
    """
    Basic validation tests:
    - missing required columns raises KeyError
    - non-positive ratios raise ValueError
    - non-positive co raises ValueError
    """
    df = pd.DataFrame(
        {
            "entity": ["A"],
            "actual_qty": [10.0],
            "forecast_qty": [9.0],
        }
    )

    # Missing entity_col
    with pytest.raises(KeyError):
        estimate_entity_R_from_balance(
            df=df.drop(columns=["entity"]),
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
        )

    # Non-positive ratios
    with pytest.raises(ValueError):
        estimate_entity_R_from_balance(
            df=df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            ratios=(0.0, -1.0),
        )

    # Non-positive co
    with pytest.raises(ValueError):
        estimate_entity_R_from_balance(
            df=df,
            entity_col="entity",
            y_true_col="actual_qty",
            y_pred_col="forecast_qty",
            ratios=(0.5, 1.0),
            co=0.0,
        )

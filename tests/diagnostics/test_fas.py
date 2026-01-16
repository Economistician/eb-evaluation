"""
Unit tests for Forecast Admissibility Surface (FAS) diagnostics.

These tests validate deterministic behavior, joinability, and basic invariants
of the FAS surface construction utilities. They do NOT attempt to validate any
specific threshold policy as "correct" for a domain; they validate that the
implementation is stable and auditable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eb_evaluation.diagnostics.fas import (
    FASThresholds,
    build_fas_surface,
    compute_error_anatomy,
    slice_keys,
)


def test_slice_keys_modes() -> None:
    assert slice_keys("entity") == ["forecast_entity_id"]
    assert slice_keys("entity_interval") == ["forecast_entity_id", "INTERVAL_30_INDEX"]
    assert slice_keys("site_entity_interval") == [
        "site_id",
        "forecast_entity_id",
        "INTERVAL_30_INDEX",
    ]

    with pytest.raises(ValueError, match="Unknown slice mode"):
        slice_keys("nope")  # type: ignore[arg-type]


def test_compute_error_anatomy_requires_required_columns() -> None:
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, 1, 2],
            "y": [0.0, 1.0, 2.0],
        }
    )
    # Missing yhat_col
    with pytest.raises(KeyError, match="Missing required columns"):
        compute_error_anatomy(df, yhat_col="y_hat", keys=["forecast_entity_id"])


def test_compute_error_anatomy_outputs_expected_columns_and_values() -> None:
    # Two slices:
    # - entity 1: y=[0,1,0,1], yhat=[0,1,10,1] -> abs_error=[0,0,10,0]
    # - entity 2: y=[2,2,2,2], yhat=[2,12,2,12] -> abs_error=[0,10,0,10]
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "y": [0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "y_hat": [0.0, 1.0, 10.0, 1.0, 2.0, 12.0, 2.0, 12.0],
        }
    )

    anatomy = compute_error_anatomy(
        df,
        y_col="y",
        yhat_col="y_hat",
        keys=["forecast_entity_id"],
        spike_ge=10.0,
    )

    expected_cols = {
        "forecast_entity_id",
        "n_valid",
        "n_nonzero",
        "zero_rate",
        "spike_rate",
        "p90_ae",
        "p95_ae",
        "mae",
        # New diagnostics
        "shortfall_rate",
        "shortfall_spike_rate",
        "p90_shortfall",
        "p95_shortfall",
        "mean_shortfall",
        # Audit
        "spike_ge",
    }
    assert expected_cols.issubset(set(anatomy.columns))

    # Validate spike_ge is recorded for auditability.
    assert anatomy["spike_ge"].nunique() == 1
    assert float(anatomy["spike_ge"].iloc[0]) == pytest.approx(10.0)

    row1 = anatomy.loc[anatomy["forecast_entity_id"] == 1].iloc[0]
    assert int(row1["n_valid"]) == 4
    # 2 zeros out of 4
    assert row1["zero_rate"] == pytest.approx(0.5)
    # one abs_error == 10 out of 4 when spike_ge=10
    assert row1["spike_rate"] == pytest.approx(0.25)
    assert row1["mae"] == pytest.approx(2.5)

    # Underbuild (shortfall) for entity 1:
    # y - yhat = [0-0,1-1,0-10,1-1] => [0,0,-10,0] clipped => [0,0,0,0]
    assert row1["shortfall_rate"] == pytest.approx(0.0)
    assert row1["shortfall_spike_rate"] == pytest.approx(0.0)
    assert row1["mean_shortfall"] == pytest.approx(0.0)
    assert row1["p90_shortfall"] == pytest.approx(0.0)
    assert row1["p95_shortfall"] == pytest.approx(0.0)

    row2 = anatomy.loc[anatomy["forecast_entity_id"] == 2].iloc[0]
    assert int(row2["n_valid"]) == 4
    assert row2["zero_rate"] == pytest.approx(0.0)
    # two spikes (10,10) out of 4
    assert row2["spike_rate"] == pytest.approx(0.5)
    assert row2["mae"] == pytest.approx(5.0)

    # Underbuild (shortfall) for entity 2:
    # y - yhat = [2-2,2-12,2-2,2-12] => [0,-10,0,-10] clipped => [0,0,0,0]
    assert row2["shortfall_rate"] == pytest.approx(0.0)
    assert row2["shortfall_spike_rate"] == pytest.approx(0.0)
    assert row2["mean_shortfall"] == pytest.approx(0.0)
    assert row2["p90_shortfall"] == pytest.approx(0.0)
    assert row2["p95_shortfall"] == pytest.approx(0.0)


def test_build_fas_surface_missing_required_columns_raises() -> None:
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [1],
            "n_valid": [10],
            "zero_rate": [0.1],
            # Missing spike_rate and p95_ae
        }
    )
    with pytest.raises(KeyError, match="Anatomy missing required columns"):
        build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"])


def test_build_fas_surface_classification_allowed_conditional_blocked() -> None:
    # Construct an anatomy table directly so we can deterministically hit each class.
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [101, 202, 303],
            "n_valid": [500, 500, 500],
            "n_nonzero": [500, 500, 500],
            "zero_rate": [0.0, 0.0, 0.0],
            "spike_rate": [0.01, 0.06, 0.31],  # allowed, conditional, blocked by spike
            "p90_ae": [1.0, 1.0, 1.0],
            "p95_ae": [2.0, 11.0, 2.0],  # conditional also satisfied by p95 for id=202
            "mae": [1.0, 2.0, 3.0],
            # Include spike_ge to exercise audit payload inclusion.
            "spike_ge": [10.0, 10.0, 10.0],
        }
    )

    thr = FASThresholds(
        blocked_spike_rate_ge=0.30,
        blocked_p95_ae_ge=25.0,
        conditional_spike_rate_ge=0.05,
        conditional_p95_ae_ge=10.0,
        min_valid_rows=200,
        min_nonzero_rows=None,
        # Leave shortfall thresholds off by default (None)
        blocked_shortfall_spike_rate_ge=None,
        blocked_p95_shortfall_ge=None,
        conditional_shortfall_spike_rate_ge=None,
        conditional_p95_shortfall_ge=None,
    )

    fas = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"], thr=thr)

    by_id = {int(r["forecast_entity_id"]): r for _, r in fas.iterrows()}

    assert by_id[101]["fas_class"] == "ALLOWED"
    assert by_id[101]["fas_allowed"] is True
    assert by_id[101]["fas_conditional"] is False
    assert by_id[101]["fas_blocked"] is False

    assert by_id[202]["fas_class"] == "CONDITIONAL"
    assert by_id[202]["fas_allowed"] is False
    assert by_id[202]["fas_conditional"] is True
    assert by_id[202]["fas_blocked"] is False

    assert by_id[303]["fas_class"] == "BLOCKED"
    assert by_id[303]["fas_allowed"] is False
    assert by_id[303]["fas_conditional"] is False
    assert by_id[303]["fas_blocked"] is True

    # Audit fields present and stable shape
    assert "thr_fingerprint" in fas.columns
    assert "thr_json" in fas.columns
    assert fas["thr_fingerprint"].nunique() == 1
    assert fas["thr_json"].nunique() == 1


def test_build_fas_surface_min_valid_rows_forces_conditional() -> None:
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [1],
            "n_valid": [10],  # below min_valid_rows
            "n_nonzero": [10],
            "zero_rate": [0.0],
            "spike_rate": [0.0],
            "p90_ae": [0.0],
            "p95_ae": [0.0],
            "mae": [0.0],
        }
    )
    thr = FASThresholds(min_valid_rows=200)
    fas = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"], thr=thr)
    assert fas.iloc[0]["fas_class"] == "CONDITIONAL"


def test_thr_fingerprint_changes_when_thresholds_change() -> None:
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [1],
            "n_valid": [500],
            "n_nonzero": [500],
            "zero_rate": [0.0],
            "spike_rate": [0.10],
            "p90_ae": [1.0],
            "p95_ae": [12.0],
            "mae": [3.0],
            "spike_ge": [10.0],
        }
    )

    thr_a = FASThresholds(conditional_p95_ae_ge=10.0)
    thr_b = FASThresholds(conditional_p95_ae_ge=9.0)  # change one field

    fas_a = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"], thr=thr_a)
    fas_b = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"], thr=thr_b)

    assert fas_a.iloc[0]["thr_fingerprint"] != fas_b.iloc[0]["thr_fingerprint"]
    assert fas_a.iloc[0]["thr_json"] != fas_b.iloc[0]["thr_json"]


def test_thr_fingerprint_changes_when_spike_ge_changes_in_anatomy() -> None:
    # Verify audit payload incorporates spike_ge when present.
    anatomy_a = pd.DataFrame(
        {
            "forecast_entity_id": [1],
            "n_valid": [500],
            "n_nonzero": [500],
            "zero_rate": [0.0],
            "spike_rate": [0.10],
            "p90_ae": [1.0],
            "p95_ae": [12.0],
            "mae": [3.0],
            "spike_ge": [10.0],
        }
    )
    anatomy_b = anatomy_a.copy()
    anatomy_b["spike_ge"] = 12.0

    thr = FASThresholds(conditional_p95_ae_ge=10.0)

    fas_a = build_fas_surface(anatomy=anatomy_a, keys=["forecast_entity_id"], thr=thr)
    fas_b = build_fas_surface(anatomy=anatomy_b, keys=["forecast_entity_id"], thr=thr)

    assert fas_a.iloc[0]["thr_fingerprint"] != fas_b.iloc[0]["thr_fingerprint"]
    assert fas_a.iloc[0]["thr_json"] != fas_b.iloc[0]["thr_json"]

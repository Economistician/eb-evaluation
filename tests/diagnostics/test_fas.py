"""
Unit tests for Forecast Admissibility Surface (FAS) diagnostics.

These tests validate deterministic behavior, joinability, and basic invariants
of the FAS surface construction utilities. They do NOT attempt to validate any
specific threshold policy as "correct" for a domain; they validate that the
implementation is stable and auditable.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eb_evaluation.diagnostics.fas import (
    FASThresholds,
    build_fas_surface,
    compute_error_anatomy,
    slice_keys,
    valid_slice_modes,
)


def _as_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    return arr.reshape(-1)[0]


def _rows_by_key(df: pd.DataFrame, key: str) -> dict[Any, pd.Series]:
    keys = df[key].to_numpy()
    return {keys[i]: df.iloc[i] for i in range(len(df))}


def test_slice_keys_modes() -> None:
    assert slice_keys("entity") == ["forecast_entity_id"]
    assert slice_keys("entity_interval") == ["forecast_entity_id", "INTERVAL_INDEX"]
    assert slice_keys("site_entity_interval") == [
        "site_id",
        "forecast_entity_id",
        "INTERVAL_INDEX",
    ]

    with pytest.raises(ValueError, match="Unknown slice mode"):
        slice_keys("nope")  # type: ignore[arg-type]


def test_valid_slice_modes_is_stable() -> None:
    assert valid_slice_modes() == ("entity", "entity_interval", "site_entity_interval")


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


def test_compute_error_anatomy_rejects_duplicate_keys() -> None:
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, 1],
            "y": [0.0, 1.0],
            "y_hat": [0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="keys must be unique"):
        compute_error_anatomy(
            df,
            y_col="y",
            yhat_col="y_hat",
            keys=["forecast_entity_id", "forecast_entity_id"],
        )


def test_compute_error_anatomy_rejects_duplicate_columns() -> None:
    # Duplicate y_hat columns -> df["y_hat"] becomes a DataFrame in pandas
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, 1],
            "y": [0.0, 1.0],
            "y_hat": [0.0, 1.0],
        }
    )
    df["y_hat_dup"] = df["y_hat"]
    df = df.rename(columns={"y_hat_dup": "y_hat"})  # create duplicate label

    with pytest.raises(ValueError, match="df\\.columns must be unique"):
        compute_error_anatomy(df, y_col="y", yhat_col="y_hat", keys=["forecast_entity_id"])


def test_compute_error_anatomy_coerces_numeric_and_drops_bad_rows() -> None:
    # One row contains non-numeric y_hat; it should be coerced to NaN and dropped.
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, 1, 1],
            "y": [1.0, 2.0, 3.0],
            "y_hat": [1.0, "bad", 4.0],
        }
    )

    anatomy = compute_error_anatomy(
        df,
        y_col="y",
        yhat_col="y_hat",
        keys=["forecast_entity_id"],
        spike_ge=10.0,
    )

    row = anatomy.loc[anatomy["forecast_entity_id"] == "1"].iloc[0]
    # Should have dropped the "bad" row -> only 2 valid rows remain.
    assert int(row["n_valid"]) == 2
    assert float(row["spike_ge"]) == pytest.approx(10.0)


def test_compute_error_anatomy_coerces_keys_to_string_for_joinability() -> None:
    # Mixed int/str keys are common in real pipelines (CSV/Parquet/Snowflake).
    # FAS should normalize keys to StringDtype to avoid mixed-type join/sort errors.
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, "1", 2, "2"],
            "y": [1.0, 1.0, 2.0, 2.0],
            "y_hat": [1.0, 1.0, 2.0, 2.0],
        }
    )

    anatomy = compute_error_anatomy(
        df,
        y_col="y",
        yhat_col="y_hat",
        keys=["forecast_entity_id"],
        spike_ge=10.0,
    )

    assert "forecast_entity_id" in anatomy.columns
    # We coerce keys to pandas StringDtype in compute_error_anatomy.
    assert str(anatomy["forecast_entity_id"].dtype) == "string"
    # Both groups should exist as string keys "1" and "2".
    assert set(anatomy["forecast_entity_id"].tolist()) == {"1", "2"}


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

    row1 = anatomy.loc[anatomy["forecast_entity_id"] == "1"].iloc[0]
    assert int(row1["n_valid"]) == 4
    # 2 zeros out of 4
    assert row1["zero_rate"] == pytest.approx(0.5)
    # abs_error == spike_ge is not a spike (strict inequality)
    assert row1["spike_rate"] == pytest.approx(0.0)
    assert row1["mae"] == pytest.approx(2.5)

    # Underbuild (shortfall) for entity 1:
    # y - yhat = [0-0,1-1,0-10,1-1] => [0,0,-10,0] clipped => [0,0,0,0]
    assert row1["shortfall_rate"] == pytest.approx(0.0)
    assert row1["shortfall_spike_rate"] == pytest.approx(0.0)
    assert row1["mean_shortfall"] == pytest.approx(0.0)
    assert row1["p90_shortfall"] == pytest.approx(0.0)
    assert row1["p95_shortfall"] == pytest.approx(0.0)

    row2 = anatomy.loc[anatomy["forecast_entity_id"] == "2"].iloc[0]
    assert int(row2["n_valid"]) == 4
    assert row2["zero_rate"] == pytest.approx(0.0)
    # two abs_error == 10 out of 4; equality is not a spike
    assert row2["spike_rate"] == pytest.approx(0.0)
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


def test_build_fas_surface_rejects_duplicate_keys() -> None:
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [1],
            "n_valid": [500],
            "zero_rate": [0.0],
            "spike_rate": [0.0],
            "p90_ae": [0.0],
            "p95_ae": [0.0],
            "mae": [0.0],
        }
    )
    with pytest.raises(ValueError, match="keys must be unique"):
        build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id", "forecast_entity_id"])


def test_build_fas_surface_rejects_duplicate_columns() -> None:
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [1],
            "n_valid": [500],
            "zero_rate": [0.0],
            "spike_rate": [0.0],
            "p90_ae": [0.0],
            "p95_ae": [0.0],
            "mae": [0.0],
        }
    )
    anatomy["mae_dup"] = anatomy["mae"]
    anatomy = anatomy.rename(columns={"mae_dup": "mae"})  # create duplicate label

    with pytest.raises(ValueError, match="df\\.columns must be unique"):
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

    by_id = {int(_as_scalar(k)): r for k, r in _rows_by_key(fas, "forecast_entity_id").items()}

    assert by_id[101]["fas_class"] == "ALLOWED"
    assert by_id[101]["fas_status"] == "ALLOWED"
    assert bool(_as_scalar(by_id[101]["fas_allowed"])) is True
    assert bool(_as_scalar(by_id[101]["fas_conditional"])) is False
    assert bool(_as_scalar(by_id[101]["fas_blocked"])) is False

    assert by_id[202]["fas_class"] == "CONDITIONAL"
    assert by_id[202]["fas_status"] == "CONDITIONAL"
    assert bool(_as_scalar(by_id[202]["fas_allowed"])) is False
    assert bool(_as_scalar(by_id[202]["fas_conditional"])) is True
    assert bool(_as_scalar(by_id[202]["fas_blocked"])) is False

    assert by_id[303]["fas_class"] == "BLOCKED"
    assert by_id[303]["fas_status"] == "BLOCKED"
    assert bool(_as_scalar(by_id[303]["fas_allowed"])) is False
    assert bool(_as_scalar(by_id[303]["fas_conditional"])) is False
    assert bool(_as_scalar(by_id[303]["fas_blocked"])) is True

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
    assert fas.iloc[0]["fas_status"] == "CONDITIONAL"


def test_fas_flags_are_mutually_consistent() -> None:
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [1, 2, 3],
            "n_valid": [500, 500, 500],
            "n_nonzero": [500, 500, 500],
            "zero_rate": [0.0, 0.0, 0.0],
            "spike_rate": [0.0, 0.06, 0.31],
            "p90_ae": [0.0, 0.0, 0.0],
            "p95_ae": [0.0, 11.0, 0.0],
            "mae": [0.0, 1.0, 2.0],
        }
    )
    fas = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"])

    # Exactly one of the three flags should be True per row.
    row_sums = (
        fas[["fas_allowed", "fas_conditional", "fas_blocked"]].astype(int).sum(axis=1).tolist()
    )
    assert row_sums == [1, 1, 1]


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


def test_low_support_overrides_blocked_anatomy() -> None:
    # n below min_valid_rows with extreme spike/tail must still be CONDITIONAL.
    anatomy = pd.DataFrame(
        {
            "forecast_entity_id": [1],
            "n_valid": [10],
            "n_nonzero": [10],
            "zero_rate": [0.0],
            "spike_rate": [0.99],
            "p90_ae": [80.0],
            "p95_ae": [100.0],
            "mae": [50.0],
            "spike_ge": [10.0],
        }
    )
    thr = FASThresholds(
        blocked_spike_rate_ge=0.30,
        blocked_p95_ae_ge=25.0,
        min_valid_rows=200,
    )
    fas = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"], thr=thr)
    assert fas.iloc[0]["fas_class"] == "CONDITIONAL"
    assert bool(fas.iloc[0]["fas_conditional"]) is True
    assert bool(fas.iloc[0]["fas_blocked"]) is False


def test_all_nan_slice_is_preserved_and_conditional() -> None:
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, 1, 2, 2],
            "y": [1.0, 2.0, float("nan"), float("nan")],
            "y_hat": [1.0, 2.0, float("nan"), float("nan")],
        }
    )
    anatomy = compute_error_anatomy(
        df,
        y_col="y",
        yhat_col="y_hat",
        keys=["forecast_entity_id"],
        spike_ge=10.0,
    )
    by_id = {str(_as_scalar(k)): r for k, r in _rows_by_key(anatomy, "forecast_entity_id").items()}
    assert set(by_id) == {"1", "2"}
    assert int(_as_scalar(by_id["1"]["n_valid"])) == 2
    assert int(_as_scalar(by_id["2"]["n_valid"])) == 0
    assert bool(pd.isna(_as_scalar(by_id["2"]["spike_rate"])))
    assert bool(pd.isna(_as_scalar(by_id["2"]["p95_ae"])))

    fas = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"])
    row2 = fas.loc[fas["forecast_entity_id"] == "2"].iloc[0]
    assert row2["fas_class"] == "CONDITIONAL"
    assert bool(_as_scalar(row2["fas_conditional"])) is True
    assert bool(_as_scalar(row2["fas_blocked"])) is False


def test_spike_rate_excludes_error_equal_to_spike_ge() -> None:
    # abs_error = [0, 10, 11]; only 11 exceeds spike_ge=10.
    df = pd.DataFrame(
        {
            "forecast_entity_id": [1, 1, 1],
            "y": [0.0, 0.0, 0.0],
            "y_hat": [0.0, 10.0, 11.0],
        }
    )
    anatomy = compute_error_anatomy(
        df,
        y_col="y",
        yhat_col="y_hat",
        keys=["forecast_entity_id"],
        spike_ge=10.0,
    )
    row = anatomy.iloc[0]
    assert int(row["n_valid"]) == 3
    assert float(row["spike_rate"]) == pytest.approx(1.0 / 3.0)


def test_thr_fingerprint_identity_is_stable_and_includes_spike_ge() -> None:
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
    thr = FASThresholds(conditional_p95_ae_ge=10.0)

    fas_a = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"], thr=thr)
    fas_b = build_fas_surface(anatomy=anatomy, keys=["forecast_entity_id"], thr=thr)

    fp_a = str(fas_a.iloc[0]["thr_fingerprint"])
    fp_b = str(fas_b.iloc[0]["thr_fingerprint"])
    assert fp_a == fp_b
    assert len(fp_a) == 16
    assert all(ch in "0123456789abcdef" for ch in fp_a)

    payload_a = json.loads(str(fas_a.iloc[0]["thr_json"]))
    payload_b = json.loads(str(fas_b.iloc[0]["thr_json"]))
    assert payload_a == payload_b
    assert "spike_ge" in payload_a
    assert payload_a["spike_ge"] == pytest.approx(10.0)

    # Missing spike_ge column still serializes the key (as null).
    anatomy_no_spike = anatomy.drop(columns=["spike_ge"])
    fas_c = build_fas_surface(anatomy=anatomy_no_spike, keys=["forecast_entity_id"], thr=thr)
    payload_c = json.loads(str(fas_c.iloc[0]["thr_json"]))
    assert "spike_ge" in payload_c
    assert payload_c["spike_ge"] is None
    fp_c = str(fas_c.iloc[0]["thr_fingerprint"])
    assert fp_c == str(
        build_fas_surface(anatomy=anatomy_no_spike, keys=["forecast_entity_id"], thr=thr).iloc[0][
            "thr_fingerprint"
        ]
    )
    assert fp_c != fp_a

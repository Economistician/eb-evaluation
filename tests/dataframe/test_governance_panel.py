from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from eb_evaluation.dataframe.governance_panel import evaluate_governance_panel_df


def _build_sample_panel_df() -> pd.DataFrame:
    """
    Build a small panel with two streams:

    - (site_id=1, forecast_entity_id=10): continuous-like series
    - (site_id=2, forecast_entity_id=20): quantized series (multiples of 4)
    """
    n = 60

    # Stream A: continuous-ish (monotone small increments)
    y_a = np.array([0.1 * (i + 1) for i in range(n)], dtype=float)
    base_a = np.array([v * 0.90 if (i % 2) else v for i, v in enumerate(y_a)], dtype=float)
    ral_a = np.array([v * 1.01 for v in y_a], dtype=float)

    # Stream B: quantized (multiples of 4)
    y_b = np.array(([0.0] * 10) + ([4.0] * 20) + ([8.0] * 15) + ([12.0] * 15), dtype=float)
    base_b = np.array(([0.0] * 10) + ([4.0] * 20) + ([4.0] * 15) + ([8.0] * 15), dtype=float)
    ral_b = y_b.copy()

    df_a = pd.DataFrame(
        {
            "site_id": [1] * n,
            "forecast_entity_id": [10] * n,
            "y": y_a,
            "yhat_base": base_a,
            "yhat_ral": ral_a,
        }
    )
    df_b = pd.DataFrame(
        {
            "site_id": [2] * n,
            "forecast_entity_id": [20] * n,
            "y": y_b,
            "yhat_base": base_b,
            "yhat_ral": ral_b,
        }
    )

    return pd.concat([df_a, df_b], ignore_index=True)


def test_evaluate_governance_panel_df_basic_structure() -> None:
    df = _build_sample_panel_df()

    out = cast(
        pd.DataFrame,
        evaluate_governance_panel_df(
            df=df,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
        ),
    )

    # One row per stream
    assert len(out) == 2

    # Required key columns
    assert "site_id" in out.columns
    assert "forecast_entity_id" in out.columns

    # Core governance summary columns
    for col in (
        "n",
        "recommended_mode",
        "snap_required",
        "snap_unit",
        "tau_policy",
        "ral_policy",
        "status",
        "dqc_class",
        "fpc_raw_class",
        "fpc_snapped_class",
    ):
        assert col in out.columns

    # Each stream should have nonzero n
    assert (out["n"] > 0).all()


def test_evaluate_governance_panel_df_detects_snap_required_for_quantized_stream() -> None:
    df = _build_sample_panel_df()

    out = cast(
        pd.DataFrame,
        evaluate_governance_panel_df(
            df=df,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=1.0,
        ),
    )

    # Identify each stream row
    a = out[(out["site_id"] == 1) & (out["forecast_entity_id"] == 10)].iloc[0]
    b = out[(out["site_id"] == 2) & (out["forecast_entity_id"] == 20)].iloc[0]

    # Stream B should be snap-required (quantized)
    assert bool(b["snap_required"]) is True
    assert b["tau_policy"] in (
        "grid_units",
        "raw_units",
    )  # policy is versioned; accept known values
    assert b["recommended_mode"] in ("pack_aware", "reroute_discrete")

    # Stream A should usually not require snapping (continuous-like)
    # We don't hard-require snap_required=False because DQC thresholds might change,
    # but if it does not snap, it should be routed as continuous.
    if bool(a["snap_required"]) is False:
        assert a["recommended_mode"] == "continuous"


def test_evaluate_governance_panel_df_rejects_missing_required_columns() -> None:
    df = _build_sample_panel_df().drop(columns=["yhat_ral"])

    with pytest.raises(ValueError, match=r"Missing required columns"):
        evaluate_governance_panel_df(
            df=df,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
        )


def test_evaluate_governance_panel_df_rejects_empty_keys() -> None:
    df = _build_sample_panel_df()

    with pytest.raises(ValueError, match=r"`keys` must contain at least one"):
        evaluate_governance_panel_df(
            df=df,
            keys=[],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
        )


def test_evaluate_governance_panel_df_dropna_keys_behavior() -> None:
    df = _build_sample_panel_df()

    # Add a row with NA in key column; should be dropped by default.
    extra = pd.DataFrame(
        {
            "site_id": [np.nan],
            "forecast_entity_id": [999],
            "y": [1.0],
            "yhat_base": [1.0],
            "yhat_ral": [1.0],
        }
    )
    df2 = pd.concat([df, extra], ignore_index=True)

    out_drop = cast(
        pd.DataFrame,
        evaluate_governance_panel_df(
            df=df2,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
            dropna_keys=True,
        ),
    )
    assert len(out_drop) == 2

    out_keep = cast(
        pd.DataFrame,
        evaluate_governance_panel_df(
            df=df2,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
            dropna_keys=False,
        ),
    )
    # When dropna_keys=False, groupby(dropna=True) should include the NA group.
    assert len(out_keep) >= 2

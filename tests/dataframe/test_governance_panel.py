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

    # Stream A should usually not require snapping (continuous-like).
    # If it does not snap, it should be routed as continuous.
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


def test_evaluate_governance_panel_recommendations_use_comma_delimiter() -> None:
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
    recs = out["recommendations"].astype(str)
    assert recs.str.contains(r"\|", regex=True).sum() == 0
    pack = out[(out["site_id"] == 2) & (out["forecast_entity_id"] == 20)].iloc[0]
    rec = str(pack["recommendations"])
    if "snap_forecasts_to_grid" in rec:
        assert "," in rec


def test_evaluate_governance_panel_per_stream_fas_class_col() -> None:
    df = _build_sample_panel_df()
    df["fas_class"] = np.where(df["forecast_entity_id"] == 10, "BLOCKED", "ALLOWED")
    out = cast(
        pd.DataFrame,
        evaluate_governance_panel_df(
            df=df,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
            fas_class_col="fas_class",
        ),
    )
    blocked = out[(out["site_id"] == 1) & (out["forecast_entity_id"] == 10)].iloc[0]
    allowed = out[(out["site_id"] == 2) & (out["forecast_entity_id"] == 20)].iloc[0]
    assert blocked["ral_policy"] == "disallow"
    assert blocked["status"] == "red"
    assert blocked["fas_class"] == "BLOCKED"
    assert allowed["fas_class"] == "ALLOWED"


def test_run_governance_panel_df_per_stream_fas_class_col() -> None:
    from eb_evaluation.dataframe.panel import run_governance_panel_df

    df = _build_sample_panel_df()
    df["fas_class"] = np.where(df["forecast_entity_id"] == 10, "BLOCKED", "ALLOWED")
    out = run_governance_panel_df(
        df=df,
        group_cols=["site_id", "forecast_entity_id"],
        actual_col="y",
        forecast_base_col="yhat_base",
        forecast_ral_col="yhat_ral",
        tau=2.0,
        fas_class_col="fas_class",
    )
    blocked = out[(out["site_id"] == 1) & (out["forecast_entity_id"] == 10)].iloc[0]
    allowed = out[(out["site_id"] == 2) & (out["forecast_entity_id"] == 20)].iloc[0]
    assert blocked["ral_policy"] == "disallow"
    assert blocked["status"] == "red"
    assert blocked["fas_class"] == "BLOCKED"
    assert allowed["fas_class"] == "ALLOWED"
    recs = out["recommendations"].astype(str)
    assert recs.str.contains(r"\|", regex=True).sum() == 0


def test_run_governance_panel_df_empty_after_dropna_is_red_disallow() -> None:
    from eb_evaluation.dataframe.panel import run_governance_panel_df

    df = pd.DataFrame(
        {
            "site_id": [1, 1],
            "forecast_entity_id": [99, 99],
            "y": [np.nan, np.nan],
            "yhat_base": [np.nan, 1.0],
            "yhat_ral": [2.0, np.nan],
        }
    )
    out = run_governance_panel_df(
        df=df,
        group_cols=["site_id", "forecast_entity_id"],
        actual_col="y",
        forecast_base_col="yhat_base",
        forecast_ral_col="yhat_ral",
        tau=2.0,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["n_points_used"]) == 0
    assert row["warnings"] == "empty_series_after_dropna"
    assert row["ral_policy"] == "disallow"
    assert row["status"] == "red"
    assert row["fpc_raw_class"] == "incompatible"
    assert row["recommended_mode"] == "reroute_discrete"


def test_evaluate_governance_panel_df_nan_stream_fails_closed() -> None:
    n_ok = 10
    df = pd.DataFrame(
        {
            "site_id": [1, 1, *([2] * n_ok)],
            "forecast_entity_id": [99, 99, *([10] * n_ok)],
            "y": [np.nan, np.nan, *list(range(1, n_ok + 1))],
            "yhat_base": [np.nan, 1.0, *list(range(1, n_ok + 1))],
            "yhat_ral": [2.0, np.nan, *[v + 0.1 for v in range(1, n_ok + 1)]],
        }
    )
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
    assert len(out) == 2
    nan_row = out[(out["site_id"] == 1) & (out["forecast_entity_id"] == 99)].iloc[0]
    ok_row = out[(out["site_id"] == 2) & (out["forecast_entity_id"] == 10)].iloc[0]
    assert nan_row["status"] == "red"
    assert nan_row["ral_policy"] == "disallow"
    assert nan_row["fpc_raw_class"] == "incompatible"
    assert nan_row["fpc_snapped_class"] == "incompatible"
    assert nan_row["dqc_class"] == "unknown"
    assert nan_row["recommended_mode"] == "reroute_discrete"
    assert nan_row["recommendations"] == "empty_series_fail_closed"
    assert int(ok_row["n"]) == n_ok
    assert int(ok_row["n_finite"]) == n_ok
    assert ok_row["recommendations"] != "empty_series_fail_closed"
    assert ok_row["recommendations"] != "insufficient_finite_coverage_fail_closed"


def test_evaluate_governance_panel_df_sparse_finite_coverage_fails_closed() -> None:
    n = 10
    y = [float(i + 1) for i in range(n)]
    yhat_base = [float(i + 1) if i < 7 else np.nan for i in range(n)]
    yhat_ral = [float(i + 1) for i in range(n)]
    df = pd.DataFrame(
        {
            "site_id": [1] * n,
            "forecast_entity_id": [50] * n,
            "y": y,
            "yhat_base": yhat_base,
            "yhat_ral": yhat_ral,
        }
    )
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
    row = out.iloc[0]
    assert int(row["n"]) == n
    assert int(row["n_finite"]) == 7
    assert float(row["finite_coverage"]) == pytest.approx(0.7)
    assert row["status"] == "red"
    assert row["ral_policy"] == "disallow"
    assert row["fpc_raw_class"] == "incompatible"
    assert row["recommendations"] == "insufficient_finite_coverage_fail_closed"


def test_evaluate_governance_panel_df_short_finite_window_fails_closed() -> None:
    df = pd.DataFrame(
        {
            "site_id": [1, 1],
            "forecast_entity_id": [10, 10],
            "y": [1.0, 2.0],
            "yhat_base": [1.0, 2.0],
            "yhat_ral": [1.1, 2.1],
        }
    )
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
    row = out.iloc[0]
    assert int(row["n_finite"]) == 2
    assert row["status"] == "red"
    assert row["ral_policy"] == "disallow"
    assert row["fpc_raw_class"] == "incompatible"
    assert row["recommendations"] == "insufficient_finite_coverage_fail_closed"


def test_evaluate_governance_panel_df_inf_stream_fails_closed() -> None:
    df = pd.DataFrame(
        {
            "site_id": [1, 1],
            "forecast_entity_id": [99, 99],
            "y": [1.0, 2.0],
            "yhat_base": [np.inf, -np.inf],
            "yhat_ral": [1.0, 2.0],
        }
    )
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
    row = out.iloc[0]
    assert row["status"] == "red"
    assert row["ral_policy"] == "disallow"
    assert row["fpc_raw_class"] == "incompatible"
    assert row["recommendations"] == "empty_series_fail_closed"


def test_run_governance_panel_df_inf_stream_fails_closed() -> None:
    from eb_evaluation.dataframe.panel import run_governance_panel_df

    df = pd.DataFrame(
        {
            "site_id": [1, 1],
            "forecast_entity_id": [99, 99],
            "y": [np.inf, 1.0],
            "yhat_base": [1.0, np.inf],
            "yhat_ral": [2.0, 2.0],
        }
    )
    out = run_governance_panel_df(
        df=df,
        group_cols=["site_id", "forecast_entity_id"],
        actual_col="y",
        forecast_base_col="yhat_base",
        forecast_ral_col="yhat_ral",
        tau=2.0,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["n_points_used"]) == 0
    assert row["ral_policy"] == "disallow"
    assert row["status"] == "red"
    assert row["fpc_raw_class"] == "incompatible"

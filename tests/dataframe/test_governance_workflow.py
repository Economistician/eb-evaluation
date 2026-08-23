from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from eb_evaluation.dataframe.governance_workflow import (
    run_governance_workflow_df,
    run_governance_workflow_df_dict,
)


def _build_sample_panel_df() -> pd.DataFrame:
    """
    Build a small panel with two streams:

    - (site_id=1, forecast_entity_id=10): continuous-ish series
    - (site_id=2, forecast_entity_id=20): quantized series (multiples of 4)
    """
    n = 60

    # Stream A: continuous-ish
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


def test_run_governance_workflow_df_returns_panel_and_decisions() -> None:
    df = _build_sample_panel_df()

    panel, decisions = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
    )

    assert isinstance(panel, pd.DataFrame)
    assert isinstance(decisions, pd.DataFrame)

    # Decisions: one row per stream
    assert len(decisions) == 2
    assert {
        "site_id",
        "forecast_entity_id",
        "recommended_mode",
        "snap_required",
    }.issubset(decisions.columns)

    # Panel: governed columns exist (defaults are <forecast_col>_governed)
    assert "yhat_base_governed" in panel.columns
    assert "yhat_ral_governed" in panel.columns

    # Audit columns exist
    assert "ral_apply_nonneg_policy" in panel.columns
    assert "ral_apply_snap_mode" in panel.columns

    # Join succeeded (no missing decisions when require_complete_decisions=True)
    assert panel["snap_required"].isna().sum() == 0


def test_run_governance_workflow_df_uses_custom_output_column_names() -> None:
    df = _build_sample_panel_df()

    panel, _ = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
        out_base_col="base_gov",
        out_ral_col="ral_gov",
    )

    assert "base_gov" in panel.columns
    assert "ral_gov" in panel.columns
    assert "yhat_base_governed" not in panel.columns
    assert "yhat_ral_governed" not in panel.columns


def test_run_governance_workflow_df_dict_shape() -> None:
    df = _build_sample_panel_df()

    out = run_governance_workflow_df_dict(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
    )

    assert set(out.keys()) == {"panel", "decisions"}
    assert isinstance(out["panel"], pd.DataFrame)
    assert isinstance(out["decisions"], pd.DataFrame)


def test_run_governance_workflow_df_rejects_missing_required_columns() -> None:
    df = _build_sample_panel_df().drop(columns=["yhat_ral"])

    with pytest.raises(ValueError, match=r"Missing required columns"):
        _ = run_governance_workflow_df(
            df=df,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
        )


def test_run_governance_workflow_df_rejects_empty_keys() -> None:
    df = _build_sample_panel_df()

    with pytest.raises(ValueError, match=r"`keys` must contain at least one"):
        _ = run_governance_workflow_df(
            df=df,
            keys=[],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
        )


def test_run_governance_workflow_df_can_disable_decision_completeness_check() -> None:
    df = _build_sample_panel_df()

    panel, _ = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
        require_complete_decisions=False,
    )

    assert isinstance(panel, pd.DataFrame)


def test_run_governance_workflow_df_completeness_check_raises_when_missing_decisions() -> None:
    df = _build_sample_panel_df()

    # Add a row with NA in a key column. With dropna_keys=True (default),
    # the decisions table will drop this row, but the apply step will still
    # contain it, creating a missing-decision join that must raise.
    extra = pd.DataFrame(
        {
            "site_id": [np.nan],
            "forecast_entity_id": [999],
            "y": [1.0],
            "yhat_base": [1.0],
            "yhat_ral": [1.0],
        }
    )
    df_bad = pd.concat([df, extra], ignore_index=True)

    with pytest.raises(ValueError, match=r"Missing governance decisions"):
        _ = run_governance_workflow_df(
            df=df_bad,
            keys=["site_id", "forecast_entity_id"],
            actual_col="y",
            base_forecast_col="yhat_base",
            ral_forecast_col="yhat_ral",
            tau=2.0,
            require_complete_decisions=True,
        )


def test_run_governance_workflow_df_suppresses_ral_when_fas_blocked() -> None:
    df = _build_sample_panel_df()
    df["fas_class"] = np.where(df["forecast_entity_id"] == 10, "BLOCKED", "ALLOWED")
    panel, decisions = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
        fas_class_col="fas_class",
    )
    blocked = panel.loc[panel["forecast_entity_id"] == 10]
    np.testing.assert_allclose(
        blocked["yhat_ral_governed"].to_numpy(dtype=float),
        blocked["yhat_base_governed"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    applied = blocked["ral_apply_ral_applied"].to_numpy(dtype=bool)
    assert not bool(applied.any())
    recs = decisions["recommendations"].astype(str)
    assert recs.str.contains(r"\|", regex=True).sum() == 0


def test_run_governance_workflow_df_nan_stream_fails_closed() -> None:
    df = pd.DataFrame(
        {
            "site_id": [1, 1],
            "forecast_entity_id": [99, 99],
            "y": [np.nan, np.nan],
            "yhat_base": [1.0, 2.0],
            "yhat_ral": [9.0, 10.0],
        }
    )
    panel, decisions = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
    )
    assert len(decisions) == 1
    row = decisions.iloc[0]
    assert row["status"] == "red"
    assert row["ral_policy"] == "disallow"
    assert row["fpc_raw_class"] == "incompatible"
    assert row["recommended_mode"] == "reroute_discrete"
    np.testing.assert_allclose(
        panel["yhat_ral_governed"].to_numpy(dtype=float),
        panel["yhat_base_governed"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    assert not panel["ral_apply_ral_applied"].to_numpy(dtype=bool).any()


def test_run_governance_workflow_df_injected_decisions_missing_controls_fail_closed() -> None:
    df = _build_sample_panel_df()
    decisions = pd.DataFrame(
        {
            "site_id": [1, 2],
            "forecast_entity_id": [10, 20],
        }
    )
    panel, out_decisions = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
        decisions_df=decisions,
    )
    assert (out_decisions["ral_policy"] == "disallow").all()
    assert (out_decisions["status"] == "red").all()
    row_q = out_decisions[out_decisions["forecast_entity_id"] == 20].iloc[0]
    assert str(row_q["dqc_class"]).lower() in {"quantized", "piecewise_packed"}
    assert bool(row_q["snap_required"]) is True
    np.testing.assert_allclose(
        panel["yhat_ral_governed"].to_numpy(dtype=float),
        panel["yhat_base_governed"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    assert not panel["ral_apply_ral_applied"].to_numpy(dtype=bool).any()


def test_run_governance_workflow_df_injected_decisions_na_controls_fail_closed() -> None:
    df = _build_sample_panel_df()
    decisions = pd.DataFrame(
        {
            "site_id": [1, 2],
            "forecast_entity_id": [10, 20],
            "ral_policy": ["allow", np.nan],
            "status": ["green", "green"],
            "fas_class": ["ALLOWED", "ALLOWED"],
            "dqc_class": ["continuous_like", "quantized"],
            "snap_required": [False, True],
            "snap_unit": [np.nan, 4.0],
        }
    )
    panel, out_decisions = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
        decisions_df=decisions,
    )
    row_na = out_decisions[out_decisions["forecast_entity_id"] == 20].iloc[0]
    assert row_na["ral_policy"] == "disallow"
    assert row_na["status"] == "red"
    blocked = panel.loc[panel["forecast_entity_id"] == 20]
    assert not blocked["ral_apply_ral_applied"].to_numpy(dtype=bool).any()


def test_run_governance_workflow_df_injected_forged_allow_cannot_bypass_snap() -> None:
    df = _build_sample_panel_df()
    decisions = pd.DataFrame(
        {
            "site_id": [1, 2],
            "forecast_entity_id": [10, 20],
            "ral_policy": ["allow", "allow"],
            "status": ["green", "green"],
            "fas_class": ["ALLOWED", "ALLOWED"],
            "dqc_class": ["continuous_like", "continuous_like"],
            "snap_required": [False, False],
            "snap_unit": [np.nan, np.nan],
        }
    )
    _panel, out_decisions = run_governance_workflow_df(
        df=df,
        keys=["site_id", "forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=2.0,
        decisions_df=decisions,
    )
    row_q = out_decisions[out_decisions["forecast_entity_id"] == 20].iloc[0]
    assert str(row_q["dqc_class"]).lower() in {"quantized", "piecewise_packed"}
    assert bool(row_q["snap_required"]) is True
    snap_unit = float(str(row_q["snap_unit"]))
    assert np.isfinite(snap_unit) and snap_unit > 0.0

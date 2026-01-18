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
    assert {"site_id", "forecast_entity_id", "recommended_mode", "snap_required"}.issubset(
        decisions.columns
    )

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

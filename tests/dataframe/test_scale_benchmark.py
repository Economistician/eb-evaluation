"""
High-scale panel benchmark: 7,000 nodes × 15 series × 28 days.

The production grain for a rolling 28-day window is *daily aggregated
summaries* (one row per node × series × day), not the raw intra-day
fact table. ``evaluate_groups_df`` is then grouped by
``["node_id", "series_id"]`` so each of the 105,000 streams gets one
CWSL / NSL / UD / FRS row over the window.

A full intra-day panel at this cardinality is 141M rows and is not the
batch path; this file exercises the daily-summary shape instead.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from eb_evaluation.dataframe import evaluate_groups_df
from eb_metrics.metrics import cwsl, frs, hr_at_tau, nsl, ud

N_NODES = 7_000
N_SERIES = 15
N_DAYS = 28
N_STREAMS = N_NODES * N_SERIES  # 105,000
N_ROWS = N_STREAMS * N_DAYS  # 2,940,000
GROUP_COLS = ["node_id", "series_id"]
CWSL_MAX = 0.30

# Workstation (2026-08-22): primitives ~0.06 s, multi-key groups ~0.35 s.
PRIMITIVE_BUDGET_S = 1.0
GROUPS_BUDGET_S = 1.0
INPUT_BUDGET_MB = 128.0


def _build_daily_window() -> pd.DataFrame:
    """Contiguous 28-day summaries for every node × series stream."""
    rng = np.random.default_rng(20260822)
    node_id = np.repeat(np.arange(N_NODES, dtype=np.int32), N_SERIES * N_DAYS)
    series_id = np.tile(
        np.repeat(np.arange(N_SERIES, dtype=np.int16), N_DAYS),
        N_NODES,
    )
    day = np.tile(np.arange(N_DAYS, dtype=np.int16), N_STREAMS)
    y_true = rng.random(N_ROWS, dtype=np.float64) * 20.0 + 0.5
    y_pred = rng.random(N_ROWS, dtype=np.float64) * 22.0
    return pd.DataFrame(
        {
            "node_id": node_id,
            "series_id": series_id,
            "day": day,
            "actual_qty": y_true,
            "forecast_qty": y_pred,
        }
    )


@pytest.mark.scale
def test_daily_window_groups_by_node_and_series_is_subsecond() -> None:
    df = _build_daily_window()
    input_mb = (
        df["node_id"].to_numpy().nbytes
        + df["series_id"].to_numpy().nbytes
        + df["actual_qty"].to_numpy().nbytes
        + df["forecast_qty"].to_numpy().nbytes
    ) / 1e6
    assert input_mb < INPUT_BUDGET_MB
    assert len(df) == N_ROWS
    assert df.groupby(GROUP_COLS, sort=False).ngroups == N_STREAMS

    y_true = df["actual_qty"].to_numpy(dtype=float)
    y_pred = df["forecast_qty"].to_numpy(dtype=float)

    t0 = time.perf_counter()
    cwsl_val = cwsl(y_true, y_pred, cu=2.0, co=1.0)
    nsl_val = nsl(y_true, y_pred)
    ud_val = ud(y_true, y_pred)
    frs_val = nsl_val - min(1.0, cwsl_val / CWSL_MAX)
    primitive_s = time.perf_counter() - t0

    assert np.isfinite([cwsl_val, nsl_val, ud_val, frs_val]).all()
    assert primitive_s < PRIMITIVE_BUDGET_S

    t0 = time.perf_counter()
    grouped = evaluate_groups_df(df, GROUP_COLS, cwsl_max=CWSL_MAX)
    groups_s = time.perf_counter() - t0

    assert set(GROUP_COLS).issubset(grouped.columns)
    assert len(grouped) == N_STREAMS
    assert grouped.duplicated(GROUP_COLS).sum() == 0
    assert grouped[["CWSL", "NSL", "UD", "FRS"]].notna().all().all()
    assert groups_s < GROUPS_BUDGET_S

    # One stream must match a direct eb_metrics call on that 28-day slice.
    node_id, series_id = 12, 3
    slice_df = df.loc[
        (df["node_id"] == node_id) & (df["series_id"] == series_id),
        :,
    ]
    y_s = slice_df["actual_qty"].to_numpy(dtype=float)
    yhat_s = slice_df["forecast_qty"].to_numpy(dtype=float)
    assert len(y_s) == N_DAYS

    row = grouped.loc[
        (grouped["node_id"] == node_id) & (grouped["series_id"] == series_id)
    ].iloc[0]
    expected = {
        "CWSL": cwsl(y_s, yhat_s, cu=2.0, co=1.0),
        "NSL": nsl(y_s, yhat_s),
        "UD": ud(y_s, yhat_s),
        "HR@tau": hr_at_tau(y_s, yhat_s, tau=2.0),
        "FRS": frs(y_s, yhat_s, cu=2.0, co=1.0, cwsl_max=CWSL_MAX),
    }
    for name, value in expected.items():
        assert np.isclose(row[name], value, rtol=1e-12, atol=1e-12), name

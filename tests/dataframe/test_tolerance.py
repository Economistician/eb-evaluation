# tests/dataframe/test_tolerance.py

import math

import numpy as np
import pandas as pd
import pytest

from eb_evaluation.dataframe.tolerance import (
    TauEstimate,
    estimate_entity_tau,
    estimate_tau,
    hr_at_tau,
    hr_auto_tau,
)


def test_hr_at_tau_basic():
    y = np.array([10, 10, 10, 10], dtype=float)
    yhat = np.array([10, 11, 9, 13], dtype=float)  # abs errors: [0,1,1,3]

    assert hr_at_tau(y, yhat, tau=0) == 0.25
    assert hr_at_tau(y, yhat, tau=1) == 0.75
    assert hr_at_tau(y, yhat, tau=3) == 1.0


def test_hr_at_tau_matches_eb_metrics_on_finite_data():
    from eb_metrics.metrics.service import hr_at_tau as core

    y = np.array([10, 10, 10, 10], dtype=float)
    yhat = np.array([10, 11, 9, 13], dtype=float)

    for tau in [0.0, 1.0, 3.0]:
        assert hr_at_tau(y, yhat, tau=tau) == pytest.approx(
            core(y_true=y, y_pred=yhat, tau=tau)
        )


def test_hr_at_tau_ignores_nans():
    y = np.array([10, np.nan, 10, 10], dtype=float)
    yhat = np.array([9, 10, np.nan, 11], dtype=float)
    # finite pairs are indices 0 and 3: abs errors [1,1]
    assert hr_at_tau(y, yhat, tau=0) == 0.0
    assert hr_at_tau(y, yhat, tau=1) == 1.0


def test_estimate_tau_target_hit_rate_quantile():
    # abs errors: [0,1,1,3]
    y = np.array([10, 10, 10, 10], dtype=float)
    yhat = np.array([10, 11, 9, 13], dtype=float)

    abs_errors = np.array([0, 1, 1, 3], dtype=float)
    expected_tau = float(np.quantile(abs_errors, 0.75))  # match numpy behavior in this environment

    est = estimate_tau(y, yhat, method="target_hit_rate", target_hit_rate=0.75)
    assert isinstance(est, TauEstimate)
    assert est.method == "target_hit_rate"
    assert est.n == 4
    assert est.tau == pytest.approx(expected_tau)
    assert est.diagnostics["target_hit_rate"] == pytest.approx(0.75)

    # Achieved HR on calibration should equal fraction <= tau_used
    expected_hr = float(np.mean(abs_errors <= expected_tau))
    assert est.diagnostics["achieved_hr_calibration"] == pytest.approx(expected_hr)


def test_estimate_tau_target_hit_rate_respects_floor_and_cap():
    # abs errors: [0,1,1,3]
    y = np.array([10, 10, 10, 10], dtype=float)
    yhat = np.array([10, 11, 9, 13], dtype=float)

    est_floor = estimate_tau(y, yhat, method="target_hit_rate", target_hit_rate=0.75, tau_floor=2.0)
    assert est_floor.tau == pytest.approx(2.0)

    est_cap = estimate_tau(y, yhat, method="target_hit_rate", target_hit_rate=1.0, tau_cap=2.0)
    # q(1.0) is max error (3), but cap forces 2
    assert est_cap.tau == pytest.approx(2.0)


def test_estimate_tau_knee_returns_valid_tau_and_is_bounded():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    yhat = y + rng.normal(0, 0.5, 500)

    est = estimate_tau(
        y,
        yhat,
        method="knee",
        grid_size=51,
        grid_quantiles=(0.0, 0.99),
        knee_rule="slope_threshold",
        slope_threshold=0.01,
    )

    assert est.method == "knee"
    assert est.n > 0
    assert np.isfinite(est.tau)
    assert est.tau >= 0
    assert est.diagnostics["tau_grid_min"] <= est.tau <= est.diagnostics["tau_grid_max"]


def test_estimate_tau_utility_penalizes_large_tau():
    # errors: [0,1,2,3,4]
    y = np.zeros(5, dtype=float)
    yhat = np.array([0, 1, 2, 3, 4], dtype=float)
    grid = np.array([0, 1, 2, 3, 4], dtype=float)

    est_low_penalty = estimate_tau(
        y,
        yhat,
        method="utility",
        grid=grid,
        lambda_=0.0,  # no penalty => should pick max tau (best HR)
        tau_max=4.0,
    )
    assert est_low_penalty.tau == pytest.approx(4.0)

    est_high_penalty = estimate_tau(
        y,
        yhat,
        method="utility",
        grid=grid,
        lambda_=10.0,  # strong penalty => should prefer small tau
        tau_max=4.0,
    )
    assert est_high_penalty.tau == pytest.approx(0.0)


def test_estimate_tau_no_finite_pairs():
    y = np.array([np.nan, np.nan], dtype=float)
    yhat = np.array([1.0, np.nan], dtype=float)

    est = estimate_tau(y, yhat, method="target_hit_rate", target_hit_rate=0.9)
    assert math.isnan(est.tau)
    assert est.n == 0
    assert est.diagnostics.get("reason") == "no_finite_pairs"


def test_estimate_entity_tau_basic_and_min_n():
    df = pd.DataFrame(
        {
            "entity": ["A"] * 5 + ["B"] * 2,
            "y":      [10, 10, 10, 10, 10,  10, 10],
            "yhat":   [10, 11,  9, 13, 10,  10, 12],  # A errors [0,1,1,3,0], B errors [0,2]
        }
    )

    out = estimate_entity_tau(
        df,
        entity_col="entity",
        y_col="y",
        yhat_col="yhat",
        method="target_hit_rate",
        min_n=3,
        estimate_kwargs={"target_hit_rate": 0.8},
        include_diagnostics=True,
    )

    assert set(out["entity"]) == {"A", "B"}

    row_a = out[out["entity"] == "A"].iloc[0]
    row_b = out[out["entity"] == "B"].iloc[0]

    assert np.isfinite(row_a["tau"])
    assert int(row_a["n"]) == 5
    assert row_a["method"] == "target_hit_rate"
    assert "diagnostics" in row_a

    assert np.isnan(row_b["tau"])
    assert int(row_b["n"]) == 2
    assert row_b["reason"].startswith("min_n_not_met")


def test_estimate_entity_tau_global_cap():
    rng = np.random.default_rng(123)

    n_a = 200
    n_b = 200

    df = pd.DataFrame(
        {
            "entity": ["A"] * n_a + ["B"] * n_b,
            "y": np.concatenate([np.zeros(n_a), np.zeros(n_b)]),
            "yhat": np.concatenate([rng.normal(0, 10, n_a), rng.normal(0, 0.5, n_b)]),
        }
    )

    out_uncapped = estimate_entity_tau(
        df,
        entity_col="entity",
        y_col="y",
        yhat_col="yhat",
        method="target_hit_rate",
        min_n=30,
        estimate_kwargs={"target_hit_rate": 0.95},
        cap_with_global=False,
        include_diagnostics=False,
    )

    out_capped = estimate_entity_tau(
        df,
        entity_col="entity",
        y_col="y",
        yhat_col="yhat",
        method="target_hit_rate",
        min_n=30,
        estimate_kwargs={"target_hit_rate": 0.95},
        cap_with_global=True,
        global_cap_quantile=0.90,
        include_diagnostics=False,
    )

    tau_a_uncapped = float(out_uncapped.loc[out_uncapped["entity"] == "A", "tau"].iloc[0])
    tau_a_capped = float(out_capped.loc[out_capped["entity"] == "A", "tau"].iloc[0])
    global_cap = float(out_capped.loc[out_capped["entity"] == "A", "global_cap_tau"].iloc[0])

    assert np.isfinite(tau_a_uncapped)
    assert np.isfinite(tau_a_capped)
    # Can't use <= pytest.approx; instead allow a tiny epsilon
    assert tau_a_capped <= global_cap + 1e-12
    assert tau_a_capped <= tau_a_uncapped + 1e-12


def test_hr_auto_tau_returns_hr_tau_and_diagnostics():
    y = np.array([10, 10, 10, 10], dtype=float)
    yhat = np.array([10, 11, 9, 13], dtype=float)

    abs_errors = np.array([0, 1, 1, 3], dtype=float)
    expected_tau = float(np.quantile(abs_errors, 0.75))
    expected_hr = float(np.mean(abs_errors <= expected_tau))

    hr, tau, diag = hr_auto_tau(y, yhat, method="target_hit_rate", target_hit_rate=0.75)
    assert tau == pytest.approx(expected_tau)
    assert hr == pytest.approx(expected_hr)
    assert isinstance(diag, dict)
    assert diag["target_hit_rate"] == pytest.approx(0.75)


def test_estimate_tau_rejects_invalid_inputs():
    y = np.array([1.0, 2.0], dtype=float)
    yhat = np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError):
        estimate_tau(y, yhat, method="target_hit_rate", target_hit_rate=0.0)

    with pytest.raises(ValueError):
        hr_at_tau(y, yhat, tau=-1.0)

    with pytest.raises(ValueError):
        estimate_tau(y, yhat, method="knee", grid_quantiles=(0.9, 0.1))

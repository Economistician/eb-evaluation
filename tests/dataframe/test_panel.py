from __future__ import annotations

import numpy as np
import pandas as pd

from eb_evaluation.dataframe import evaluate_hierarchy_df, evaluate_panel_df


def _build_sample_df() -> pd.DataFrame:
    """
    Small helper to build a simple hierarchical dataset with:
      - 2 stores
      - 2 items per store
      - multiple intervals
    """
    return pd.DataFrame(
        {
            "store_id": [1, 1, 1, 2, 2, 2],
            "item_id": ["A", "A", "B", "A", "A", "B"],
            "actual_qty": [10, 12, 8, 9, 11, 7],
            "forecast_qty": [9, 15, 7, 10, 10, 8],
        }
    )


def test_evaluate_panel_df_basic_structure():
    """
    Structural test:
    - ensure the function returns a DataFrame
    - ensure it contains 'level', 'metric', and 'value' columns
    - ensure at least some group columns are preserved.
    """
    df = _build_sample_df()

    levels = {
        "overall": [],
        "by_store": ["store_id"],
        "by_store_item": ["store_id", "item_id"],
    }

    cu = 2.0
    co = 1.0
    tau = 2.0

    panel = evaluate_panel_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # Required structural columns
    assert "level" in panel.columns
    assert "metric" in panel.columns
    assert "value" in panel.columns

    # Should contain some group columns from levels
    # (for this setup we expect store_id and item_id to appear)
    assert "store_id" in panel.columns
    assert "item_id" in panel.columns

    # There should be rows for multiple levels
    assert set(panel["level"]) == {"overall", "by_store", "by_store_item"}


def test_evaluate_panel_df_matches_hierarchy_values():
    """
    Correctness test:
    The long-form panel produced by evaluate_panel_df should be
    consistent with the wide-form metrics returned by evaluate_hierarchy_df.

    For each level/group/metric, the value in the panel should equal the
    corresponding entry in the hierarchy output.
    """
    df = _build_sample_df()

    levels = {
        "overall": [],
        "by_store": ["store_id"],
        "by_store_item": ["store_id", "item_id"],
    }

    cu = 2.0
    co = 1.0
    tau = 2.0

    # Wide form per level
    hier = evaluate_hierarchy_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # Long-form panel
    panel = evaluate_panel_df(
        df=df,
        levels=levels,
        actual_col="actual_qty",
        forecast_col="forecast_qty",
        cu=cu,
        co=co,
        tau=tau,
    )

    # Metrics that panel is expected to melt
    metric_cols = [
        "n_intervals",
        "total_demand",
        "cwsl",
        "nsl",
        "ud",
        "wmape",
        "hr_at_tau",
        "frs",
    ]

    for level_name, group_cols in levels.items():
        wide = hier[level_name]

        # For this level, filter the panel rows
        panel_level = panel[panel["level"] == level_name]

        for _, wide_row in wide.iterrows():
            # Build a mask in the panel for this group's coordinates
            mask = panel_level["level"] == level_name  # redundant but clear
            for col in group_cols:
                mask &= panel_level[col] == wide_row[col]

            group_panel = panel_level[mask]

            for metric in metric_cols:
                if metric not in wide.columns:
                    continue  # e.g. hr_at_tau may be absent if tau=None

                expected_val = wide_row[metric]

                # In the panel, this metric is a row with metric name and value
                metric_rows = group_panel[group_panel["metric"] == metric]
                assert len(metric_rows) == 1, (
                    f"Expected exactly one panel row for "
                    f"level={level_name}, group={dict(wide_row[group_cols])}, "
                    f"metric={metric}"
                )

                panel_val = metric_rows["value"].iloc[0]

                # Use np.isclose for numeric metrics
                assert np.isclose(panel_val, expected_val), (
                    f"Mismatch for {metric} at level={level_name}"
                )

"""
Hierarchy-level evaluation (DataFrame utilities).

This module provides a convenience helper for evaluating forecasts at multiple levels of a
grouping hierarchy (e.g., overall, by store, by item, by store x item).

It returns a dictionary mapping each hierarchy level name to a DataFrame of metrics for that
level. Metric definitions are delegated to ``eb_metrics.metrics``; this module focuses on
grouping orchestration and tabular output suitable for reporting.

The EB metric suite here includes CWSL and related service/readiness diagnostics (NSL, UD,
HR@tau, FRS) as well as wMAPE.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .group import _compose_frs, _evaluate_groups_df

_OVERALL_SENTINEL = "_eb_hierarchy_overall"

_GROUPS_TO_HIERARCHY = {
    "CWSL": "cwsl",
    "NSL": "nsl",
    "UD": "ud",
    "wMAPE": "wmape",
    "HR@tau": "hr_at_tau",
    "FRS": "frs",
}

_GROUPS_ONLY_COLS = ("MAE", "RMSE", "MAPE")


def evaluate_hierarchy_df(
    df: pd.DataFrame,
    levels: dict[str, Sequence[str]],
    actual_col: str,
    forecast_col: str,
    cu: float | str,
    co: float | str,
    tau: float | None = None,
    *,
    cwsl_max: float,
    sample_weight_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate EB metrics at multiple hierarchy levels.

    This helper evaluates forecast performance across several grouping levels, each defined
    by a list of column names. For each level, it computes:

    - CWSL
    - NSL
    - UD
    - wMAPE
    - HR@tau (optional)
    - FRS

    where each metric is computed over the subset (group) implied by that level.

    The ``levels`` mapping accepts an empty list to represent the overall aggregate, e.g.
    ``{"overall": []}``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at minimum ``actual_col`` and ``forecast_col`` plus any
        grouping columns referenced by ``levels``.
    levels : dict[str, Sequence[str]]
        Mapping from level name to the column names used to group at that level.

        Example:

        >>> levels = {
        ...     "overall": [],
        ...     "by_store": ["store_id"],
        ...     "by_item": ["item_id"],
        ...     "by_store_item": ["store_id", "item_id"],
        ... }

        An empty sequence means evaluate the entire DataFrame as a single group.
    actual_col : str
        Column name for actual demand / realized values.
    forecast_col : str
        Column name for forecast values.
    cu : float | str
        Underbuild (shortfall) cost coefficient passed through to ``eb_metrics.metrics.cwsl``
        and ``eb_metrics.metrics.frs``.
    co : float | str
        Overbuild (excess) cost coefficient passed through to ``eb_metrics.metrics.cwsl``
        and ``eb_metrics.metrics.frs``.
    tau : float | None, default=None
        Tolerance parameter for HR@tau. If ``None``, HR@tau is omitted from outputs.
    cwsl_max : float
        Required upper bound for FRS scaling: the largest economically meaningful CWSL
        for the application. Must be finite and strictly greater than 0. There is no
        default.
    sample_weight_col : str | None, default=None
        Optional column of non-negative sample weights. If provided, weights are
        passed through to CWSL, NSL, UD, HR@tau, and FRS. ``wmape`` remains
        unweighted.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping level name to a DataFrame of metrics for that level.

        Each DataFrame includes:

        - the level's grouping columns (if any), first
        - ``n_intervals`` : number of rows evaluated in that group
        - ``total_demand`` : sum of ``actual_col`` for that group
        - ``cwsl`` : cost-weighted service loss
        - ``nsl`` : no-shortage level
        - ``ud`` : underbuild deviation
        - ``wmape`` : weighted mean absolute percentage error (per eb_metrics definition)
        - ``hr_at_tau`` : hit rate within tolerance tau (only if ``tau`` is provided)
        - ``frs`` : forecast readiness score

    Raises
    ------
    KeyError
        If required columns are missing from ``df`` (actual/forecast and any columns referenced in
        ``levels``).
    ValueError
        If ``df`` is empty, or if ``levels`` is empty.

    Notes
    -----
    - Grouped levels are reduced through ``evaluate_groups_df`` (panel-level masks and one
      ``groupby.sum()``), then renamed onto this function's schema. Invalid groups yield
      NaN for the failing metric, matching ``evaluate_groups_df``.
    - ``groupby(..., dropna=False)`` is used so that missing values in grouping keys form explicit
      groups, which is often desirable in operational reporting.
    """
    if df.empty:
        raise ValueError("df is empty.")
    if not levels:
        raise ValueError("levels must be a non-empty mapping of level name -> group columns.")

    # Validate FRS bound up front so an invalid cwsl_max still raises.
    _compose_frs(0.0, 0.0, cwsl_max)

    # Validate required columns (actual/forecast + all referenced group columns)
    required_cols = {actual_col, forecast_col}
    for cols in levels.values():
        required_cols.update(cols)

    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {sorted(missing)}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in df")

    tau_for_groups = float(tau) if tau is not None else 2.0
    results: dict[str, pd.DataFrame] = {}

    for level_name, group_cols in levels.items():
        group_cols = list(group_cols)

        if len(group_cols) == 0:
            results[level_name] = _evaluate_overall_level(
                df,
                actual_col=actual_col,
                forecast_col=forecast_col,
                cu=cu,
                co=co,
                tau=tau,
                tau_for_groups=tau_for_groups,
                cwsl_max=cwsl_max,
                sample_weight_col=sample_weight_col,
            )
            continue

        groups = _evaluate_groups_df(
            df,
            group_cols,
            actual_col=actual_col,
            forecast_col=forecast_col,
            cu=cu,
            co=co,
            tau=tau_for_groups,
            sample_weight_col=sample_weight_col,
            cwsl_max=cwsl_max,
            dropna=False,
        )
        extras = _group_size_and_demand(df, group_cols, actual_col, dropna=False)
        metrics = _hierarchy_metrics_from_groups(groups, tau=tau)
        level_df = extras.merge(metrics, on=group_cols, how="left")
        ordered_cols = list(group_cols) + [c for c in level_df.columns if c not in group_cols]
        results[level_name] = level_df.loc[:, ordered_cols]

    return results


def _evaluate_overall_level(
    df: pd.DataFrame,
    *,
    actual_col: str,
    forecast_col: str,
    cu: float | str,
    co: float | str,
    tau: float | None,
    tau_for_groups: float,
    cwsl_max: float,
    sample_weight_col: str | None,
) -> pd.DataFrame:
    """Evaluate the empty-group (overall) level via a one-group reducer."""
    keep = [actual_col, forecast_col]
    if isinstance(cu, str):
        keep.append(cu)
    if isinstance(co, str):
        keep.append(co)
    if sample_weight_col is not None:
        keep.append(sample_weight_col)

    work = df.loc[:, keep].copy()
    work[_OVERALL_SENTINEL] = np.int8(0)
    groups = _evaluate_groups_df(
        work,
        [_OVERALL_SENTINEL],
        actual_col=actual_col,
        forecast_col=forecast_col,
        cu=cu,
        co=co,
        tau=tau_for_groups,
        sample_weight_col=sample_weight_col,
        cwsl_max=cwsl_max,
        dropna=False,
    )
    metrics = _hierarchy_metrics_from_groups(groups, tau=tau).drop(
        columns=[_OVERALL_SENTINEL], errors="ignore"
    )
    y_true = df[actual_col].to_numpy(dtype=float)
    extras = pd.DataFrame(
        {
            "n_intervals": [len(df)],
            "total_demand": [float(np.sum(y_true))],
        }
    )
    level_df = pd.concat([extras, metrics.reset_index(drop=True)], axis=1)
    ordered = [
        "n_intervals",
        "total_demand",
        "cwsl",
        "nsl",
        "ud",
        "wmape",
        *(["hr_at_tau"] if tau is not None else []),
        "frs",
    ]
    return level_df.loc[:, ordered]


def _hierarchy_metrics_from_groups(groups: pd.DataFrame, *, tau: float | None) -> pd.DataFrame:
    """Rename group-evaluator columns onto the hierarchy schema."""
    out = groups.rename(columns=_GROUPS_TO_HIERARCHY)
    drop_cols = [c for c in _GROUPS_ONLY_COLS if c in out.columns]
    if tau is None and "hr_at_tau" in out.columns:
        drop_cols.append("hr_at_tau")
    return out.drop(columns=drop_cols)


def _group_size_and_demand(
    df: pd.DataFrame,
    group_cols: list[str],
    actual_col: str,
    *,
    dropna: bool,
) -> pd.DataFrame:
    """Row counts and numpy-style demand sums in first-appearance group order."""
    payload: dict[str, np.ndarray] = {c: df[c].to_numpy() for c in group_cols}
    payload["_y"] = df[actual_col].to_numpy(dtype=float)
    work = pd.DataFrame(payload)
    grouped = work.groupby(group_cols, dropna=dropna, sort=False)
    extras = pd.DataFrame(
        {
            "n_intervals": grouped.size(),
            "total_demand": grouped["_y"].sum(skipna=False),
        }
    )
    return extras.reset_index()

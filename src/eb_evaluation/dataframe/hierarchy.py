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

import pandas as pd

from eb_metrics.metrics import cwsl, hr_at_tau, nsl, ud, wmape

from .group import _compose_frs


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
    - Unlike ``evaluate_groups_df``, metric ``ValueError``s propagate; callers must
      catch if they want NaN-on-failure.
    - ``groupby(..., dropna=False)`` is used so that missing values in grouping keys form explicit
      groups, which is often desirable in operational reporting.
    """
    if df.empty:
        raise ValueError("df is empty.")
    if not levels:
        raise ValueError("levels must be a non-empty mapping of level name -> group columns.")

    # Validate required columns (actual/forecast + all referenced group columns)
    required_cols = {actual_col, forecast_col}
    for cols in levels.values():
        required_cols.update(cols)

    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {sorted(missing)}")

    if sample_weight_col is not None and sample_weight_col not in df.columns:
        raise KeyError(f"sample_weight_col {sample_weight_col!r} not found in df")

    results: dict[str, pd.DataFrame] = {}

    for level_name, group_cols in levels.items():
        group_cols = list(group_cols)  # normalize to list[str]

        if len(group_cols) == 0:
            # Single overall group
            y_true = df[actual_col].to_numpy(dtype=float)
            y_pred = df[forecast_col].to_numpy(dtype=float)
            sample_weight = (
                df[sample_weight_col].to_numpy(dtype=float)
                if sample_weight_col is not None
                else None
            )

            cwsl_val = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co, sample_weight=sample_weight)
            nsl_val = nsl(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
            metrics_row = {
                "n_intervals": len(df),
                "total_demand": float(y_true.sum()),
                "cwsl": cwsl_val,
                "nsl": nsl_val,
                "ud": ud(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
                "wmape": wmape(y_true=y_true, y_pred=y_pred),
            }
            if tau is not None:
                metrics_row["hr_at_tau"] = hr_at_tau(
                    y_true=y_true, y_pred=y_pred, tau=tau, sample_weight=sample_weight
                )

            metrics_row["frs"] = _compose_frs(nsl_val, cwsl_val, cwsl_max)

            results[level_name] = pd.DataFrame([metrics_row])
            continue

        # Grouped evaluation
        group_rows: list[dict] = []

        grouped = df.groupby(group_cols, dropna=False, sort=False)
        for keys, df_g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)

            y_true = df_g[actual_col].to_numpy(dtype=float)
            y_pred = df_g[forecast_col].to_numpy(dtype=float)
            sample_weight = (
                df_g[sample_weight_col].to_numpy(dtype=float)
                if sample_weight_col is not None
                else None
            )

            cwsl_val = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co, sample_weight=sample_weight)
            nsl_val = nsl(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
            row = {
                "n_intervals": len(df_g),
                "total_demand": float(y_true.sum()),
                "cwsl": cwsl_val,
                "nsl": nsl_val,
                "ud": ud(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
                "wmape": wmape(y_true=y_true, y_pred=y_pred),
            }
            if tau is not None:
                row["hr_at_tau"] = hr_at_tau(
                    y_true=y_true, y_pred=y_pred, tau=tau, sample_weight=sample_weight
                )

            row["frs"] = _compose_frs(nsl_val, cwsl_val, cwsl_max)

            # Attach grouping keys (ensure values are treated as scalars for type-checkers)
            for col, value in zip(group_cols, keys, strict=False):
                row[col] = value

            group_rows.append(row)

        level_df = pd.DataFrame(group_rows)

        # Put group columns first for readability.
        # Use `loc` to keep the return type a DataFrame for type checkers.
        ordered_cols = list(group_cols) + [c for c in level_df.columns if c not in group_cols]
        results[level_name] = level_df.loc[:, ordered_cols]

    return results

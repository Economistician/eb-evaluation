from __future__ import annotations

from typing import Dict, Sequence, List

import pandas as pd

from ebmetrics.metrics import cwsl, nsl, ud, wmape, hr_at_tau, frs


def evaluate_hierarchy_df(
    df: pd.DataFrame,
    levels: Dict[str, Sequence[str]],
    actual_col: str,
    forecast_col: str,
    cu,
    co,
    tau: float | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate CWSL and related diagnostics at multiple grouping levels.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the actual and forecast columns,
        plus any grouping columns referenced in ``levels``.

    levels : dict[str, Sequence[str]]
        Mapping of level name -> list/tuple of column names to group by.

        Examples
        --------
        levels = {
            "overall": [],
            "by_store": ["store_id"],
            "by_item": ["item_id"],
            "by_store_item": ["store_id", "item_id"],
        }

        An empty list means "treat the entire DataFrame as one group".

    actual_col : str
        Column name for actual demand.

    forecast_col : str
        Column name for forecasted demand.

    cu : float or array-like
        Underbuild cost parameter passed to ``cwsl``.

    co : float or array-like
        Overbuild cost parameter passed to ``cwsl``.

    tau : float, optional
        Tolerance passed to ``hr_at_tau``. If None, HR@τ is omitted.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping level name -> DataFrame of metrics for that level.

        Each DataFrame includes:
            - any grouping columns for that level
            - n_intervals
            - total_demand
            - cwsl
            - nsl
            - ud
            - wmape
            - hr_at_tau (if tau is not None)
            - frs
    """
    results: Dict[str, pd.DataFrame] = {}

    # Ensure required columns exist up front
    required_cols = {actual_col, forecast_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    for level_name, group_cols in levels.items():
        group_cols = list(group_cols)  # normalize

        if len(group_cols) == 0:
            # Single overall group
            y_true = df[actual_col].to_numpy(dtype=float)
            y_pred = df[forecast_col].to_numpy(dtype=float)

            metrics_row = {
                "n_intervals": len(df),
                "total_demand": float(df[actual_col].sum()),
                "cwsl": cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co),
                "nsl": nsl(y_true=y_true, y_pred=y_pred),
                "ud": ud(y_true=y_true, y_pred=y_pred),
                "wmape": wmape(y_true=y_true, y_pred=y_pred),
            }
            if tau is not None:
                metrics_row["hr_at_tau"] = hr_at_tau(
                    y_true=y_true,
                    y_pred=y_pred,
                    tau=tau,
                )
            metrics_row["frs"] = frs(
                y_true=y_true,
                y_pred=y_pred,
                cu=cu,
                co=co,
            )

            overall_df = pd.DataFrame([metrics_row])
            results[level_name] = overall_df

        else:
            # Grouped evaluation
            group_rows: List[dict] = []

            grouped = df.groupby(group_cols, dropna=False, sort=False)
            for keys, df_g in grouped:
                # keys is a scalar or tuple of scalars depending on number of group_cols
                if not isinstance(keys, tuple):
                    keys = (keys,)

                y_true = df_g[actual_col].to_numpy(dtype=float)
                y_pred = df_g[forecast_col].to_numpy(dtype=float)

                row = {
                    "n_intervals": len(df_g),
                    "total_demand": float(df_g[actual_col].sum()),
                    "cwsl": cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co),
                    "nsl": nsl(y_true=y_true, y_pred=y_pred),
                    "ud": ud(y_true=y_true, y_pred=y_pred),
                    "wmape": wmape(y_true=y_true, y_pred=y_pred),
                }
                if tau is not None:
                    row["hr_at_tau"] = hr_at_tau(
                        y_true=y_true,
                        y_pred=y_pred,
                        tau=tau,
                    )
                row["frs"] = frs(
                    y_true=y_true,
                    y_pred=y_pred,
                    cu=cu,
                    co=co,
                )

                # Attach grouping keys
                for col, value in zip(group_cols, keys):
                    row[col] = value

                group_rows.append(row)

            level_df = pd.DataFrame(group_rows)
            # Put group columns first
            results[level_name] = level_df[
                list(group_cols)
                + [c for c in level_df.columns if c not in group_cols]
            ]

    return results
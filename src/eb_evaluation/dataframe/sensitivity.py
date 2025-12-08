from __future__ import annotations

from typing import Dict, Sequence, Union, Optional, List

import numpy as np
import pandas as pd

from ebmetrics.metrics import cwsl_sensitivity


def compute_cwsl_sensitivity_df(
    df: pd.DataFrame,
    *,
    # Primary parameter names (parallel to other DF helpers)
    actual_col: str = "actual_qty",
    forecast_col: str = "forecast_qty",
    R_list: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: Union[float, str] = 1.0,
    group_cols: Optional[list[str]] = None,
    sample_weight_col: Optional[str] = None,
    # Backwards-compatible aliases used in tests and earlier drafts
    y_true_col: Optional[str] = None,
    y_pred_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute CWSL cost-sensitivity curves from a pandas DataFrame.

    This is a DataFrame-level wrapper around ``ebmetrics.metrics.cwsl_sensitivity``.
    It evaluates the Cost-Weighted Service Loss (CWSL) across a set of cost ratios
    R = cu / co, optionally per group, and returns a tidy long-form table that is
    convenient for plotting and diagnostic analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least the actual and forecast columns, and
        any grouping columns referenced in ``group_cols``.

    actual_col : str, default "actual_qty"
        Name of the column containing actual demand.

    forecast_col : str, default "forecast_qty"
        Name of the column containing forecasted demand.

    y_true_col : str or None, optional
        Backwards-compatible alias for ``actual_col``. If provided, it overrides
        ``actual_col``. Kept for compatibility with older code/tests.

    y_pred_col : str or None, optional
        Backwards-compatible alias for ``forecast_col``. If provided, it overrides
        ``forecast_col``.

    R_list : sequence of float, default=(0.5, 1.0, 2.0, 3.0)
        Candidate cost ratios R = cu / co to evaluate. Must contain at least
        one positive value; non-positive values are ignored by the underlying
        ``cwsl_sensitivity`` function.

    co : float or str, default 1.0
        Overbuild cost specification. Either:
        - scalar: same overbuild cost for all rows and groups, OR
        - name of a column in ``df`` containing per-row overbuild costs.

        For each R in ``R_list``, we set ``cu = R * co`` and compute CWSL.

    group_cols : list of str or None, default None
        Optional list of column names to group by (e.g. ["store_id", "item_id"]).
        If None or an empty list, the entire DataFrame is treated as a single group.

    sample_weight_col : str or None, default None
        Optional column name containing non-negative sample weights per row.
        If provided, these are passed through to ``cwsl_sensitivity`` as the
        ``sample_weight`` argument.

    Returns
    -------
    pandas.DataFrame
        Long-form table of CWSL sensitivity results. Columns include:

        - If ``group_cols`` is None or empty:
            ["R", "CWSL"]

        - Otherwise:
            group_cols + ["R", "CWSL"]

        Each row corresponds to a single (group, R) combination.
    """
    # Resolve backwards-compatible aliases
    if y_true_col is not None:
        actual_col = y_true_col
    if y_pred_col is not None:
        forecast_col = y_pred_col

    if group_cols is None:
        group_cols = []

    # Basic column validation
    required_cols = [actual_col, forecast_col]
    if isinstance(co, str):
        required_cols.append(co)
    if sample_weight_col is not None:
        required_cols.append(sample_weight_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    # Normalize R_list into a NumPy array early to catch obvious issues
    R_arr = np.asarray(R_list, dtype=float)
    if R_arr.ndim != 1 or R_arr.size == 0:
        raise ValueError("R_list must be a non-empty 1D sequence of floats.")

    results: List[Dict] = []

    # Helper: iterate over either a single global slice or grouped slices
    if len(group_cols) == 0:
        iter_groups = [((None,), df)]
    else:
        iter_groups = df.groupby(group_cols, sort=False)

    for keys, g in iter_groups:
        # Normalize group key into a tuple of values
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = g[actual_col].to_numpy(dtype=float)
        y_pred = g[forecast_col].to_numpy(dtype=float)

        # Handle overbuild cost: scalar vs column
        if isinstance(co, str):
            co_value: Union[float, np.ndarray] = g[co].to_numpy(dtype=float)
        else:
            co_value = co

        # Optional sample weights
        if sample_weight_col is not None:
            sample_weight = g[sample_weight_col].to_numpy(dtype=float)
        else:
            sample_weight = None

        # Compute sensitivity curve for this slice
        sensitivity_map = cwsl_sensitivity(
            y_true=y_true,
            y_pred=y_pred,
            R_list=R_arr,
            co=co_value,
            sample_weight=sample_weight,
        )

        # Build rows for each R
        for R_val, cwsl_val in sensitivity_map.items():
            row: Dict = {
                "R": float(R_val),
                "CWSL": float(cwsl_val),
            }

            # Attach group identifiers if applicable
            for col, value in zip(group_cols, keys):
                row[col] = value

            results.append(row)

    result_df = pd.DataFrame(results)

    # If grouped, put group columns first, then R, CWSL
    if len(group_cols) > 0:
        result_df = result_df[group_cols + ["R", "CWSL"]]
    else:
        result_df = result_df[["R", "CWSL"]]

    return result_df
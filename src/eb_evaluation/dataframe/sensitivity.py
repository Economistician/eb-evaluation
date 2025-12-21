from __future__ import annotations

"""
CWSL cost-ratio sensitivity (DataFrame utilities).

This module provides a DataFrame-oriented wrapper around
``eb_metrics.metrics.cwsl_sensitivity``.

The primary helper computes a CWSL sensitivity curve across a grid of cost ratios

$$
R = \frac{c_u}{c_o}
$$

optionally per group, and returns a tidy long-form table suitable for plotting and
diagnostic analysis.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from eb_metrics.metrics import cwsl_sensitivity


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
    r"""
    Compute CWSL sensitivity curves from a DataFrame.

    This is a DataFrame-level wrapper around ``eb_metrics.metrics.cwsl_sensitivity``.
    It evaluates Cost-Weighted Service Loss (CWSL) over a grid of cost ratios

    $$
    R = \frac{c_u}{c_o}
    $$

    and returns results in tidy long form.

    For each ratio value $R$ in ``R_list``, the underlying computation uses:

    $$
    c_u = R \cdot c_o
    $$

    where ``co`` may be a scalar (global overbuild cost) or a per-row column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least the actual and forecast columns, and any grouping
        columns referenced by ``group_cols``.
    actual_col : str, default="actual_qty"
        Name of the column containing actual demand values.
    forecast_col : str, default="forecast_qty"
        Name of the column containing forecast values.
    R_list : Sequence[float], default=(0.5, 1.0, 2.0, 3.0)
        Candidate cost ratios $R$ to evaluate. Must be a non-empty 1D sequence.
        (The underlying function may ignore non-positive values.)
    co : float | str, default=1.0
        Overbuild cost specification.

        - If ``float``: constant $c_o$ applied to all rows and groups.
        - If ``str``: name of a column in ``df`` containing per-row $c_o(i)$ values.
    group_cols : list[str] | None, default=None
        Optional grouping columns. If ``None`` or empty, the entire DataFrame is treated
        as a single group.
    sample_weight_col : str | None, default=None
        Optional column name containing non-negative sample weights per row. If provided,
        weights are passed as ``sample_weight`` to the underlying sensitivity computation.
    y_true_col : str | None, optional
        Backwards-compatible alias for ``actual_col``. If provided, it overrides ``actual_col``.
    y_pred_col : str | None, optional
        Backwards-compatible alias for ``forecast_col``. If provided, it overrides ``forecast_col``.

    Returns
    -------
    pandas.DataFrame
        Long-form table of sensitivity results with columns:

        - if not grouped: ``["R", "CWSL"]``
        - if grouped: ``group_cols + ["R", "CWSL"]``

        Each row corresponds to one (group, R) pair.

    Raises
    ------
    KeyError
        If required columns are missing from ``df``.
    ValueError
        If ``R_list`` is empty or not 1D.

    Notes
    -----
    - This function preserves backwards-compatible parameter aliases (``y_true_col``, ``y_pred_col``)
      to avoid breaking older tests or scripts.
    - CWSL values are returned exactly as computed by ``eb_metrics.metrics.cwsl_sensitivity``.

    """
    # Resolve backwards-compatible aliases
    if y_true_col is not None:
        actual_col = y_true_col
    if y_pred_col is not None:
        forecast_col = y_pred_col

    group_cols = [] if group_cols is None else list(group_cols)

    # Basic column validation
    required_cols: list[str] = [actual_col, forecast_col]
    if isinstance(co, str):
        required_cols.append(co)
    if sample_weight_col is not None:
        required_cols.append(sample_weight_col)
    if group_cols:
        required_cols.extend(group_cols)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    # Normalize R_list into a NumPy array early to catch obvious issues
    R_arr = np.asarray(R_list, dtype=float)
    if R_arr.ndim != 1 or R_arr.size == 0:
        raise ValueError("R_list must be a non-empty 1D sequence of floats.")

    results: List[Dict] = []

    # Iterate over either a single global slice or grouped slices
    if len(group_cols) == 0:
        iter_groups = [((None,), df)]
    else:
        iter_groups = df.groupby(group_cols, dropna=False, sort=False)

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
        sample_weight = (
            g[sample_weight_col].to_numpy(dtype=float)
            if sample_weight_col is not None
            else None
        )

        sensitivity_map = cwsl_sensitivity(
            y_true=y_true,
            y_pred=y_pred,
            R_list=R_arr,
            co=co_value,
            sample_weight=sample_weight,
        )

        for R_val, cwsl_val in sensitivity_map.items():
            row: Dict = {"R": float(R_val), "CWSL": float(cwsl_val)}

            for col, value in zip(group_cols, keys):
                row[col] = value

            results.append(row)

    result_df = pd.DataFrame(results)

    # Column order
    if len(group_cols) > 0:
        result_df = result_df[group_cols + ["R", "CWSL"]]
    else:
        result_df = result_df[["R", "CWSL"]]

    return result_df

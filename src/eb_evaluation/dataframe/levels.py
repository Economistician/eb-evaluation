"""
Standard hierarchy level definitions for panel-style evaluation.

This module provides small, explicit helpers for constructing the ``levels``
mapping used by panel-style evaluation utilities such as
``evaluate_panel_df`` and ``evaluate_hierarchy_df``.

The goal is not to introduce new logic, but to:

- make the intended aggregation grain explicit,
- reduce ad hoc dictionary construction in notebooks and pipelines,
- avoid misinterpretation of NaN grouping keys in long-form panel outputs.

Notes
-----
Panel evaluation returns a *union* of grouping key columns across all levels.
For levels that do not use a given key, the corresponding column will contain
NaN. This is expected and intentional.

These helpers exist to make that behavior obvious and standardized.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final


def levels_overall() -> dict[str, Sequence[str]]:
    """
    Single overall aggregation level.

    Returns
    -------
    dict
        {"overall": []}
    """
    return {"overall": []}


def levels_by_site(site_col: str = "site_id") -> dict[str, Sequence[str]]:
    """
    Aggregate by site only.

    Parameters
    ----------
    site_col : str, default="site_id"
        Column name representing the site/store identifier.

    Returns
    -------
    dict
        {"by_site": [site_col]}
    """
    return {"by_site": [site_col]}


def levels_by_entity(entity_col: str = "forecast_entity_id") -> dict[str, Sequence[str]]:
    """
    Aggregate by forecast entity only.

    Parameters
    ----------
    entity_col : str, default="forecast_entity_id"
        Column name representing the forecast entity identifier.

    Returns
    -------
    dict
        {"by_entity": [entity_col]}
    """
    return {"by_entity": [entity_col]}


def levels_by_site_entity(
    *,
    site_col: str = "site_id",
    entity_col: str = "forecast_entity_id",
) -> dict[str, Sequence[str]]:
    """
    Aggregate by site x forecast entity.

    Note
    ----
    Use plain ASCII characters in docstrings to avoid ambiguous-character
    linting and encoding issues.

    Parameters
    ----------
    site_col : str, default="site_id"
        Column name representing the site/store identifier.
    entity_col : str, default="forecast_entity_id"
        Column name representing the forecast entity identifier.

    Returns
    -------
    dict
        {"by_site_entity": [site_col, entity_col]}
    """
    return {"by_site_entity": [site_col, entity_col]}


def levels_standard_panel(
    *,
    site_col: str = "site_id",
    entity_col: str = "forecast_entity_id",
) -> dict[str, Sequence[str]]:
    """
    Canonical multi-level hierarchy for panel evaluation.

    This is the recommended default for most panel-style analyses and
    reporting workflows.

    Levels included:
    - overall
    - by_site
    - by_entity
    - by_site_entity

    Parameters
    ----------
    site_col : str, default="site_id"
        Column name representing the site/store identifier.
    entity_col : str, default="forecast_entity_id"
        Column name representing the forecast entity identifier.

    Returns
    -------
    dict
        Mapping suitable for ``evaluate_panel_df``.
    """
    return {
        "overall": [],
        "by_site": [site_col],
        "by_entity": [entity_col],
        "by_site_entity": [site_col, entity_col],
    }


# Explicit constant for discoverability / docs
STANDARD_PANEL_LEVELS: Final[str] = "levels_standard_panel"

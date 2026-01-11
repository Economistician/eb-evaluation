from __future__ import annotations

from collections.abc import Sequence

from eb_evaluation.dataframe.levels import (
    levels_by_entity,
    levels_by_site,
    levels_by_site_entity,
    levels_overall,
    levels_standard_panel,
)


def _assert_levels_shape(levels: dict[str, Sequence[str]]) -> None:
    """
    Shared structural assertion for levels mappings.
    """
    assert isinstance(levels, dict)
    for name, cols in levels.items():
        assert isinstance(name, str)
        assert isinstance(cols, Sequence)
        for c in cols:
            assert isinstance(c, str)


def test_levels_overall() -> None:
    levels = levels_overall()

    _assert_levels_shape(levels)
    assert levels == {"overall": []}


def test_levels_by_site_default_column() -> None:
    levels = levels_by_site()

    _assert_levels_shape(levels)
    assert levels == {"by_site": ["site_id"]}


def test_levels_by_site_custom_column() -> None:
    levels = levels_by_site(site_col="store_id")

    _assert_levels_shape(levels)
    assert levels == {"by_site": ["store_id"]}


def test_levels_by_entity_default_column() -> None:
    levels = levels_by_entity()

    _assert_levels_shape(levels)
    assert levels == {"by_entity": ["forecast_entity_id"]}


def test_levels_by_entity_custom_column() -> None:
    levels = levels_by_entity(entity_col="item_id")

    _assert_levels_shape(levels)
    assert levels == {"by_entity": ["item_id"]}


def test_levels_by_site_entity_defaults() -> None:
    levels = levels_by_site_entity()

    _assert_levels_shape(levels)
    assert levels == {"by_site_entity": ["site_id", "forecast_entity_id"]}


def test_levels_by_site_entity_custom_columns() -> None:
    levels = levels_by_site_entity(site_col="store_id", entity_col="item_id")

    _assert_levels_shape(levels)
    assert levels == {"by_site_entity": ["store_id", "item_id"]}


def test_levels_standard_panel_defaults() -> None:
    levels = levels_standard_panel()

    _assert_levels_shape(levels)
    assert set(levels.keys()) == {
        "overall",
        "by_site",
        "by_entity",
        "by_site_entity",
    }

    assert levels["overall"] == []
    assert levels["by_site"] == ["site_id"]
    assert levels["by_entity"] == ["forecast_entity_id"]
    assert levels["by_site_entity"] == ["site_id", "forecast_entity_id"]


def test_levels_standard_panel_custom_columns() -> None:
    levels = levels_standard_panel(site_col="store_id", entity_col="item_id")

    _assert_levels_shape(levels)
    assert levels["by_site"] == ["store_id"]
    assert levels["by_entity"] == ["item_id"]
    assert levels["by_site_entity"] == ["store_id", "item_id"]

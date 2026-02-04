"""Tests for visualization spec templates."""

import pytest

from sqlsaber_viz.templates import (
    ChartType,
    chart_template,
    list_chart_types,
    vizspec_template,
)


@pytest.mark.parametrize(
    "chart_type",
    ["bar", "line", "scatter", "boxplot", "histogram"],
)
def test_chart_template_returns_valid_structure(chart_type: ChartType):
    """Each chart type should return a dict with 'type' key."""
    result = chart_template(chart_type)
    assert isinstance(result, dict)
    assert result["type"] == chart_type


def test_chart_template_bar_has_encoding():
    result = chart_template("bar")
    assert "encoding" in result
    assert "x" in result["encoding"]
    assert "y" in result["encoding"]


def test_chart_template_histogram_has_histogram_config():
    result = chart_template("histogram")
    assert "histogram" in result
    assert "field" in result["histogram"]


def test_chart_template_invalid_type_raises():
    with pytest.raises(ValueError, match="Unknown chart type"):
        chart_template("invalid")  # type: ignore


@pytest.mark.parametrize(
    "chart_type",
    ["bar", "line", "scatter", "boxplot", "histogram"],
)
def test_vizspec_template_returns_complete_spec(chart_type: ChartType):
    """VizSpec template should include all required top-level fields."""
    result = vizspec_template(chart_type, "result_abc123.json")
    assert result["version"] == "1"
    assert result["data"]["source"]["file"] == "result_abc123.json"
    assert result["chart"]["type"] == chart_type
    assert "transform" in result


def test_list_chart_types_returns_all_types():
    result = list_chart_types()
    assert len(result) == 5
    types = {item["type"] for item in result}
    assert types == {"bar", "line", "scatter", "boxplot", "histogram"}

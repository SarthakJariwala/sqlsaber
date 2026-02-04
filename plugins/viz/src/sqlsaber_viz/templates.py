"""Template builders for visualization specs.

These functions generate minimal valid templates from Pydantic models,
ensuring they stay in sync with the schema definitions.
"""

from __future__ import annotations

from typing import Literal

from .spec import (
    BarChart,
    BarEncoding,
    BoxplotChart,
    BoxplotConfig,
    ChartSpec,
    ChartOptions,
    DataConfig,
    DataSource,
    FieldEncoding,
    HistogramChart,
    HistogramConfig,
    LineChart,
    LineEncoding,
    ScatterChart,
    ScatterEncoding,
    VizSpec,
)

ChartType = Literal["bar", "line", "scatter", "boxplot", "histogram"]

# Placeholder values for template fields
_CATEGORY_PLACEHOLDER = "<category_column>"
_NUMBER_PLACEHOLDER = "<number_column>"
_TIME_PLACEHOLDER = "<time_column>"
_LABEL_PLACEHOLDER = "<label_column>"
_VALUE_PLACEHOLDER = "<value_column>"


def _build_bar_chart() -> BarChart:
    return BarChart(
        type="bar",
        encoding=BarEncoding(
            x=FieldEncoding(field=_CATEGORY_PLACEHOLDER, type="category"),
            y=FieldEncoding(field=_NUMBER_PLACEHOLDER, type="number"),
            series=None,
        ),
        orientation="vertical",
        mode="grouped",
        options=ChartOptions(),
    )


def _build_line_chart() -> LineChart:
    return LineChart(
        type="line",
        encoding=LineEncoding(
            x=FieldEncoding(field=_TIME_PLACEHOLDER, type="time"),
            y=FieldEncoding(field=_NUMBER_PLACEHOLDER, type="number"),
            series=None,
        ),
        options=ChartOptions(),
    )


def _build_scatter_chart() -> ScatterChart:
    return ScatterChart(
        type="scatter",
        encoding=ScatterEncoding(
            x=FieldEncoding(field=_NUMBER_PLACEHOLDER, type="number"),
            y=FieldEncoding(field=_NUMBER_PLACEHOLDER, type="number"),
            series=None,
        ),
        options=ChartOptions(),
    )


def _build_boxplot_chart() -> BoxplotChart:
    return BoxplotChart(
        type="boxplot",
        boxplot=BoxplotConfig(
            label_field=_LABEL_PLACEHOLDER,
            value_field=_VALUE_PLACEHOLDER,
        ),
        options=ChartOptions(),
    )


def _build_histogram_chart() -> HistogramChart:
    return HistogramChart(
        type="histogram",
        histogram=HistogramConfig(
            field=_NUMBER_PLACEHOLDER,
            bins=20,
        ),
        options=ChartOptions(),
    )


_CHART_BUILDERS: dict[ChartType, callable] = {
    "bar": _build_bar_chart,
    "line": _build_line_chart,
    "scatter": _build_scatter_chart,
    "boxplot": _build_boxplot_chart,
    "histogram": _build_histogram_chart,
}


def _build_chart(chart_type: ChartType) -> ChartSpec:
    """Build a chart object for the given type."""
    builder = _CHART_BUILDERS.get(chart_type)
    if builder is None:
        raise ValueError(f"Unknown chart type: {chart_type}")
    return builder()


def chart_template(chart_type: ChartType) -> dict:
    """Return a minimal valid chart template for the given chart type.

    The template uses placeholder field names that the model should replace
    with actual column names from the data.
    """
    return _build_chart(chart_type).model_dump(exclude_none=True)


def vizspec_template(chart_type: ChartType, file: str) -> dict:
    """Return a complete VizSpec template with data source pre-filled.

    The template includes the chart structure for the specified type
    and has placeholders for field names.
    """
    spec = VizSpec(
        version="1",
        title=None,
        description=None,
        data=DataConfig(source=DataSource(file=file)),
        chart=_build_chart(chart_type),
        transform=[],
    )

    return spec.model_dump(exclude_none=True)


def list_chart_types() -> list[dict]:
    """Return available chart types with descriptions.

    Helps the model choose the appropriate chart type for the data.
    """
    return [
        {
            "type": "bar",
            "description": "Compare categories. Use x for category, y for numeric value.",
            "use_when": "Comparing values across categories (e.g., sales by region)",
        },
        {
            "type": "line",
            "description": "Show trends over time/sequence. Use x for time/sequence, y for value.",
            "use_when": "Showing change over time (e.g., monthly revenue)",
        },
        {
            "type": "scatter",
            "description": "Show correlation between two numeric variables.",
            "use_when": "Exploring relationship between two numbers (e.g., age vs income)",
        },
        {
            "type": "boxplot",
            "description": "Show distribution of values across groups.",
            "use_when": "Comparing distributions (e.g., salary by department)",
        },
        {
            "type": "histogram",
            "description": "Show distribution of a single numeric variable.",
            "use_when": "Understanding value distribution (e.g., age distribution)",
        },
    ]

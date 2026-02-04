"""Spec model validation tests."""

from sqlsaber_viz.spec import VizSpec


def test_viz_spec_bar_validates() -> None:
    spec = {
        "version": "1",
        "title": "Revenue",
        "data": {"source": {"file": "result_abc123.json"}},
        "chart": {
            "type": "bar",
            "encoding": {
                "x": {"field": "product", "type": "category"},
                "y": {"field": "revenue", "type": "number"},
            },
            "options": {"color": "blue+"},
        },
        "transform": [{"limit": 10}],
    }

    parsed = VizSpec.model_validate(spec)
    assert parsed.chart.type == "bar"
    assert parsed.data.source.file == "result_abc123.json"


def test_viz_spec_scatter_series_validates() -> None:
    spec = {
        "version": "1",
        "data": {"source": {"file": "result_series.json"}},
        "chart": {
            "type": "scatter",
            "encoding": {
                "x": {"field": "x", "type": "number"},
                "y": {"field": "y", "type": "number"},
                "series": {"field": "group", "type": "category"},
            },
        },
    }

    parsed = VizSpec.model_validate(spec)
    assert parsed.chart.type == "scatter"
    assert parsed.chart.encoding.series is not None


def test_viz_spec_bar_series_mode_validates() -> None:
    spec = {
        "version": "1",
        "data": {"source": {"file": "result_series.json"}},
        "chart": {
            "type": "bar",
            "encoding": {
                "x": {"field": "year", "type": "category"},
                "y": {"field": "count", "type": "number"},
                "series": {"field": "status", "type": "category"},
            },
            "mode": "stacked",
        },
    }

    parsed = VizSpec.model_validate(spec)
    assert parsed.chart.type == "bar"
    assert parsed.chart.mode == "stacked"

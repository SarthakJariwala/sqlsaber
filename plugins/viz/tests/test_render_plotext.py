"""Plotext renderer tests."""

from sqlsaber_viz.renderers.plotext_renderer import PlotextRenderer
from sqlsaber_viz.spec import VizSpec


def test_plotext_renderer_bar_outputs_string() -> None:
    spec = VizSpec.model_validate(
        {
            "version": "1",
            "title": "Test",
            "data": {"source": {"file": "result_abc.json"}},
            "chart": {
                "type": "bar",
                "encoding": {
                    "x": {"field": "category", "type": "category"},
                    "y": {"field": "value", "type": "number"},
                },
            },
        }
    )
    rows = [
        {"category": "A", "value": 1},
        {"category": "B", "value": 2},
    ]

    renderer = PlotextRenderer()
    output = renderer.render(spec, rows)
    assert isinstance(output, str)
    assert output.strip()


def test_plotext_renderer_grouped_bar_outputs_string() -> None:
    spec = VizSpec.model_validate(
        {
            "version": "1",
            "title": "Grouped",
            "data": {"source": {"file": "result_series.json"}},
            "chart": {
                "type": "bar",
                "encoding": {
                    "x": {"field": "year", "type": "category"},
                    "y": {"field": "count", "type": "number"},
                    "series": {"field": "status", "type": "category"},
                },
                "mode": "grouped",
            },
        }
    )
    rows = [
        {"year": "2023", "status": "A", "count": 4},
        {"year": "2023", "status": "B", "count": 2},
        {"year": "2024", "status": "A", "count": 6},
        {"year": "2024", "status": "B", "count": 1},
    ]

    renderer = PlotextRenderer()
    output = renderer.render(spec, rows)
    assert isinstance(output, str)
    assert output.strip()


def test_plotext_renderer_time_only_strings_render() -> None:
    spec = VizSpec.model_validate(
        {
            "version": "1",
            "title": "Time Line",
            "data": {"source": {"file": "result_time.json"}},
            "chart": {
                "type": "line",
                "encoding": {
                    "x": {"field": "time_of_day", "type": "time"},
                    "y": {"field": "value", "type": "number"},
                },
            },
        }
    )
    rows = [
        {"time_of_day": "09:30:00", "value": 1},
        {"time_of_day": "10:45:15", "value": 2},
    ]

    renderer = PlotextRenderer()
    output = renderer.render(spec, rows)
    assert isinstance(output, str)
    assert output.strip()
    assert "[No data" not in output

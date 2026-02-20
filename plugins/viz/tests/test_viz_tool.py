"""VizTool integration tests."""

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

import sqlsaber_viz.tools as tools
from sqlsaber.overrides import ModelOverides, build_tool_run_deps
from sqlsaber_viz.spec import VizSpec
from sqlsaber_viz.tools import VizTool


def _make_ctx(
    payload: dict, tool_call_id: str, deps: object | None = None
) -> SimpleNamespace:
    part = SimpleNamespace(
        part_kind="tool-return",
        tool_call_id=tool_call_id,
        content=payload,
    )
    msg = SimpleNamespace(parts=[part])
    return SimpleNamespace(messages=[msg], deps=deps)


class DummyAgent:
    last_model_name: str | None = None
    last_api_key: str | None = None

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        type(self).last_model_name = model_name
        type(self).last_api_key = api_key

    async def generate_spec(
        self,
        request: str,
        columns: list[dict],
        row_count: int,
        file: str,
        chart_type_hint: str | None = None,
    ) -> VizSpec:
        _ = request, columns, row_count, chart_type_hint
        spec = {
            "version": "1",
            "data": {"source": {"file": file}},
            "chart": {
                "type": "bar",
                "encoding": {
                    "x": {"field": "name", "type": "category"},
                    "y": {"field": "value", "type": "number"},
                },
            },
        }
        return VizSpec.model_validate(spec)


@pytest.mark.asyncio
async def test_viz_tool_execute_adds_bar_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "_get_spec_agent_cls", lambda: DummyAgent)

    payload = {
        "row_count": 25,
        "results": [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2},
        ],
    }
    ctx = _make_ctx(payload, "call-1")

    tool = VizTool()
    result = await tool.execute(ctx, request="show values", file="result_call-1.json")
    parsed = json.loads(result)

    assert parsed["chart"]["type"] == "bar"
    transforms = parsed.get("transform", [])
    assert any("sort" in t for t in transforms)
    assert any("limit" in t for t in transforms)


def test_viz_tool_render_result(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = VizTool()
    spec = {
        "version": "1",
        "data": {"source": {"file": "result_call-1.json"}},
        "chart": {
            "type": "bar",
            "encoding": {
                "x": {"field": "name", "type": "category"},
                "y": {"field": "value", "type": "number"},
            },
        },
    }

    tool._last_rows = [
        {"name": "A", "value": 1},
        {"name": "B", "value": 2},
    ]
    tool._last_file = "result_call-1.json"

    from sqlsaber_viz.renderers import plotext_renderer

    monkeypatch.setattr(
        plotext_renderer.PlotextRenderer, "render", lambda self, spec, rows: "chart"
    )

    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=80)
    rendered = tool.render_result(console, spec)

    assert rendered is True
    assert buffer.getvalue().strip()


@pytest.mark.asyncio
async def test_viz_tool_uses_deps_model_overide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "_get_spec_agent_cls", lambda: DummyAgent)

    payload = {
        "row_count": 2,
        "results": [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2},
        ],
    }
    ctx = _make_ctx(
        payload,
        "call-2",
        deps=build_tool_run_deps(
            {
                "viz": ModelOverides(
                    model_name="openai:gpt-5-mini", api_key="override-api-key"
                )
            }
        ),
    )
    tool = VizTool()

    result = await tool.execute(ctx, request="show values", file="result_call-2.json")

    parsed = json.loads(result)
    assert parsed["chart"]["type"] == "bar"
    assert DummyAgent.last_model_name == "openai:gpt-5-mini"
    assert DummyAgent.last_api_key == "override-api-key"

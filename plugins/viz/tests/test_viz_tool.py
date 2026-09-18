"""VizTool integration tests."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

import sqlsaber_viz.tools as tools
from sqlsaber.query_results import InMemoryQueryResultStore
from sqlsaber.render.blocks import Ansi
from sqlsaber_viz.config import VizConfig
from sqlsaber_viz.spec import VizSpec
from sqlsaber_viz.tools import VizTool


def _tool(**context_fields: Any) -> VizTool:
    fields = {
        "query_result_store": InMemoryQueryResultStore(),
        "resolve_subagent_model": lambda configured, *, tool: SimpleNamespace(
            model="inherited-model"
        ),
    }
    fields.update(context_fields)
    return VizTool(cast(Any, SimpleNamespace(**fields)))


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
    last_model: object = None

    def __init__(self, model):
        type(self).last_model = model

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

    tool = _tool()
    result = await tool.execute(ctx, request="show values", file="result_call-1.json")
    parsed = json.loads(result)

    assert parsed["chart"]["type"] == "bar"
    transforms = parsed.get("transform", [])
    assert any("sort" in t for t in transforms)
    assert any("limit" in t for t in transforms)
    assert DummyAgent.last_model == "inherited-model"


def test_viz_tool_requires_context() -> None:
    with pytest.raises(TypeError, match="context"):
        VizTool()


def test_viz_tool_render_result(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool()
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

    rendered = tool.render_result(spec)

    assert rendered is not None
    assert isinstance(rendered[0], Ansi)
    assert rendered[0].text == "chart"


@pytest.mark.asyncio
async def test_viz_tool_uses_resolved_nested_model(
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
    ctx = _make_ctx(payload, "call-2")
    tool = _tool(
        resolve_subagent_model=lambda configured, *, tool: SimpleNamespace(
            model="pinned-handle"
        )
    )
    tool.config = VizConfig()

    result = await tool.execute(ctx, request="show values", file="result_call-2.json")

    parsed = json.loads(result)
    assert parsed["chart"]["type"] == "bar"
    assert DummyAgent.last_model == "pinned-handle"

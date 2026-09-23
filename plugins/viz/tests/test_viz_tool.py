"""VizTool integration tests."""

import json
from types import SimpleNamespace

import pytest

import sqlsaber_viz.tools as tools
from sqlsaber.render.blocks import Ansi
from sqlsaber.query_results import InMemoryQueryResultStore
from sqlsaber_viz.spec import VizSpec
from sqlsaber_viz.tools import VizTool


def _tool() -> VizTool:
    return VizTool(InMemoryQueryResultStore())


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
    last_model: object | None = None

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        *,
        model: object | None = None,
    ):
        type(self).last_model_name = model_name
        type(self).last_api_key = api_key
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


def test_viz_tool_requires_query_result_store() -> None:
    with pytest.raises(TypeError, match="query_result_store"):
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
async def test_standalone_viz_tool_uses_explicit_model_configuration(
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
    tool = VizTool(
        InMemoryQueryResultStore(),
        model_name="openai:gpt-5-mini",
        api_key="override-api-key",
    )

    result = await tool.execute(ctx, request="show values", file="result_call-2.json")

    parsed = json.loads(result)
    assert parsed["chart"]["type"] == "bar"
    assert DummyAgent.last_model_name == "openai:gpt-5-mini"
    assert DummyAgent.last_api_key == "override-api-key"


@pytest.mark.asyncio
async def test_viz_tool_resolves_bound_plugin_model_through_session_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "_get_spec_agent_cls", lambda: DummyAgent)
    resolved_model = object()
    calls: list[tuple[tuple, dict]] = []

    def resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return "openai:gpt-5-mini", resolved_model, "openai"

    context = SimpleNamespace(resolve_subagent_model=resolve)
    tool = VizTool(
        InMemoryQueryResultStore(),
        context=context,
        model_name="openai:gpt-5-mini",
        api_key="explicit-key",
    )
    payload = {"row_count": 1, "results": [{"name": "A", "value": 1}]}

    await tool.execute(
        _make_ctx(payload, "call-3"),
        request="show values",
        file="result_call-3.json",
    )

    assert calls == [
        (
            (),
            {"model_name": "openai:gpt-5-mini", "api_key": "explicit-key"},
        )
    ]
    assert DummyAgent.last_model is resolved_model

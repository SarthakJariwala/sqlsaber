"""Exercise typed output and semantic retries through the real agent runtime."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from sqlsaber_viz.data_loader import extract_data_summary
from sqlsaber_viz.spec import VizSpec
from sqlsaber_viz.spec_agent import MAX_RETRIES, SpecAgent


VALID = {
    "data": {"source": {"file": "result_abc.json"}},
    "chart": {
        "type": "bar",
        "encoding": {
            "x": {"field": "region", "type": "category"},
            "y": {"field": "sales", "type": "number"},
        },
    },
}
ROWS = [{"region": "West", "sales": 6}, {"region": "East", "sales": 11}]


def test_build_agent_uses_active_session_model(monkeypatch):
    import sqlsaber_viz.spec_agent as module

    model = FunctionModel(lambda messages, info: ModelResponse([]))
    auth = object()
    resolve = Mock(return_value=SimpleNamespace(model=model))
    monkeypatch.setattr(module, "resolve_model", resolve)
    monkeypatch.setattr(
        module,
        "Config",
        lambda: SimpleNamespace(auth=auth, model=SimpleNamespace(name="session-model")),
    )
    agent = SpecAgent()
    resolve.assert_called_once_with(auth, "session-model", api_key_override=None)
    assert agent.agent.model is model


async def generate(outputs, rows=None):
    calls = []

    def respond(messages, info):
        calls.append(deepcopy(messages))
        assert info.output_tools
        assert not info.function_tools  # No template-fetch round trip.
        output = outputs[min(len(calls) - 1, len(outputs) - 1)]
        return ModelResponse([ToolCallPart(info.output_tools[0].name, output)])

    agent = SpecAgent(model=FunctionModel(respond))
    rows = ROWS if rows is None else rows
    summary = extract_data_summary({"results": rows})
    spec = await agent.generate_spec(
        "sales by region",
        columns=summary["columns"],
        row_count=len(rows),
        file="result_abc.json",
        rows=rows,
    )
    return spec, calls


async def test_typed_output_succeeds_without_template_calls():
    spec, calls = await generate([VALID])
    assert isinstance(spec, VizSpec)
    assert spec.chart.type == "bar"
    assert spec.chart.encoding.y.field == "sales"
    assert len(calls) == 1


async def test_malformed_json_is_retried():
    spec, calls = await generate(['{"chart":', VALID])
    assert spec == VizSpec.model_validate(VALID)
    assert len(calls) == 2


async def test_histogram_can_filter_and_sort_by_text_fields():
    output = {
        **VALID,
        "chart": {"type": "histogram", "histogram": {"field": "sales"}},
        "transform": [
            {"filter": {"field": "region", "op": "==", "value": "West"}},
            {"sort": [{"field": "region"}]},
        ],
    }
    spec, calls = await generate([output])
    assert spec.chart.type == "histogram"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change,feedback",
    [
        ({"data": {"source": {"file": "result_other.json"}}}, "requested source"),
        ({"chart": {"type": "pie"}}, "union_tag_invalid"),
        ({"transform": [{"sort": [{"field": "missing"}]}]}, "Unknown fields"),
        (
            {"transform": [{"filter": {"field": "missing", "op": "==", "value": 1}}]},
            "Unknown fields",
        ),
        (
            {"chart": {"type": "histogram", "histogram": {"field": "region"}}},
            "No plottable values",
        ),
        (
            {"transform": [{"filter": {"field": "sales", "op": ">", "value": 99}}]},
            "No plottable values",
        ),
    ],
)
async def test_invalid_output_is_corrected_with_feedback(change, feedback):
    invalid = deepcopy(VALID)
    invalid.update(change)
    spec, calls = await generate([invalid, VALID])
    assert spec == VizSpec.model_validate(VALID)
    assert len(calls) == 2
    retries = [
        part
        for message in calls[1]
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]
    assert feedback in str(retries)
    assert any(isinstance(message, ModelResponse) for message in calls[1])


@pytest.mark.parametrize("role", ["x", "y", "series"])
async def test_unknown_encoding_fields_are_retried(role):
    invalid = deepcopy(VALID)
    invalid["chart"]["encoding"][role] = {"field": "invented"}
    spec, calls = await generate([invalid, VALID])
    assert spec == VizSpec.model_validate(VALID)
    assert len(calls) == 2
    assert "Unknown fields" in str(calls[1])


async def test_exhaustion_and_success_on_last_retry():
    invalid = {**VALID, "data": {"source": {"file": "result_wrong.json"}}}
    with pytest.raises(UnexpectedModelBehavior, match="retries"):
        await generate([invalid])
    spec, calls = await generate([invalid] * MAX_RETRIES + [VALID])
    assert spec.data.source.file == "result_abc.json"
    assert len(calls) == MAX_RETRIES + 1


@pytest.mark.parametrize("chart_type", ["line", "scatter"])
async def test_xy_values_must_occur_in_same_row(chart_type):
    invalid = deepcopy(VALID)
    invalid["chart"]["type"] = chart_type
    rows = [{"region": 1, "sales": None}, {"region": None, "sales": "12.5"}]
    _, calls = await generate([invalid, VALID], rows)
    assert len(calls) == 2
    assert "No plottable values" in str(calls[1])


@pytest.mark.parametrize(
    "chart_type", ["bar", "line", "scatter", "boxplot", "histogram"]
)
async def test_supported_charts_and_coercions(chart_type):
    output = deepcopy(VALID)
    if chart_type == "boxplot":
        output["chart"] = {
            "type": chart_type,
            "boxplot": {"label_field": "region", "value_field": "sales"},
        }
    elif chart_type == "histogram":
        output["chart"] = {"type": chart_type, "histogram": {"field": "sales"}}
    else:
        output["chart"]["type"] = chart_type
    spec, calls = await generate([output], [{"region": "2024-03", "sales": "12.5"}])
    assert spec.chart.type == chart_type
    assert len(calls) == 1


async def test_full_rows_stay_local_and_sparse_fields_are_allowed():
    # Beyond both the five-value prompt sample and fifty-row field discovery window.
    rows = [{"region": "West", "sales": None}] * 60 + [
        {"region": "private-local-value", "sales": 42, "late_column": "local-only"}
    ]
    output = deepcopy(VALID)
    output["chart"]["encoding"]["x"]["field"] = "late_column"
    _, calls = await generate([output], rows)
    assert len(calls) == 1
    assert "private-local-value" not in str(calls)
    assert "local-only" not in str(calls)

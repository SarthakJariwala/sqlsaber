"""Tests for SpecAgent self-correction retry logic."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from sqlsaber_viz.spec import VizSpec
from sqlsaber_viz.spec_agent import MAX_RETRIES, SpecAgent, _parse_json

# -- Helpers ----------------------------------------------------------------

VALID_SPEC_JSON = json.dumps(
    {
        "version": "1",
        "data": {"source": {"file": "result_abc.json"}},
        "chart": {
            "type": "bar",
            "encoding": {
                "x": {"field": "region", "type": "category"},
                "y": {"field": "sales", "type": "number"},
            },
        },
    }
)

INVALID_SPEC_JSON = json.dumps(
    {
        "version": "1",
        "data": {"source": {"file": "result_abc.json"}},
        # Missing "type" discriminator → ValidationError
        "chart": {"encoding": {"x": {"field": "region"}, "y": {"field": "sales"}}},
    }
)

COLUMNS = [
    {"name": "region", "type": "text"},
    {"name": "sales", "type": "integer"},
]


def _make_run_result(output: str, messages: list | None = None) -> SimpleNamespace:
    """Build a minimal stand-in for a pydantic-ai AgentRunResult."""
    return SimpleNamespace(
        output=output,
        all_messages=lambda: messages or [],
    )


def _patch_agent(monkeypatch: pytest.MonkeyPatch, agent: SpecAgent) -> None:
    """Prevent SpecAgent.__init__ from building a real pydantic-ai agent."""


async def _make_spec_agent(monkeypatch: pytest.MonkeyPatch) -> SpecAgent:
    """Create a SpecAgent with a stubbed-out internal agent."""
    # Bypass __init__ which calls _build_agent (needs real config/provider)
    obj = object.__new__(SpecAgent)
    obj.config = None  # type: ignore[assignment]
    obj._model_name_override = None
    obj._api_key_override = None
    obj.agent = AsyncMock()
    return obj


# -- Tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_spec_succeeds_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the agent returns valid JSON on the first try, no retry occurs."""
    agent = await _make_spec_agent(monkeypatch)
    agent.agent.run = AsyncMock(
        return_value=_make_run_result(VALID_SPEC_JSON),
    )

    spec = await agent.generate_spec(
        request="bar chart of sales by region",
        columns=COLUMNS,
        row_count=10,
        file="result_abc.json",
    )

    assert isinstance(spec, VizSpec)
    assert spec.chart.type == "bar"
    assert agent.agent.run.call_count == 1


@pytest.mark.asyncio
async def test_generate_spec_self_corrects_after_invalid_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the first attempt fails validation, the agent retries with error
    feedback and succeeds on the second attempt."""
    agent = await _make_spec_agent(monkeypatch)
    agent.agent.run = AsyncMock(
        side_effect=[
            _make_run_result(INVALID_SPEC_JSON),
            _make_run_result(VALID_SPEC_JSON),
        ],
    )

    spec = await agent.generate_spec(
        request="bar chart of sales by region",
        columns=COLUMNS,
        row_count=10,
        file="result_abc.json",
    )

    assert isinstance(spec, VizSpec)
    assert spec.chart.type == "bar"
    assert agent.agent.run.call_count == 2

    # The second call should include the error message as prompt
    second_call_prompt = agent.agent.run.call_args_list[1].args[0]
    assert "failed validation" in second_call_prompt

    # And should include the message history from the first attempt
    second_call_kwargs = agent.agent.run.call_args_list[1].kwargs
    assert second_call_kwargs.get("message_history") is not None


@pytest.mark.asyncio
async def test_generate_spec_self_corrects_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the agent returns non-JSON on the first try, it retries."""
    agent = await _make_spec_agent(monkeypatch)
    agent.agent.run = AsyncMock(
        side_effect=[
            _make_run_result("Here is your chart spec: not json"),
            _make_run_result(VALID_SPEC_JSON),
        ],
    )

    spec = await agent.generate_spec(
        request="bar chart",
        columns=COLUMNS,
        row_count=10,
        file="result_abc.json",
    )

    assert isinstance(spec, VizSpec)
    assert agent.agent.run.call_count == 2


@pytest.mark.asyncio
async def test_generate_spec_raises_after_all_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After MAX_RETRIES + 1 attempts all fail, the last error is raised."""
    agent = await _make_spec_agent(monkeypatch)
    agent.agent.run = AsyncMock(
        return_value=_make_run_result(INVALID_SPEC_JSON),
    )

    with pytest.raises(ValidationError):
        await agent.generate_spec(
            request="bar chart",
            columns=COLUMNS,
            row_count=10,
            file="result_abc.json",
        )

    # 1 initial + MAX_RETRIES retries
    assert agent.agent.run.call_count == MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_generate_spec_succeeds_on_last_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent can succeed on the very last attempt."""
    agent = await _make_spec_agent(monkeypatch)
    bad_results = [_make_run_result(INVALID_SPEC_JSON)] * MAX_RETRIES
    agent.agent.run = AsyncMock(
        side_effect=[*bad_results, _make_run_result(VALID_SPEC_JSON)],
    )

    spec = await agent.generate_spec(
        request="bar chart",
        columns=COLUMNS,
        row_count=10,
        file="result_abc.json",
    )

    assert isinstance(spec, VizSpec)
    assert agent.agent.run.call_count == MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_generate_spec_passes_message_history_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each retry passes the previous conversation's messages as history."""
    fake_messages = [{"role": "assistant", "content": "..."}]
    agent = await _make_spec_agent(monkeypatch)
    agent.agent.run = AsyncMock(
        side_effect=[
            _make_run_result(INVALID_SPEC_JSON, messages=fake_messages),
            _make_run_result(VALID_SPEC_JSON),
        ],
    )

    await agent.generate_spec(
        request="bar chart",
        columns=COLUMNS,
        row_count=10,
        file="result_abc.json",
    )

    # First call: no message_history
    first_kwargs = agent.agent.run.call_args_list[0].kwargs
    assert first_kwargs.get("message_history") is None

    # Second call: message_history from first result
    second_kwargs = agent.agent.run.call_args_list[1].kwargs
    assert second_kwargs["message_history"] == fake_messages


# -- _parse_json edge cases -------------------------------------------------


def test_parse_json_extracts_from_markdown_fenced() -> None:
    text = 'Here is the spec:\n```json\n{"version": "1"}\n```'
    # The fallback finds the { ... } substring
    result = _parse_json(text)
    assert result == {"version": "1"}


def test_parse_json_raises_on_no_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_json("no json here")


def test_parse_json_raises_on_array() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_json("[1, 2, 3]")

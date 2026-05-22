from io import StringIO

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RequestUsage, RunUsage
from saber_tui.utils import strip_ansi

from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.usage import (
    SessionUsage,
    format_cost_usd,
    request_usages_from_messages,
)
from sqlsaber.theme.manager import create_console


def test_session_usage_tracks_cumulative_usage_context_and_cache_aware_cost() -> None:
    usage = SessionUsage()

    usage.add_run(
        RunUsage(
            input_tokens=3100,
            cache_write_tokens=1000,
            cache_read_tokens=2000,
            output_tokens=10,
            requests=1,
            tool_calls=1,
        ),
        final_context_tokens=100,
        model_name="anthropic:claude-opus-4-6",
    )

    assert usage.total_input_tokens == 3100
    assert usage.total_output_tokens == 10
    assert usage.current_context_tokens == 100
    assert usage.cache_write_tokens == 1000
    assert usage.cache_read_tokens == 2000
    assert usage.total_cost_usd == pytest.approx(0.008)


def test_session_usage_marks_cost_unknown_when_model_name_is_missing() -> None:
    usage = SessionUsage()

    usage.add_run(
        RunUsage(input_tokens=1000, output_tokens=100, requests=1),
        final_context_tokens=1000,
        model_name=None,
    )

    assert usage.total_input_tokens == 1000
    assert usage.total_output_tokens == 100
    assert usage.current_context_tokens == 1000
    assert usage.total_cost_usd is None
    assert format_cost_usd(usage.total_cost_usd) == "n/a"


def test_session_usage_marks_multi_request_aggregate_cost_unknown_without_request_usages() -> None:
    usage = SessionUsage()

    usage.add_run(
        RunUsage(input_tokens=3100, output_tokens=10, requests=2),
        final_context_tokens=100,
        model_name="anthropic:claude-opus-4-6",
    )

    assert usage.total_input_tokens == 3100
    assert usage.total_output_tokens == 10
    assert usage.current_context_tokens == 100
    assert usage.total_cost_usd is None
    assert format_cost_usd(usage.total_cost_usd) == "n/a"


def test_session_usage_prices_multi_request_usages_individually() -> None:
    usage = SessionUsage()

    usage.add_run(
        RunUsage(input_tokens=300_000, output_tokens=0, requests=2),
        final_context_tokens=150_000,
        model_name="anthropic:claude-sonnet-4-5",
        request_usages=[
            RequestUsage(input_tokens=150_000, output_tokens=0),
            RequestUsage(input_tokens=150_000, output_tokens=0),
        ],
    )

    assert usage.total_input_tokens == 300_000
    assert usage.current_context_tokens == 150_000
    assert usage.total_cost_usd == pytest.approx(0.9)
    assert format_cost_usd(usage.total_cost_usd) == "$0.9000"


def test_request_usages_from_messages_extracts_only_model_responses() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[TextPart(content="hi")],
            usage=RequestUsage(input_tokens=10, output_tokens=2),
        ),
        ModelResponse(
            parts=[TextPart(content="again")],
            usage=RequestUsage(input_tokens=15, output_tokens=3),
        ),
    ]

    assert request_usages_from_messages(messages) == [
        RequestUsage(input_tokens=10, output_tokens=2),
        RequestUsage(input_tokens=15, output_tokens=3),
    ]


def test_format_cost_usd_handles_known_zero_tiny_and_unknown_costs() -> None:
    assert format_cost_usd(0) == "$0.0000"
    assert format_cost_usd(0.00001) == "<$0.0001"
    assert format_cost_usd(0.01234) == "$0.0123"
    assert format_cost_usd(None) == "n/a"


def test_session_summary_labels_total_usage_and_current_context() -> None:
    buffer = StringIO()
    console = create_console(file=buffer, force_terminal=True, width=100)
    session_usage = SessionUsage()
    session_usage.add_run(
        RunUsage(input_tokens=4200, output_tokens=820, requests=1, tool_calls=7),
        final_context_tokens=999,
        model_name="anthropic:claude-opus-4-6",
    )

    DisplayManager(console).show_session_summary(session_usage)

    output = strip_ansi(buffer.getvalue())
    assert "Session Summary" in output
    assert "Usage:" in output
    assert "4.2k in / 820 out" in output
    assert "Cost:" in output
    assert "Current context:" in output
    assert "999 tokens" in output
    assert "Input:" not in output

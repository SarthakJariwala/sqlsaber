from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseState,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

import sqlsaber.cli.usage as usage_mod
from sqlsaber.cli.usage import (
    SessionUsage,
    UsageMeter,
    format_cost_usd,
    session_summary_blocks,
)
from sqlsaber.render.markdown_text import md_of

SONNET = "anthropic:claude-sonnet-4-5"
OPUS = "anthropic:claude-opus-4-6"
RUN_ID = "run-1"


def _response(
    usage: RequestUsage,
    *,
    run_id: str = RUN_ID,
    state: ModelResponseState = "complete",
) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart("ok")], usage=usage, run_id=run_id, state=state
    )


async def _no_events() -> AsyncIterator[Any]:
    return
    yield


async def _drain(ctx: Any, event_stream: AsyncIterator[Any]) -> None:
    async for _ in event_stream:
        pass


class FakeResult:
    """The ``SQLSaberResult`` fields the meter reads, built the way the SDK builds them."""

    def __init__(
        self, new_messages: Iterable[ModelMessage], usage: RunUsage | None
    ) -> None:
        self.request_usages = [
            message.usage
            for message in new_messages
            if isinstance(message, ModelResponse)
        ]
        self.usage = usage
        self.final_context_tokens = (
            self.request_usages[-1].input_tokens if self.request_usages else 0
        )


class FakeRun:
    """Drives an event-stream handler with live run state, like pydantic-ai's graph.

    ``ctx.messages`` and ``ctx.usage`` are the objects the run mutates, so a
    handler that keeps the context sees later appends.
    """

    def __init__(
        self,
        handler: Callable[..., Awaitable[None]] | None,
        *,
        history: Sequence[ModelMessage] = (),
    ) -> None:
        self.handler = handler
        self.history = list(history)
        self.messages: list[ModelMessage] = list(history)
        self.usage = RunUsage()
        self.ctx = SimpleNamespace(
            run_id=RUN_ID, messages=self.messages, usage=self.usage
        )

    async def node(self) -> None:
        if self.handler is not None:
            await self.handler(self.ctx, _no_events())

    def respond(self, usage: RequestUsage, **kwargs: Any) -> None:
        self.messages.append(_response(usage, **kwargs))
        self.usage.incr(usage)

    def tool_call(self) -> None:
        self.usage.tool_calls += 1

    def result(self) -> FakeResult:
        return FakeResult(self.messages[len(self.history) :], self.usage)


def _two_step_query(
    first: RequestUsage,
    second: RequestUsage,
    *,
    history: Sequence[ModelMessage] = (),
) -> Callable[..., Awaitable[FakeResult]]:
    """A run shaped like model message, tool call, model message.

    The handler runs once per graph node: for a model request node before its
    response is appended, for a tool node after the response and its tool call
    have been applied. That is the cadence pydantic-ai 2.9 produces.
    """

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run = FakeRun(event_stream_handler, history=history)
        await run.node()
        run.respond(first)
        run.tool_call()
        await run.node()
        await run.node()
        run.respond(second)
        await run.node()
        return run.result()

    return query


def _meter(model_id: str | None) -> tuple[UsageMeter, list[SessionUsage]]:
    snapshots: list[SessionUsage] = []
    meter = UsageMeter(
        model_id=lambda: model_id, on_change=lambda: snapshots.append(meter.session)
    )
    return meter, snapshots


@pytest.mark.asyncio
async def test_meter_notifies_after_each_completed_response_without_double_counting() -> (
    None
):
    meter, snapshots = _meter(SONNET)
    query = _two_step_query(
        RequestUsage(input_tokens=150_000), RequestUsage(input_tokens=150_000)
    )

    result = await meter.metered(query)("show revenue")

    assert [snapshot.total_input_tokens for snapshot in snapshots] == [
        150_000,
        300_000,
    ]
    assert meter.session.requests == 2
    assert meter.session.tool_calls == 1
    assert meter.session.total_input_tokens == sum(
        usage.input_tokens for usage in result.request_usages
    )
    assert meter.session.current_context_tokens == 150_000
    assert meter.session.total_cost_usd == pytest.approx(0.9)
    assert format_cost_usd(meter.session.total_cost_usd) == "$0.9000"


@pytest.mark.asyncio
async def test_replaying_an_observation_folds_nothing_and_does_not_notify() -> None:
    meter, snapshots = _meter(SONNET)

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run = FakeRun(event_stream_handler)
        run.respond(RequestUsage(input_tokens=100, output_tokens=10))
        await run.node()
        await run.node()
        return run.result()

    await meter.metered(query)("go")

    assert len(snapshots) == 1
    assert meter.session.requests == 1
    assert meter.session.total_input_tokens == 100
    assert meter.session.total_output_tokens == 10


@pytest.mark.asyncio
async def test_metered_forwards_extra_keyword_arguments() -> None:
    meter, _ = _meter(SONNET)
    received: dict[str, Any] = {}

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        received.update(kwargs)
        return FakeResult([], RunUsage())

    await meter.metered(query)("go", conversation_id="conv-1")

    assert received == {"conversation_id": "conv-1"}


@pytest.mark.asyncio
async def test_result_without_run_usage_still_folds_request_usages() -> None:
    meter, snapshots = _meter(SONNET)

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        return FakeResult([_response(RequestUsage(input_tokens=150_000))], None)

    await meter.metered(query)("go")

    assert len(snapshots) == 1
    assert meter.session.requests == 1
    assert meter.session.tool_calls == 0
    assert meter.session.total_input_tokens == 150_000


@pytest.mark.asyncio
async def test_shrunk_observation_replaces_this_turn_instead_of_stacking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[dict[str, Any]] = []
    monkeypatch.setattr(
        usage_mod,
        "log",
        SimpleNamespace(
            warning=lambda event, **fields: warnings.append({"event": event, **fields})
        ),
    )
    meter, _ = _meter(None)
    prior = _two_step_query(RequestUsage(input_tokens=7), RequestUsage(input_tokens=8))
    await meter.metered(prior)("earlier")
    assert meter.session.total_input_tokens == 15

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run = FakeRun(event_stream_handler)
        run.respond(RequestUsage(input_tokens=10))
        run.respond(RequestUsage(input_tokens=20))
        await run.node()
        run.messages[:] = [_response(RequestUsage(input_tokens=30))]
        await run.node()
        run.messages.append(_response(RequestUsage(input_tokens=40)))
        await run.node()
        return FakeResult(run.messages, None)

    await meter.metered(query)("go")

    assert meter.session.requests == 4
    assert meter.session.total_input_tokens == 85
    assert meter.session.current_context_tokens == 40
    assert warnings == [
        {"event": "usage.meter.cursor_reset", "observed": 1, "folded": 2}
    ]


@pytest.mark.asyncio
async def test_exception_from_the_presenter_handler_keeps_completed_responses() -> None:
    meter, _ = _meter(SONNET)

    class Interrupted(Exception):
        pass

    calls = 0

    async def presenter(ctx: Any, event_stream: AsyncIterator[Any]) -> None:
        nonlocal calls
        calls += 1
        await _drain(ctx, event_stream)
        if calls == 3:
            raise Interrupted

    query = _two_step_query(
        RequestUsage(input_tokens=150_000), RequestUsage(input_tokens=150_000)
    )

    with pytest.raises(Interrupted):
        await meter.metered(query)("go", event_stream_handler=presenter)

    assert meter.session.requests == 1
    assert meter.session.tool_calls == 1
    assert meter.session.total_input_tokens == 150_000
    assert meter.session.total_cost_usd == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_cancel_between_nodes_banks_responses_from_the_last_context() -> None:
    meter, snapshots = _meter(SONNET)

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run = FakeRun(event_stream_handler)
        await run.node()
        run.respond(RequestUsage(input_tokens=150_000))
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await meter.metered(query)("go")

    assert len(snapshots) == 1
    assert meter.session.requests == 1
    assert meter.session.total_input_tokens == 150_000
    assert meter.session.total_cost_usd == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_reset_clears_totals_latch_and_cursor() -> None:
    meter, snapshots = _meter("nosuchprovider:nosuchmodel")
    query = _two_step_query(
        RequestUsage(input_tokens=100), RequestUsage(input_tokens=100)
    )
    await meter.metered(query)("go")
    assert meter.session.total_input_tokens == 200
    assert meter.session.total_cost_usd is None
    notified = len(snapshots)

    meter.reset()

    assert meter.session == SessionUsage()
    assert meter.session.total_cost_usd == 0.0
    assert len(snapshots) == notified + 1
    meter._absorb([RequestUsage(input_tokens=1)] * 3, tool_calls=0)
    assert meter.session.requests == 3


def test_reset_of_an_empty_session_does_not_notify() -> None:
    meter, snapshots = _meter(SONNET)

    meter.reset()

    assert snapshots == []


@pytest.mark.asyncio
async def test_prior_turn_responses_with_another_run_id_are_excluded() -> None:
    meter, _ = _meter(SONNET)
    history = [
        ModelRequest.user_text_prompt("earlier turn"),
        _response(
            RequestUsage(input_tokens=999_999, output_tokens=999_999), run_id="run-0"
        ),
    ]
    query = _two_step_query(
        RequestUsage(input_tokens=100),
        RequestUsage(input_tokens=100),
        history=history,
    )

    await meter.metered(query)("go")

    assert meter.session.requests == 2
    assert meter.session.total_input_tokens == 200
    assert meter.session.total_output_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["incomplete", "interrupted"])
async def test_responses_that_did_not_complete_are_excluded(
    state: ModelResponseState,
) -> None:
    meter, _ = _meter(SONNET)

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run = FakeRun(event_stream_handler)
        run.respond(RequestUsage(input_tokens=100))
        run.respond(RequestUsage(input_tokens=999), state=state)
        await run.node()
        raise RuntimeError("model stream cut")

    with pytest.raises(RuntimeError):
        await meter.metered(query)("go")

    assert meter.session.requests == 1
    assert meter.session.total_input_tokens == 100
    assert meter.session.current_context_tokens == 100


@pytest.mark.asyncio
async def test_meter_prices_cache_aware_request_and_counts_tool_calls() -> None:
    meter, _ = _meter(OPUS)

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run = FakeRun(event_stream_handler)
        run.respond(
            RequestUsage(
                input_tokens=3100,
                cache_write_tokens=1000,
                cache_read_tokens=2000,
                output_tokens=10,
            )
        )
        run.tool_call()
        await run.node()
        return run.result()

    await meter.metered(query)("go")

    session = meter.session
    assert session.requests == 1
    assert session.tool_calls == 1
    assert session.total_input_tokens == 3100
    assert session.total_output_tokens == 10
    assert session.current_context_tokens == 3100
    assert session.cache_write_tokens == 1000
    assert session.cache_read_tokens == 2000
    assert session.total_cost_usd == pytest.approx(0.008)


@pytest.mark.asyncio
async def test_meter_marks_cost_unknown_when_model_id_is_missing() -> None:
    meter, _ = _meter(None)

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run = FakeRun(event_stream_handler)
        run.respond(RequestUsage(input_tokens=1000, output_tokens=100))
        await run.node()
        return run.result()

    await meter.metered(query)("go")

    assert meter.session.total_input_tokens == 1000
    assert meter.session.total_output_tokens == 100
    assert meter.session.current_context_tokens == 1000
    assert meter.session.total_cost_usd is None
    assert format_cost_usd(meter.session.total_cost_usd) == "n/a"


@pytest.mark.asyncio
async def test_meter_matches_pydantic_ai_totals_on_a_real_agent_run() -> None:
    steps = 0

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        nonlocal steps
        steps += 1
        if steps == 1:
            yield {0: DeltaToolCall(name="echo", json_args='{"text": "hi"}')}
        else:
            yield "done"

    agent = Agent(FunctionModel(stream_function=stream_fn), name="probe")

    @agent.tool_plain
    def echo(text: str) -> str:
        return text

    history = [
        ModelRequest.user_text_prompt("earlier turn"),
        _response(
            RequestUsage(input_tokens=999_999, output_tokens=999_999), run_id="run-0"
        ),
    ]

    async def query(
        prompt: str, /, *, event_stream_handler: Any = None, **kwargs: Any
    ) -> FakeResult:
        run_result = await agent.run(
            prompt, message_history=history, event_stream_handler=event_stream_handler
        )
        return FakeResult(run_result.new_messages(), run_result.usage)

    meter, snapshots = _meter(SONNET)

    result = await meter.metered(query)("go", event_stream_handler=_drain)

    run_usage = result.usage
    assert run_usage is not None
    assert meter.session.requests == run_usage.requests == 2
    assert meter.session.tool_calls == run_usage.tool_calls == 1
    assert meter.session.total_input_tokens == run_usage.input_tokens
    assert meter.session.total_output_tokens == run_usage.output_tokens
    assert (
        meter.session.current_context_tokens == result.request_usages[-1].input_tokens
    )
    assert meter.session.total_cost_usd is not None
    assert snapshots[0].requests == 1
    assert snapshots[-1].requests == 2


def test_format_cost_usd_handles_known_zero_tiny_and_unknown_costs() -> None:
    assert format_cost_usd(0) == "$0.0000"
    assert format_cost_usd(0.00001) == "<$0.0001"
    assert format_cost_usd(0.01234) == "$0.0123"
    assert format_cost_usd(None) == "n/a"


def test_session_summary_labels_total_usage_and_current_context() -> None:
    session_usage = SessionUsage(
        requests=1,
        tool_calls=7,
        total_input_tokens=4200,
        total_output_tokens=820,
        current_context_tokens=999,
        total_cost_usd=0.0415,
    )

    output = md_of(session_summary_blocks(session_usage))
    assert "Session Summary" in output
    assert "Usage:" in output
    assert "4.2k in / 820 out" in output
    assert "Cost:" in output
    assert "Current context:" in output
    assert "999 tokens" in output
    assert "Input:" not in output

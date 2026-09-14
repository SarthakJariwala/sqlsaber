from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    EnqueuedMessagesEvent,
    ModelMessage,
    ModelRequest,
    PartStartEvent,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from sqlsaber import (
    RunInProgressError,
    SQLSaber,
    SQLSaberOptions,
    SteerIdleError,
)
from sqlsaber.config.settings import Config


def _options(**overrides: Any) -> SQLSaberOptions:
    values: dict[str, Any] = {
        "database": "sqlite:///:memory:",
        "settings": Config.in_memory(
            model_name="anthropic:claude-3-5-sonnet",
            api_keys={"anthropic": "test-key"},
        ),
    }
    values.update(overrides)
    return SQLSaberOptions(**values)


def _user_texts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def _tool_calls(messages: list[ModelMessage]) -> int:
    return sum(
        1
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
    )


def _messages_contain_user(messages: list[ModelMessage], text: str) -> bool:
    return any(text in part for part in _user_texts(messages))


class _Harness:
    def __init__(self) -> None:
        self.requests: list[list[ModelMessage]] = []
        self.first_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.tool_entered = asyncio.Event()
        self.tool_release = asyncio.Event()
        self.delivered: list[str] = []
        self.handler_events: list[object] = []

    async def stream_text(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            self.first_entered.set()
            await self.release.wait()
            yield "first"
            return
        yield f"answer #{len(self.requests)}"

    async def stream_tool_then_text(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        self.requests.append(list(messages))
        if _tool_calls(messages) < 1:
            yield {0: DeltaToolCall(name="list_tables", json_args="{}")}
            return
        yield "after tool"

    async def stream_custom_tool(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        self.requests.append(list(messages))
        if _tool_calls(messages) < 1:
            yield {0: DeltaToolCall(name="emit_note", json_args="{}")}
            return
        yield "after tool"

    async def stream_immediate(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[Any]:
        self.requests.append(list(messages))
        yield "done"

    async def handler(self, ctx: RunContext[Any], events: Any) -> None:
        async for event in events:
            self.handler_events.append(event)
            if isinstance(event, EnqueuedMessagesEvent):
                self.delivered.append(event.enqueue_id)


async def _wait_or_fail(
    event: asyncio.Event, task: asyncio.Task[Any], *, name: str
) -> None:
    waiter = asyncio.create_task(event.wait())
    done, _ = await asyncio.wait(
        {waiter, task}, timeout=10, return_when=asyncio.FIRST_COMPLETED
    )
    waiter.cancel()
    if task in done:
        raise AssertionError(f"query ended before {name}: {task.exception()}")
    if not done:
        task.cancel()
        raise AssertionError(f"{name} never happened within 10s")


@pytest.mark.asyncio
async def test_steer_during_model_request_lands_on_the_next_request() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(harness.first_entered, task, name="first request")
            enqueue_id = saber.steer("only US")
            assert isinstance(enqueue_id, str) and enqueue_id
            assert saber.pending_steers == ("only US",)
            harness.release.set()
            await task
        assert len(harness.requests) == 2
        last_request = harness.requests[1][-1]
        assert isinstance(last_request, ModelRequest)
        assert any(
            isinstance(part, UserPromptPart) and part.content == "only US"
            for part in last_request.parts
        )
        assert _messages_contain_user(saber.messages, "only US")
        assert saber.pending_steers == ()
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_steer_while_a_tool_runs_lands_after_the_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()

    async def hung_list_tables() -> dict[str, Any]:
        harness.tool_entered.set()
        await harness.tool_release.wait()
        return {"tables": []}

    for entry in saber.registry:
        monkeypatch.setattr(entry.schema_manager, "list_tables", hung_list_tables)

    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_tool_then_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(harness.tool_entered, task, name="list_tables")
            saber.steer("only US")
            assert saber.pending_steers == ("only US",)
            harness.tool_release.set()
            await task
        second = _user_texts(harness.requests[1])
        assert "hello" in second
        assert "only US" in second
        assert any(
            isinstance(part, ToolCallPart)
            for message in harness.requests[1]
            for part in message.parts
        )
        assert saber.pending_steers == ()
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_steer_before_the_first_request_is_delivered_with_the_prompt() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await asyncio.sleep(0)
            enqueue_id = saber.steer("only US")
            harness.release.set()
            await task
        assert len(harness.requests) == 1
        users = _user_texts(harness.requests[0])
        assert "hello" in users
        assert "only US" in users
        assert harness.delivered[0] == enqueue_id
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_steer_is_legal_one_loop_pass_after_query_is_scheduled() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await asyncio.sleep(0)
            enqueue_id = saber.steer("only US")
            assert isinstance(enqueue_id, str) and enqueue_id
            assert saber.pending_steers == ("only US",)
            harness.release.set()
            await task
        assert saber.pending_steers == ()
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_clear_steers_returns_texts_in_order_and_prevents_delivery() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(harness.first_entered, task, name="first request")
            saber.steer("a")
            saber.steer("b")
            assert saber.clear_steers() == ["a", "b"]
            assert saber.pending_steers == ()
            harness.release.set()
            await task
        assert len(harness.requests) == 1
        assert not _messages_contain_user(saber.messages, "a")
        assert not _messages_contain_user(saber.messages, "b")
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_delivered_steer_is_visible_to_the_event_stream_handler() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(harness.first_entered, task, name="first request")
            enqueue_id = saber.steer("only US")
            harness.release.set()
            await task
        first_enqueued = next(
            event
            for event in harness.handler_events
            if isinstance(event, EnqueuedMessagesEvent)
        )
        first_part_of_request_2 = None
        seen_enqueued = False
        for event in harness.handler_events:
            if isinstance(event, EnqueuedMessagesEvent):
                seen_enqueued = True
            if seen_enqueued and isinstance(event, PartStartEvent):
                first_part_of_request_2 = event
                break
        assert first_enqueued.enqueue_id == enqueue_id
        assert first_part_of_request_2 is not None
        assert harness.handler_events.index(
            first_enqueued
        ) < harness.handler_events.index(first_part_of_request_2)
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_steer_while_idle_raises_and_queue_reads_are_empty() -> None:
    saber = SQLSaber(options=_options())
    try:
        with pytest.raises(SteerIdleError, match="No query is running.") as exc:
            saber.steer("too late")
        assert str(exc.value) == "No query is running."
        assert saber.pending_steers == ()
        assert saber.clear_steers() == []
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_steer_rejects_blank_message() -> None:
    saber = SQLSaber(options=_options())
    try:
        with pytest.raises(ValueError, match="non-empty"):
            saber.steer("   ")
        assert saber.pending_steers == ()
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_abort_drops_pending_steers_and_commits_nothing() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(harness.first_entered, task, name="first request")
            saber.steer("never delivered")
            assert saber.pending_steers == ("never delivered",)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                assert type(exc).__name__ == "RunCancelled"
        assert saber.messages == []
        assert saber.pending_steers == ()
        with pytest.raises(SteerIdleError):
            saber.steer("after abort")
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_steer_from_another_thread_fails_loudly() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    loop = asyncio.get_running_loop()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_text)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(harness.first_entered, task, name="first request")
            with pytest.raises(RuntimeError, match="call_soon_threadsafe"):
                await loop.run_in_executor(None, saber.steer, "x")
            assert saber.pending_steers == ()
            harness.release.set()
            await task
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_close_still_refused_during_the_tail() -> None:
    entered_save = asyncio.Event()
    release_save = asyncio.Event()

    class SlowThreadManager:
        current_thread_id = None
        first_message = True

        async def save_run(self, **kwargs: Any) -> list[Any]:
            entered_save.set()
            await release_save.wait()
            return []

        async def end_current_thread(self) -> None:
            return None

    saber = SQLSaber(options=_options(thread_manager=SlowThreadManager()))
    harness = _Harness()
    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_immediate)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(entered_save, task, name="thread save")
            with pytest.raises(SteerIdleError):
                saber.steer("too late")
            with pytest.raises(RunInProgressError):
                await saber.close()
            release_save.set()
            await task
    finally:
        if not entered_save.is_set():
            release_save.set()
        await saber.close()


@pytest.mark.asyncio
async def test_late_steer_starts_one_more_model_request() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()
    steered = False

    async def handler(ctx: RunContext[Any], events: Any) -> None:
        nonlocal steered
        async for event in events:
            if isinstance(event, EnqueuedMessagesEvent):
                harness.delivered.append(event.enqueue_id)
        if len(harness.requests) == 1 and not steered:
            saber.steer("also break out by region")
            steered = True

    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_immediate)
        ):
            await saber.query("hello", event_stream_handler=handler)
        assert len(harness.requests) == 2
        assert _user_texts(harness.requests[-1])[-1] == "also break out by region"
        assert _messages_contain_user(saber.messages, "also break out by region")
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_plugin_tool_enqueue_survives_clear_steers() -> None:
    saber = SQLSaber(options=_options())
    harness = _Harness()

    @saber.agent.agent.tool
    async def emit_note(ctx: RunContext[Any]) -> str:
        ctx.enqueue("[tool note] partitions were pruned")
        harness.tool_entered.set()
        await harness.tool_release.wait()
        return "ok"

    try:
        with saber.agent.agent.override(
            model=FunctionModel(stream_function=harness.stream_custom_tool)
        ):
            task = asyncio.create_task(
                saber.query("hello", event_stream_handler=harness.handler)
            )
            await _wait_or_fail(harness.tool_entered, task, name="emit_note")
            saber.steer("mine, to be cleared")
            assert saber.pending_steers == ("mine, to be cleared",)
            assert saber.clear_steers() == ["mine, to be cleared"]
            assert saber.pending_steers == ()
            harness.tool_release.set()
            await task
        last = _user_texts(harness.requests[-1])
        assert last == ["hello", "[tool note] partitions were pruned"]
    finally:
        await saber.close()

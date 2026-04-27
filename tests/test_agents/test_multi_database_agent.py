import json
import tempfile
from collections.abc import AsyncIterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from sqlsaber.agents.multi_database_agent import (
    ChildAnswerPayload,
    DatabaseAnswer,
    DatabaseChild,
    DatabaseDescriptor,
    MultiDatabaseCoordinator,
)
from sqlsaber.threads.manager import ThreadManager
from sqlsaber.threads.storage import ThreadStorage


@pytest.fixture
def temp_storage():
    """Local storage fixture for coordinator thread tests."""
    with tempfile.TemporaryDirectory() as tmp:
        storage = ThreadStorage()
        storage.db_path = Path(tmp) / "threads.db"
        yield storage


def _messages(user_text: str, assistant_text: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(user_text)]),
        ModelResponse(parts=[TextPart(assistant_text)]),
    ]


class FakeChildAgent:
    def __init__(self, outputs: list[ChildAnswerPayload]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []
        self.agent = SimpleNamespace(model=SimpleNamespace(model_name="fake-child"))

    async def run(
        self,
        prompt: str,
        *,
        message_history: list[ModelMessage] | None = None,
        usage: Any | None = None,
    ) -> Any:
        self.calls.append(
            {
                "prompt": prompt,
                "message_history": message_history,
                "usage": usage,
            }
        )
        output = self.outputs.pop(0)
        messages = [
            *(message_history or []),
            *_messages(prompt, output.summary),
        ]
        return SimpleNamespace(
            output=output,
            all_messages=lambda: messages,
            all_messages_json=lambda: ModelMessagesTypeAdapter.dump_json(messages),
        )


class FailingChildAgent:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.agent = SimpleNamespace(model=SimpleNamespace(model_name="fake-child"))

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        raise self.error


class InvalidOutputChildAgent:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(model=SimpleNamespace(model_name="fake-child"))

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        return SimpleNamespace(output="not structured")


class AsyncClosableChildAgent:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_database_answer_uses_specified_fields_only() -> None:
    assert set(DatabaseAnswer.model_fields) == {
        "database_id",
        "database_name",
        "thread_id",
        "summary",
        "evidence",
        "limitations",
    }


def test_database_descriptor_schema_and_system_prompt_include_database_metadata() -> (
    None
):
    coordinator = MultiDatabaseCoordinator(
        children={
            "warehouse": DatabaseChild(
                descriptor=DatabaseDescriptor(
                    id="warehouse",
                    name="Warehouse",
                    type="postgresql",
                    description="Orders warehouse",
                    summary="Contains order and fulfillment facts.",
                    thread_id="thread-123",
                ),
                agent=FakeChildAgent([]),
                thread_manager=ThreadManager(),
                database_name="Warehouse",
                message_history=[],
            )
        },
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )

    prompt = coordinator.system_prompt_text()

    assert coordinator.database_label == "analytics"
    assert "Warehouse" in prompt
    assert "postgresql" in prompt
    assert "Orders warehouse" in prompt
    assert "Contains order and fulfillment facts." in prompt
    assert "thread-123" in prompt
    assert "cross-database SQL joins cannot be executed" in prompt
    assert "query databases independently" in prompt
    assert "thread ID" in prompt


@pytest.mark.asyncio
async def test_ask_database_direct_routes_reuses_thread_history_and_preserves_payload(
    temp_storage,
) -> None:
    child_agent = FakeChildAgent(
        outputs=[
            ChildAnswerPayload(
                summary="Orders total is 42.",
                evidence=["SELECT count(*) FROM orders"],
                limitations=["Only completed orders were included."],
            ),
            ChildAnswerPayload(
                summary="Revenue total is 99.",
                evidence=["SELECT sum(total) FROM orders"],
                limitations=["Refund data is unavailable."],
            ),
        ]
    )
    child = DatabaseChild(
        descriptor=DatabaseDescriptor(
            id="orders",
            name="Orders",
            type="sqlite",
            description="Order records",
            summary="Orders and revenue.",
        ),
        agent=child_agent,
        thread_manager=ThreadManager(storage=temp_storage),
        database_name="Orders",
        message_history=[],
    )
    coordinator = MultiDatabaseCoordinator(
        children={"orders": child},
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )
    usage = object()

    first = await coordinator.ask_database_direct(
        "orders", "How many orders?", usage=usage
    )
    second = await coordinator.ask_database_direct(
        "orders", "What revenue?", usage=usage
    )

    assert isinstance(first, DatabaseAnswer)
    assert first.database_id == "orders"
    assert first.database_name == "Orders"
    assert first.thread_id is not None
    assert first.summary == "Orders total is 42."
    assert first.evidence == ["SELECT count(*) FROM orders"]
    assert first.limitations == ["Only completed orders were included."]

    assert second.thread_id == first.thread_id
    assert second.summary == "Revenue total is 99."
    assert second.evidence == ["SELECT sum(total) FROM orders"]
    assert second.limitations == ["Refund data is unavailable."]

    assert child.descriptor.thread_id == first.thread_id
    assert child_agent.calls[0]["message_history"] == []
    assert child_agent.calls[0]["usage"] is usage
    assert child_agent.calls[1]["message_history"] == child.message_history[:2]
    assert child_agent.calls[1]["usage"] is usage
    assert len(child.message_history) == 4

    stored_messages = await temp_storage.get_thread_messages(first.thread_id)
    assert len(stored_messages) == 4
    stored_thread = await temp_storage.get_thread(first.thread_id)
    assert stored_thread.title == "[Orders] How many orders?"
    assert stored_thread.model_name == "fake-child"
    assert json.loads(stored_thread.extra_metadata) == {
        "kind": "multi_database_child",
        "database_id": "orders",
    }


@pytest.mark.asyncio
async def test_ask_database_direct_returns_limitation_for_unknown_database() -> None:
    coordinator = MultiDatabaseCoordinator(
        children={},
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )

    answer = await coordinator.ask_database_direct("missing", "What is here?")

    assert answer == DatabaseAnswer(
        database_id="missing",
        database_name="missing",
        thread_id=None,
        summary="Unable to answer from database 'missing'.",
        evidence=[],
        limitations=["Unknown database id 'missing'."],
    )


@pytest.mark.asyncio
async def test_ask_database_direct_returns_failed_answer_when_child_run_raises(
    temp_storage,
) -> None:
    child = DatabaseChild(
        descriptor=DatabaseDescriptor(
            id="orders",
            name="Orders",
            type="sqlite",
            description="Order records",
            summary="Orders and revenue.",
        ),
        agent=FailingChildAgent(ValueError("structured output validation failed")),
        thread_manager=ThreadManager(storage=temp_storage),
        database_name="Orders",
        message_history=[],
    )
    coordinator = MultiDatabaseCoordinator(
        children={"orders": child},
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )

    answer = await coordinator.ask_database_direct("orders", "How many orders?")

    assert answer.database_id == "orders"
    assert answer.database_name == "Orders"
    assert answer.thread_id is not None
    assert answer.evidence == []
    assert answer.limitations == [
        "Child agent failed: structured output validation failed"
    ]
    assert child.message_history == []

    stored_thread = await temp_storage.get_thread(answer.thread_id)
    assert stored_thread.title == "[Orders] How many orders?"
    assert stored_thread.model_name == "fake-child"
    assert json.loads(stored_thread.extra_metadata) == {
        "kind": "multi_database_child",
        "database_id": "orders",
    }


@pytest.mark.asyncio
async def test_ask_database_direct_returns_limitation_when_child_output_is_invalid(
    temp_storage,
) -> None:
    child = DatabaseChild(
        descriptor=DatabaseDescriptor(
            id="orders",
            name="Orders",
            type="sqlite",
            description="Order records",
            summary="Orders and revenue.",
        ),
        agent=InvalidOutputChildAgent(),
        thread_manager=ThreadManager(storage=temp_storage),
        database_name="Orders",
        message_history=[],
    )
    coordinator = MultiDatabaseCoordinator(
        children={"orders": child},
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )

    answer = await coordinator.ask_database_direct("orders", "How many orders?")

    assert answer.database_id == "orders"
    assert answer.database_name == "Orders"
    assert answer.thread_id is not None
    assert answer.summary == (
        "Unable to answer from database 'orders' because the child agent returned "
        "invalid structured output."
    )
    assert answer.evidence == []
    assert answer.limitations == [
        "Child agent did not return ChildAnswerPayload structured output."
    ]
    assert not hasattr(answer, "success")
    assert child.message_history == []


@pytest.mark.asyncio
async def test_registered_ask_database_tool_forwards_usage(monkeypatch) -> None:
    coordinator = MultiDatabaseCoordinator(
        children={},
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )
    usage = object()
    captured: dict[str, Any] = {}

    async def fake_ask_database_direct(
        database_id: str,
        question: str,
        *,
        usage: Any | None = None,
    ) -> DatabaseAnswer:
        captured["database_id"] = database_id
        captured["question"] = question
        captured["usage"] = usage
        return DatabaseAnswer(
            database_id=database_id,
            database_name="Orders",
            thread_id="thread-1",
            summary="ok",
            evidence=[],
            limitations=[],
        )

    monkeypatch.setattr(coordinator, "ask_database_direct", fake_ask_database_direct)
    ask_tool = coordinator.agent._function_toolset.tools["ask_database"]

    answer = await ask_tool.function(
        SimpleNamespace(usage=usage),
        "orders",
        "How many orders?",
    )

    assert answer.summary == "ok"
    assert captured == {
        "database_id": "orders",
        "question": "How many orders?",
        "usage": usage,
    }


@pytest.mark.asyncio
async def test_close_closes_child_agents_with_async_close() -> None:
    child_agent = AsyncClosableChildAgent()
    coordinator = MultiDatabaseCoordinator(
        children={
            "orders": DatabaseChild(
                descriptor=DatabaseDescriptor(
                    id="orders",
                    name="Orders",
                    type="sqlite",
                    description="Order records",
                    summary="Orders and revenue.",
                ),
                agent=child_agent,
                thread_manager=ThreadManager(),
                database_name="Orders",
                message_history=[],
            )
        },
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )

    await coordinator.close()

    assert child_agent.closed is True


@pytest.mark.asyncio
async def test_run_forwards_history_stream_handler_and_deps(monkeypatch) -> None:
    coordinator = MultiDatabaseCoordinator(
        children={},
        database_label="analytics",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )
    message_history = _messages("hello", "there")
    captured: dict[str, Any] = {}

    async def fake_stream_handler(
        ctx: RunContext[Any],
        events: AsyncIterable[AgentStreamEvent],
    ) -> None:
        _ = ctx
        async for _ in events:
            pass

    async def fake_run(prompt: str, **kwargs: Any) -> Any:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return SimpleNamespace(output="ok")

    monkeypatch.setattr(coordinator.agent, "run", fake_run)

    result = await coordinator.run(
        "Compare databases.",
        message_history=message_history,
        event_stream_handler=fake_stream_handler,
    )

    assert result.output == "ok"
    assert captured["prompt"] == "Compare databases."
    assert captured["message_history"] == message_history
    assert captured["event_stream_handler"] is fake_stream_handler
    assert captured["deps"].children is coordinator.children

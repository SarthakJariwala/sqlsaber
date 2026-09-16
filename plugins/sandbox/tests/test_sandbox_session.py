"""Persistent session semantics with a fake kernel execution boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RunUsage, UsageLimits

from sqlsaber_sandbox.result import ArtifactRef, CellResult, Workspace, WorkspaceFile
from sqlsaber_sandbox.session import SandboxSession


class FakeExecution:
    instances: list["FakeExecution"] = []

    def __init__(self, config, backend=None):
        self.config = config
        self.root = "/fake"
        self.lost = False
        self.settled = None
        self.open_calls = 0
        self.staged: list[str] = []
        self.uploads: list[tuple[bytes, str]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.executions: list[str] = []
        self.artifact_calls = 0
        self.instances.append(self)

    async def open(self):
        self.open_calls += 1

    async def close(self):
        pass

    async def stage(self, workspace):
        self.staged.extend(f.name for f in workspace.files)

    async def upload(self, data, remote):
        self.uploads.append((data, remote))

    async def execute(self, code, execution_id):
        self.executions.append(code)
        self.started.set()
        if self.block:
            await self.release.wait()
        return CellResult(execution_id, "ok", ())

    async def artifacts(self):
        self.artifact_calls += 1
        return (ArtifactRef("raw.bin", b"\x00\xff"),)

    async def request(self, operation, **kwargs):
        return {}


@pytest.fixture(autouse=True)
def fake_execution(monkeypatch):
    FakeExecution.instances.clear()
    monkeypatch.setattr("sqlsaber_sandbox.session.KernelExecution", FakeExecution)


def _text_model(answer="done"):
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(answer)]))


async def test_request_zero_preflight_does_not_provision() -> None:
    session = SandboxSession(model=_text_model())
    with pytest.raises(UsageLimitExceeded):
        await session.analyze("goal", usage_limits=UsageLimits(request_limit=0))
    assert session._execution.open_calls == 0


async def test_cumulative_staging_deduplicates_and_rejects_conflicts() -> None:
    session = SandboxSession(model=_text_model())
    a = Workspace((WorkspaceFile("a.bin", b"a"),))
    b = Workspace((WorkspaceFile("b.bin", b"b"),))
    await session.execute("1", workspace=a)
    await session.execute("2", workspace=Workspace((a.files[0], b.files[0])))
    assert session._execution.staged == ["a.bin", "b.bin"]
    assert (
        len([r for _, r in session._execution.uploads if r.endswith("manifest.json")])
        == 2
    )
    with pytest.raises(ValueError, match="different contents"):
        await session.execute(
            "3", workspace=Workspace((WorkspaceFile("a.bin", b"changed"),))
        )
    assert session._execution.executions == ["1", "2"]


async def test_same_session_serializes_while_separate_sessions_overlap() -> None:
    one, two = SandboxSession(model=_text_model()), SandboxSession(model=_text_model())
    one._execution.block = two._execution.block = True
    first = asyncio.create_task(one.execute("first"))
    await one._execution.started.wait()
    same = asyncio.create_task(one.execute("same"))
    other = asyncio.create_task(two.execute("other"))
    await two._execution.started.wait()
    await asyncio.sleep(0)
    assert one._execution.executions == ["first"]
    one._execution.release.set()
    two._execution.release.set()
    await asyncio.gather(first, same, other)
    assert one._execution.executions == ["first", "same"]


async def test_child_tool_loop_continues_history_and_budgets_are_per_call() -> None:
    prompts: list[list[Any]] = []

    def respond(messages, info):
        prompts.append(list(messages))
        returns = [
            p
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        completed = sum(
            isinstance(part, TextPart) and part.content == "done"
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
        )
        if len(returns) == completed:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_python",
                        {"code": f"x={len(prompts)}"},
                        tool_call_id=f"cell{len(prompts)}",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    session = SandboxSession(model=FunctionModel(respond))
    await session.analyze("first", usage_limits=UsageLimits(request_limit=2))
    await session.analyze("second", usage_limits=UsageLimits(request_limit=2))
    assert len(session._execution.executions) == 2
    assert any(
        isinstance(p, TextPart) and p.content == "done"
        for m in prompts[-2]
        if isinstance(m, ModelResponse)
        for p in m.parts
    )


async def test_failed_goal_retains_history_and_repairs_outstanding_tool() -> None:
    def respond(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "execute_python", {"code": "mutate()"}, tool_call_id="pending"
                )
            ]
        )

    session = SandboxSession(model=FunctionModel(respond))

    async def fail(code, execution_id):
        raise RuntimeError("transport")

    session._execution.execute = fail
    with pytest.raises(RuntimeError, match="transport"):
        await session.analyze("goal")
    returns = [
        p
        for m in session._history
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]
    assert returns and returns[-1].tool_call_id == "pending"
    assert "do not automatically replay" in str(returns[-1].content)


async def test_snapshot_exports_binary_without_replaying_cells() -> None:
    session = SandboxSession(model=_text_model())
    await session.execute("side_effect()")
    before = list(session._execution.executions)
    first = await session.snapshot("saved")
    second = await session.snapshot("saved again")
    assert session._execution.executions == before
    assert first.files[0].data == b"\x00\xff"
    assert first.analysis_id != second.analysis_id


async def test_current_parent_usage_is_not_reused_by_later_default_call() -> None:
    session = SandboxSession(model=_text_model())
    parent = RunUsage(requests=1)
    with pytest.raises(UsageLimitExceeded):
        await session.analyze(
            "bounded", usage_limits=UsageLimits(request_limit=1), parent_usage=parent
        )
    result = await session.analyze("fresh")
    assert result.answer == "done"


async def test_cancelled_close_is_shared_and_failed_cleanup_is_retryable():
    session = SandboxSession(model=_text_model())
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def close():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        if calls == 1:
            raise RuntimeError("temporary deletion failure")

    session._execution.close = close
    first = asyncio.create_task(session.close())
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert calls == 1
    release.set()
    with pytest.raises(RuntimeError, match="deletion"):
        await second
    await session.close()
    assert calls == 2


async def test_idle_timer_does_not_expire_during_model_request():
    from sqlsaber_sandbox import SandboxConfig

    async def slow_model(messages, info):
        await asyncio.sleep(1.1)
        return ModelResponse(parts=[TextPart("completed")])

    async with SandboxSession(
        model=FunctionModel(slow_model), config=SandboxConfig(idle_seconds=0.01)
    ) as session:
        result = await session.analyze("slow model")
        assert result.answer == "completed"
        assert not session.closed


async def test_cancellation_journals_known_partial_output():
    session = SandboxSession(model=_text_model())

    async def interrupted(code, execution_id):
        session._execution.settled = CellResult(
            execution_id,
            "interrupted",
            ({"output_type": "stream", "name": "stdout", "text": "partial"},),
            1,
        )
        raise asyncio.CancelledError

    session._execution.execute = interrupted
    with pytest.raises(asyncio.CancelledError):
        await session.execute("print('partial'); wait()")
    assert session._journal[-1][1].outputs[0]["text"] == "partial"
    assert session._journal[-1][1].execution_count == 1


async def test_external_close_does_not_deadlock_with_context_exit():
    from unittest.mock import AsyncMock

    session = SandboxSession(model=_text_model())
    session._execution.block = True
    session._execution.close = AsyncMock()

    async def use_session():
        async with session:
            await session.execute("long-running")

    caller = asyncio.create_task(use_session())
    await session._execution.started.wait()
    async with asyncio.timeout(2):
        await session.close()
        with pytest.raises(asyncio.CancelledError):
            await caller
    session._execution.close.assert_awaited_once()

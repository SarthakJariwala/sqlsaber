"""Managed SDK/capability tests; all provider boundaries are faked."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits

from sqlsaber.artifacts import ArtifactPublicationError, InMemoryArtifactStore
from sqlsaber.run_usage import bind_usage_limits
from sqlsaber_sandbox.capability import Sandbox
from sqlsaber_sandbox.config import SandboxConfig
from sqlsaber_sandbox.result import AnalysisResult
from sqlsaber_sandbox.tools import AnalyzeSandboxTool, prepare_analysis


def _context(model: Any, *, resolver: Any = None, store: Any = None) -> Any:
    return SimpleNamespace(
        query_result_store=object(),
        workspace_input_resolver=resolver,
        artifact_store=store,
        artifact_failure_mode="required",
        resolve_subagent_model=lambda *a, **k: SimpleNamespace(
            model=model, model_name="scripted", provider="test"
        ),
    )


def _ctx(conversation: str = "c1", call: str = "call") -> Any:
    return RunContext(
        deps=None,
        model=FunctionModel(
            lambda messages, info: ModelResponse(parts=[TextPart("unused")])
        ),
        messages=[],
        tool_call_id=call,
        usage=RunUsage(),
        max_retries=1,
        conversation_id=conversation,
        run_id="r1",
        metadata={},
    )


def _result(session_id: str = "ss_1") -> AnalysisResult:
    return AnalysisResult(session_id, "sa_1", "answer", (), (), b"notebook")


async def test_managed_tool_accepts_goal_not_code_and_attachment_schema_is_conditional() -> (
    None
):
    plain = Sandbox(
        _context(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("x")])))
    )
    attached = Sandbox(
        _context(
            FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("x")])),
            resolver=object(),
        )
    )
    plain_tools = await plain.get_toolset().get_tools(_ctx())
    attached_tools = await attached.get_toolset().get_tools(_ctx())
    plain_schema = plain_tools["analyze_in_sandbox"].tool_def.parameters_json_schema[
        "properties"
    ]
    attached_schema = attached_tools[
        "analyze_in_sandbox"
    ].tool_def.parameters_json_schema["properties"]
    assert "goal" in plain_schema and "code" not in plain_schema
    assert "attachment_refs" not in plain_schema
    assert "attachment_refs" in attached_schema
    assert plain_tools["close_sandbox"].tool_def.sequential is True


async def test_cross_conversation_session_id_is_rejected_before_workspace_lookup(
    monkeypatch,
) -> None:
    tool = AnalyzeSandboxTool(_context(object()), SandboxConfig())
    session = SimpleNamespace(id="ss_private", lost=False, closed=False)
    tool.sessions[session.id] = ("owner", session)
    looked_up = False

    async def workspace(*args, **kwargs):
        nonlocal looked_up
        looked_up = True
        raise AssertionError("must not resolve remote inputs")

    monkeypatch.setattr(
        "sqlsaber_sandbox.tools.build_workspace_from_history", workspace
    )
    returned = await tool.execute(_ctx("attacker"), "steal", session_id=session.id)
    assert (
        returned.return_value["error"]
        == "Sandbox session is unavailable in this conversation"
    )
    assert looked_up is False


async def test_same_config_is_passed_to_new_session(monkeypatch) -> None:
    config = SandboxConfig(max_output_chars=123)
    backend = object()
    tool = AnalyzeSandboxTool(
        _context(object()), config, backend_factory=lambda: backend
    )
    captured: dict[str, Any] = {}

    class Session:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.id = "ss_new"
            self.lost = False
            self.closed = False

        async def analyze(self, *args, **kwargs):
            return _result(self.id)

    async def workspace(*args, **kwargs):
        from sqlsaber_sandbox.result import Workspace

        return Workspace()

    monkeypatch.setattr("sqlsaber_sandbox.tools.SandboxSession", Session)
    monkeypatch.setattr(
        "sqlsaber_sandbox.tools.build_workspace_from_history", workspace
    )
    await tool.execute(_ctx(), "goal")
    assert captured["config"] is config
    assert captured["backend"] is backend


async def test_context_refresh_keeps_existing_sessions() -> None:
    first, second = _context(object()), _context(object())
    capability = Sandbox(first)
    marker = ("c1", object())
    capability.tool.sessions["ss_existing"] = marker
    capability.update_context(second)
    assert capability.tool.context is second
    assert capability.tool.sessions["ss_existing"] is marker


async def test_publication_failure_retains_result_and_retry_does_not_analyze(
    monkeypatch,
) -> None:
    store = object()
    tool = AnalyzeSandboxTool(_context(object(), store=store), SandboxConfig())
    result = _result()
    calls = 0

    class Session:
        id = "ss_1"
        lost = False
        closed = False

        async def analyze(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return result

        async def snapshot(self):
            raise AssertionError("completed result should be retained")

    session = Session()
    tool.sessions[session.id] = ("c1", session)

    async def workspace(*args, **kwargs):
        from sqlsaber_sandbox.result import Workspace

        return Workspace()

    publish_calls = 0

    async def publish(*args, **kwargs):
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise RuntimeError("store down")
        return SimpleNamespace(to_metadata=lambda: {}, artifacts=[])

    monkeypatch.setattr(
        "sqlsaber_sandbox.tools.build_workspace_from_history", workspace
    )
    monkeypatch.setattr("sqlsaber_sandbox.tools.publish_analysis", publish)
    first = await tool.execute(_ctx(), "goal", session_id=session.id)
    second = await tool.publish_artifacts(_ctx(), session.id)
    assert first.return_value["publication_state"] == "failed"
    assert second.return_value["publication_state"] == "published"
    assert calls == 1 and tool.results[session.id] is result


async def test_close_retries_publication_before_release_without_duplicate_bundles(
    monkeypatch,
) -> None:
    events = []

    class Store(InMemoryArtifactStore):
        async def publish(self, bundle, *, context):
            events.append("publish")
            if len(events) == 1:
                raise RuntimeError("storage unavailable")
            return await super().publish(bundle, context=context)

    tool = AnalyzeSandboxTool(_context(object(), store=Store()), SandboxConfig())
    result = _result()
    session = SimpleNamespace(
        id=result.session_id,
        lost=False,
        closed=False,
        analyze=AsyncMock(return_value=result),
        snapshot=AsyncMock(side_effect=AssertionError("must reuse downloaded bytes")),
    )

    async def close():
        events.append("close")
        session.closed = True

    session.close = AsyncMock(side_effect=close)
    tool.sessions[session.id] = ("c1", session)
    monkeypatch.setattr(
        "sqlsaber_sandbox.tools.build_workspace_from_history",
        AsyncMock(return_value=None),
    )
    analyzed = await tool.execute(_ctx(), "goal", session_id=session.id)
    assert analyzed.return_value["publication_state"] == "failed"

    closed = await tool.close_session(_ctx(), session.id)
    assert events == ["publish", "publish", "close"]
    assert closed.return_value["publication_state"] == "published"
    assert closed.return_value["session_state"] == "closed"
    assert closed.return_value["artifacts"][0]["name"] == "analysis.ipynb"
    assert "artifact_publication" in closed.metadata

    repeated = await tool.close_session(_ctx(), session.id)
    assert repeated.metadata == closed.metadata
    assert events.count("publish") == 2
    session.analyze.assert_awaited_once()
    session.snapshot.assert_not_awaited()


@pytest.mark.parametrize("failure_mode", ["required", "best_effort"])
async def test_close_preserves_failed_publication_for_retry(failure_mode) -> None:
    store = InMemoryArtifactStore()
    publish = AsyncMock(side_effect=RuntimeError("private storage details"))
    context = _context(object(), store=SimpleNamespace(publish=publish))
    context.artifact_failure_mode = failure_mode
    tool = AnalyzeSandboxTool(context, SandboxConfig())
    result = _result()
    session = SimpleNamespace(id=result.session_id, closed=False)

    async def close():
        session.closed = True

    session.close = AsyncMock(side_effect=close)
    tool.sessions[session.id] = ("c1", session)
    tool.results[session.id] = result

    returned = await tool.close_session(_ctx(), session.id)
    assert tool.results[session.id] is result
    assert returned.return_value["publication_state"] == "failed"
    assert "private storage details" not in str(returned.return_value)
    assert "private storage details" not in str(returned.metadata)
    if failure_mode == "required":
        session.close.assert_not_awaited()
        assert returned.return_value["session_state"] == "ready"
        assert "error" in returned.return_value
    else:
        session.close.assert_awaited_once()
        assert returned.return_value["session_state"] == "closed"
        assert "artifact_error" in returned.metadata

    context.artifact_store = store
    retried = await tool.publish_artifacts(_ctx(), session.id)
    assert retried.return_value["publication_state"] == "published"
    assert retried.return_value["session_state"] == (
        "ready" if failure_mode == "required" else "closed"
    )


@pytest.mark.parametrize(
    "failure_mode, recovers",
    [("required", True), ("required", False), ("best_effort", False)],
)
async def test_capability_shutdown_retries_with_original_scope_and_releases_resources(
    failure_mode,
    recovers,
) -> None:
    contexts = []

    class Store(InMemoryArtifactStore):
        async def publish(self, bundle, *, context):
            contexts.append(context)
            if len(contexts) == 1 or not recovers:
                raise RuntimeError("storage unavailable")
            return await super().publish(bundle, context=context)

    context = _context(object(), store=Store())
    context.artifact_failure_mode = failure_mode
    capability = Sandbox(context)
    tool = capability.tool
    result = _result()
    session = SimpleNamespace(id=result.session_id, closed=False)

    async def close():
        session.closed = True

    session.close = AsyncMock(side_effect=close)
    unused = SimpleNamespace(id="ss_unused", close=AsyncMock())
    tool.sessions[session.id] = ("c1", session)
    tool.sessions[unused.id] = ("c2", unused)
    tool.results[session.id] = result
    ctx = _ctx()
    ctx.metadata = {"tenant": "owner"}
    await tool._publish(ctx, result)

    if failure_mode == "required" and not recovers:
        with pytest.raises(ExceptionGroup, match="Sandbox cleanup failed") as error:
            await capability.close()
        assert len(error.value.exceptions) == 1
        assert isinstance(error.value.exceptions[0], ArtifactPublicationError)
    else:
        await capability.close()

    assert len(contexts) == 2
    assert contexts[1] == contexts[0]
    assert contexts[1].conversation_id == "c1"
    assert contexts[1].metadata == {"tenant": "owner"}
    session.close.assert_awaited_once()
    unused.close.assert_awaited_once()
    if not recovers:
        assert tool.results[session.id] is result
        context.artifact_store = InMemoryArtifactStore()
        await capability.close()
    assert not tool.results


async def test_close_without_a_completed_analysis_does_not_start_a_sandbox() -> None:
    tool = AnalyzeSandboxTool(_context(object(), store=object()), SandboxConfig())
    session = SimpleNamespace(id="ss_unused", close=AsyncMock(), snapshot=AsyncMock())
    tool.sessions[session.id] = ("c1", session)
    await tool.close_session(_ctx(), session.id)
    session.close.assert_awaited_once()
    session.snapshot.assert_not_awaited()


@pytest.mark.parametrize(
    "limits, expected",
    [
        (None, False),
        (UsageLimits(), True),
        (UsageLimits(request_limit=2), True),
        (UsageLimits(tool_calls_limit=2), True),
    ],
)
async def test_prepare_analysis_sequential_only_for_explicit_finite_budgets(
    limits, expected
) -> None:
    definition = ToolDefinition(name="x", description="x", parameters_json_schema={})
    with bind_usage_limits(limits):
        prepared = await prepare_analysis(_ctx(), definition)
    assert prepared.sequential is expected


async def test_parent_agent_concurrent_calls_with_explicit_budget_cannot_overshoot(
    monkeypatch,
) -> None:
    """Exercise the real parent Agent/tool prepare path, not direct tool invocation."""
    calls = 0

    class Session:
        id = "ss"
        lost = False
        closed = False

        async def analyze(self, *args, parent_usage, **kwargs):
            nonlocal calls
            # Child request admission must share the parent's accumulator.
            UsageLimits(request_limit=2).check_before_request(parent_usage)
            calls += 1
            parent_usage.requests += 1
            return _result(self.id)

    async def workspace(*args, **kwargs):
        from sqlsaber_sandbox.result import Workspace

        return Workspace()

    model_calls = 0

    def respond(messages, info):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("analyze_in_sandbox", {"goal": "a"}, tool_call_id="a"),
                    ToolCallPart("analyze_in_sandbox", {"goal": "b"}, tool_call_id="b"),
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    capability = Sandbox(_context(object()))
    monkeypatch.setattr(
        "sqlsaber_sandbox.tools.build_workspace_from_history", workspace
    )
    monkeypatch.setattr(
        "sqlsaber_sandbox.tools.SandboxSession", lambda **kwargs: Session()
    )
    agent = Agent(FunctionModel(respond), capabilities=[capability])
    with bind_usage_limits(UsageLimits(request_limit=2)):
        with pytest.raises(UsageLimitExceeded):
            await agent.run("go", usage_limits=UsageLimits(request_limit=2))
    assert calls <= 1


def test_renderer_shows_answer_artifacts_and_error_without_raw_dict():
    from sqlsaber.render import md_of

    tool = AnalyzeSandboxTool(_context(object()), SandboxConfig())
    blocks = tool.render_result(
        {
            "answer": "Forecast is **32**.",
            "session_id": "ss_example",
            "session_state": "ready",
            "publication_state": "published",
            "artifacts": [{"name": "files/weights.npz", "size": 512}],
        }
    )
    assert blocks is not None
    rendered = md_of(blocks)
    assert "Forecast is **32**" in rendered
    assert "files/weights.npz" in rendered and "512" in rendered
    assert "ss_example" in rendered and "{'answer'" not in rendered
    failed = tool.render_result(
        {"error": "Kernel lost\x1b[31m", "session_state": "lost"}
    )
    assert failed is not None
    assert "Error" in md_of(failed) and "\x1b" not in md_of(failed)

"""Tests for the public SQL tools capability."""

from dataclasses import dataclass

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from sqlsaber.capabilities import SqlTools
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.sqlite import SQLiteConnection


def _registry(name: str = "test") -> DatabaseRegistry:
    connection = SQLiteConnection("sqlite:///:memory:")
    return DatabaseRegistry(
        [
            DatabaseEntry.from_connection(
                name=name,
                connection=connection,
                description=None,
                excluded_schemas=[],
            )
        ]
    )


@dataclass
class CustomDeps:
    tenant: str


@pytest.mark.asyncio
async def test_sql_tools_compose_with_custom_agent_deps() -> None:
    registry = _registry()
    capability = SqlTools(registry=registry)
    model = TestModel(call_tools=["list_tables"])
    agent = Agent(model, deps_type=CustomDeps, capabilities=[capability])

    result = await agent.run("List tables", deps=CustomDeps(tenant="acme"))

    assert {
        tool.name for tool in model.last_model_request_parameters.function_tools
    } == {
        "list_tables",
        "introspect_schema",
        "execute_sql",
    }
    assert any(
        isinstance(part, ToolReturnPart) and part.tool_name == "list_tables"
        for message in result.all_messages()
        for part in message.parts
    )


@pytest.mark.asyncio
async def test_owned_registry_closes_with_agent_context(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "owned.sqlite"
    database_path.touch()
    capability = SqlTools(database=str(database_path))
    connection = capability.registry.get(capability.registry.primary()).connection
    close_calls = 0
    original_close = connection.close

    async def tracked_close() -> None:
        nonlocal close_calls
        close_calls += 1
        await original_close()

    monkeypatch.setattr(connection, "close", tracked_close)
    agent = Agent(TestModel(call_tools=[]), capabilities=[capability])

    async with agent:
        await agent.run("Hello")

    assert close_calls == 1


@pytest.mark.asyncio
async def test_owned_registry_stays_open_across_nested_agent_runs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "nested.sqlite"
    database_path.touch()
    capability = SqlTools(database=str(database_path))
    connection = capability.registry.get(capability.registry.primary()).connection
    lifecycle_events: list[str] = []

    async def tracked_get_pool() -> str:
        lifecycle_events.append("open")
        return str(database_path)

    async def tracked_close() -> None:
        lifecycle_events.append("close")

    monkeypatch.setattr(connection, "get_pool", tracked_get_pool)
    monkeypatch.setattr(connection, "close", tracked_close)
    agent = Agent(TestModel(call_tools=[]), capabilities=[capability])

    async with agent:
        assert lifecycle_events == ["open"]
        await agent.run("First query")
        await agent.run("Second query")
        assert lifecycle_events == ["open"]

    assert lifecycle_events == ["open", "close"]


@pytest.mark.asyncio
async def test_partial_owned_registry_initialization_is_cleaned_up(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_path = tmp_path / "first.sqlite"
    second_path = tmp_path / "second.sqlite"
    first_path.touch()
    second_path.touch()
    capability = SqlTools(database=[str(first_path), str(second_path)])
    first, second = list(capability.registry)
    lifecycle_events: list[str] = []

    class PoolInitializationError(RuntimeError):
        pass

    init_error = PoolInitializationError()

    async def open_first() -> str:
        lifecycle_events.append("open:first")
        return str(first_path)

    async def fail_second() -> str:
        lifecycle_events.append("open:second")
        raise init_error

    async def close_first() -> None:
        lifecycle_events.append("close:first")

    async def close_second() -> None:
        lifecycle_events.append("close:second")

    monkeypatch.setattr(first.connection, "get_pool", open_first)
    monkeypatch.setattr(first.connection, "close", close_first)
    monkeypatch.setattr(second.connection, "get_pool", fail_second)
    monkeypatch.setattr(second.connection, "close", close_second)
    agent = Agent(TestModel(call_tools=[]), capabilities=[capability])

    with pytest.raises(PoolInitializationError) as caught:
        async with agent:
            pass

    assert caught.value is init_error
    assert lifecycle_events == [
        "open:first",
        "open:second",
        "close:first",
        "close:second",
    ]


@pytest.mark.asyncio
async def test_owned_registry_cleanup_errors_propagate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "cleanup-error.sqlite"
    database_path.touch()
    capability = SqlTools(database=str(database_path))
    connection = capability.registry.get(capability.registry.primary()).connection

    async def failing_close() -> None:
        raise RuntimeError("database cleanup failed")

    monkeypatch.setattr(connection, "close", failing_close)
    agent = Agent(TestModel(call_tools=[]), capabilities=[capability])

    with pytest.raises(RuntimeError, match="database cleanup failed"):
        async with agent:
            await agent.run("Hello")


@pytest.mark.asyncio
async def test_borrowed_registry_is_not_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    connection = registry.get(registry.primary()).connection
    close_calls = 0

    async def tracked_close() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(connection, "close", tracked_close)
    agent = Agent(TestModel(call_tools=[]), capabilities=[SqlTools(registry=registry)])

    async with agent:
        await agent.run("Hello")

    assert close_calls == 0

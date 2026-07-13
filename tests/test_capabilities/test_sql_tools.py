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

    assert {tool.name for tool in model.last_model_request_parameters.function_tools} == {
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

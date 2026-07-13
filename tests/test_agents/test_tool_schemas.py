"""Regression snapshots for SQLSaber's model-visible function tools."""

import json
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.sqlite import SQLiteConnection

_SNAPSHOT_DIR = Path(__file__).with_name("snapshots")


def _registry(*names: str) -> DatabaseRegistry:
    return DatabaseRegistry(
        [
            DatabaseEntry.from_connection(
                name=name,
                connection=SQLiteConnection("sqlite:///:memory:"),
                description=None,
                excluded_schemas=[],
            )
            for name in names
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("names", "snapshot_name"),
    [
        (("solo",), "tool_schemas_single.json"),
        (("prod", "staging"), "tool_schemas_multi.json"),
    ],
)
async def test_tool_schema_snapshot(names: tuple[str, ...], snapshot_name: str) -> None:
    agent = SQLSaberAgent(
        registry=_registry(*names),
        model_name="anthropic:snapshot-model",
        api_key="test-key",
    )
    model = TestModel(call_tools=[])

    try:
        with agent.agent.override(model=model):
            await agent.run("Capture the tool schemas")

        actual = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_json_schema": tool.parameters_json_schema,
            }
            for tool in model.last_model_request_parameters.function_tools
        ]
        expected = json.loads((_SNAPSHOT_DIR / snapshot_name).read_text())
        assert actual == expected
    finally:
        await agent.close()

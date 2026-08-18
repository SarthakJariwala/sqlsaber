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
    ("names", "snapshot_name", "with_workspace_input_resolver"),
    [
        (("solo",), "tool_schemas_single.json", False),
        (("prod", "staging"), "tool_schemas_multi.json", False),
        (("solo",), "tool_schemas_single_with_attachments.json", True),
    ],
)
async def test_tool_schema_snapshot(
    names: tuple[str, ...],
    snapshot_name: str,
    with_workspace_input_resolver: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for variable in (
        "DAYTONA_API_KEY",
        "E2B_API_KEY",
        "SPRITES_TOKEN",
        "HOPX_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "CLOUDFLARE_SANDBOX_BASE_URL",
        "CLOUDFLARE_API_TOKEN",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / "missing-modal.toml"))

    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return []

    agent = SQLSaberAgent(
        registry=_registry(*names),
        model_name="anthropic:snapshot-model",
        api_key="test-key",
        workspace_input_resolver=(
            Resolver() if with_workspace_input_resolver else None
        ),
    )
    model = TestModel(call_tools=[])

    try:
        with agent.agent.override(model=model):
            await agent.run("Capture the tool schemas")

        request_parameters = model.last_model_request_parameters
        assert request_parameters is not None
        actual = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_json_schema": tool.parameters_json_schema,
            }
            for tool in request_parameters.function_tools
        ]
        expected = json.loads((_SNAPSHOT_DIR / snapshot_name).read_text())
        assert actual == expected
    finally:
        await agent.close()

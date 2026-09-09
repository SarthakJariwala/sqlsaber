"""CLI session options opt into installed capability plugins."""

from __future__ import annotations

import pytest

from sqlsaber import SQLSaber, SqlTools
from sqlsaber.cli.session import cli_sqlsaber_options
from sqlsaber.config.settings import Config
from sqlsaber.query_results import InMemoryQueryResultStore


def _settings() -> Config:
    return Config.in_memory(
        model_name="anthropic:claude-3-5-sonnet",
        api_keys={"anthropic": "test-key"},
    )


@pytest.mark.asyncio
async def test_cli_session_loads_installed_plugins_on_the_session_store() -> None:
    store = InMemoryQueryResultStore()
    saber = SQLSaber(
        options=cli_sqlsaber_options(
            database="sqlite:///:memory:",
            settings=_settings(),
            query_result_store=store,
        )
    )
    try:
        ids = {
            getattr(capability, "id", None) for capability in saber.agent.capabilities
        }
        assert "notebook" in ids
        assert "viz" in ids
        sql = next(c for c in saber.agent.capabilities if isinstance(c, SqlTools))
        notebook = next(
            c for c in saber.agent.capabilities if getattr(c, "id", None) == "notebook"
        )
        viz = next(
            c for c in saber.agent.capabilities if getattr(c, "id", None) == "viz"
        )
        assert sql.query_result_store is store
        assert notebook.tool._context.query_result_store is store
        assert viz.tool.query_result_store is store
        assert "analyze_data" in saber.agent._tools
        assert "viz" in saber.agent._tools
    finally:
        await saber.close()

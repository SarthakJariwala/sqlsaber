"""Tests for SQLSaberAgent overrides and lifecycle behavior."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore
from sqlsaber.overrides import ToolRunDeps
from sqlsaber.tools.knowledge_tool import SearchKnowledgeTool


class StructuredResult(BaseModel):
    value: str


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite connection for testing."""
    return SQLiteConnection("sqlite:///:memory:")


class TestSQLSaberAgentOverrides:
    """Test validation logic for model_name and api_key overrides."""

    def test_api_key_without_model_name_raises_error(self, in_memory_db):
        """api_key requires model_name to be specified."""
        with pytest.raises(ValueError):
            SQLSaberAgent(db_connection=in_memory_db, api_key="test-key")

    def test_model_name_and_api_key_together_accepted(self, in_memory_db):
        """Both model_name and api_key together should work."""
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )
        assert agent is not None
        assert agent.agent is not None
        assert agent.agent.model.model_name == "claude-3-5-sonnet"


class TestSQLSaberAgentKnowledge:
    def test_knowledge_tool_context_configured(self, in_memory_db):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )

        tool = agent._tools.get("search_knowledge")
        assert isinstance(tool, SearchKnowledgeTool)
        assert tool.database_name == "test-db"
        assert tool.knowledge_manager is agent.knowledge_manager


def test_sqlsaber_agent_passes_output_type_to_pydantic_agent(in_memory_db):
    agent = SQLSaberAgent(
        db_connection=in_memory_db,
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        output_type=StructuredResult,
    )

    assert agent.agent.output_type is StructuredResult


class TestSQLSaberAgentDeps:
    @pytest.mark.asyncio
    async def test_run_passes_tool_overides_via_deps(self, in_memory_db, monkeypatch):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            tool_overides={
                "viz": {
                    "model_name": "openai:gpt-5-mini",
                    "api_key": "override-api-key",
                }
            },
        )

        captured: dict[str, object] = {}

        async def fake_run(prompt: str, **kwargs):
            _ = prompt
            captured.update(kwargs)
            return SimpleNamespace(output="ok")

        monkeypatch.setattr(agent.agent, "run", fake_run)

        await agent.run("hello")
        deps = captured.get("deps")
        assert isinstance(deps, ToolRunDeps)
        assert deps.tool_overides["viz"].model_name == "openai:gpt-5-mini"
        assert deps.tool_overides["viz"].api_key == "override-api-key"

    @pytest.mark.asyncio
    async def test_run_passes_empty_tool_overides_when_unset(
        self, in_memory_db, monkeypatch
    ):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )

        captured: dict[str, object] = {}

        async def fake_run(prompt: str, **kwargs):
            _ = prompt
            captured.update(kwargs)
            return SimpleNamespace(output="ok")

        monkeypatch.setattr(agent.agent, "run", fake_run)

        await agent.run("hello")
        deps = captured.get("deps")
        assert isinstance(deps, ToolRunDeps)
        assert deps.tool_overides == {}

    @pytest.mark.asyncio
    async def test_sqlsaber_agent_forwards_usage_to_pydantic_run(
        self, in_memory_db, monkeypatch
    ):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            tool_overides={
                "viz": {
                    "model_name": "openai:gpt-5-mini",
                    "api_key": "override-api-key",
                }
            },
        )
        usage = object()
        captured: dict[str, object] = {}

        async def fake_run(prompt: str, **kwargs):
            _ = prompt
            captured.update(kwargs)
            return SimpleNamespace(output="ok")

        monkeypatch.setattr(agent.agent, "run", fake_run)

        await agent.run("hello", usage=usage)

        assert captured["usage"] is usage
        deps = captured.get("deps")
        assert isinstance(deps, ToolRunDeps)
        assert deps.tool_overides["viz"].model_name == "openai:gpt-5-mini"


class TestSQLSaberAgentLifecycle:
    @pytest.mark.asyncio
    async def test_close_does_not_close_injected_knowledge_manager(
        self, in_memory_db, temp_dir, monkeypatch
    ):
        manager = KnowledgeManager(
            store=SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
        )
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            knowledge_manager=manager,
        )

        close_calls = 0

        async def _track_close() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(manager, "close", _track_close)
        await agent.close()

        assert close_calls == 0

    @pytest.mark.asyncio
    async def test_close_closes_owned_knowledge_manager_once(
        self, in_memory_db, monkeypatch
    ):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )

        close_calls = 0

        async def _track_close() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(agent.knowledge_manager, "close", _track_close)

        await agent.close()
        await agent.close()

        assert close_calls == 1

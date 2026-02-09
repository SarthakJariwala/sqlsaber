"""Tests for SQLSaberAgent overrides and memory injection."""

from types import SimpleNamespace

import pytest

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.memory.manager import MemoryManager
from sqlsaber.overrides import ToolRunDeps


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


class TestSQLSaberAgentMemory:
    def test_memory_override_supersedes_saved_memories(
        self, in_memory_db, temp_dir, monkeypatch
    ):
        config_dir = temp_dir / "config"
        monkeypatch.setattr(
            "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
        )

        memory_manager = MemoryManager()
        memory_manager.add_memory("test-db", "saved-memory")

        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            memory_manager=memory_manager,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            memory="override-memory",
        )

        prompt = agent.system_prompt_text(include_memory=True)
        assert "override-memory" in prompt
        assert "saved-memory" not in prompt

    def test_memory_override_empty_disables_saved_memories(
        self, in_memory_db, temp_dir, monkeypatch
    ):
        config_dir = temp_dir / "config"
        monkeypatch.setattr(
            "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
        )

        memory_manager = MemoryManager()
        memory_manager.add_memory("test-db", "saved-memory")

        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            memory_manager=memory_manager,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            memory="",
        )

        prompt = agent.system_prompt_text(include_memory=True)
        assert "saved-memory" not in prompt


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

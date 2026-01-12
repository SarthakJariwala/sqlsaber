"""Tests for SQLSaberAgent overrides and memory injection."""

import pytest

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.memory.manager import MemoryManager


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

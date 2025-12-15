"""Tests for SQLSaberAgent model_name and api_key override functionality."""

import pytest

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.database.sqlite import SQLiteConnection


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
        # Should not raise - both overrides provided
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )
        assert agent is not None
        assert agent.agent is not None
        assert agent.agent.model.model_name == "claude-3-5-sonnet"

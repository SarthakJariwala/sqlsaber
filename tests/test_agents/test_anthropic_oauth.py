"""Tests for OAuth integration in AnthropicSQLAgent."""

from unittest.mock import Mock, patch

from sqlsaber.agents.anthropic import AnthropicSQLAgent
from sqlsaber.database.connection import PostgreSQLConnection


class TestAnthropicSQLAgentOAuth:
    """Test OAuth integration in AnthropicSQLAgent."""

    @patch("sqlsaber.agents.anthropic.Config")
    @patch("sqlsaber.agents.base.SchemaManager")
    def test_oauth_system_prompt_injection(
        self, mock_schema_manager, mock_config_class
    ):
        """Test that OAuth clients get Claude Code system prompt."""

        # Mock config to return OAuth token
        mock_config = Mock()
        mock_config.oauth_token = "test-oauth-token"
        mock_config.api_key = None
        mock_config.model_name = "anthropic:claude-3-5-sonnet-20241022"
        mock_config_class.return_value = mock_config

        # Mock database connection with proper type
        mock_db = Mock(spec=PostgreSQLConnection)
        mock_db.database_type = "postgresql"

        # Create agent (this will trigger system prompt building)
        agent = AnthropicSQLAgent(mock_db, "test_db")

        # Verify OAuth client was created
        assert agent.client.use_oauth is True
        assert agent.client.oauth_token == "test-oauth-token"

        # Verify system prompt includes Claude Code identity ONLY
        assert (
            agent.system_prompt
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )

        # Verify SQL instructions are available separately
        sql_instructions = agent._get_sql_assistant_instructions()
        assert "SQL assistant" in sql_instructions
        assert "Your responsibilities:" in sql_instructions

    @patch("sqlsaber.agents.anthropic.Config")
    @patch("sqlsaber.agents.base.SchemaManager")
    def test_api_key_system_prompt_no_injection(
        self, mock_schema_manager, mock_config_class
    ):
        """Test that API key clients don't get Claude Code system prompt."""

        # Mock config to return API key
        mock_config = Mock()
        mock_config.oauth_token = None
        mock_config.api_key = "test-api-key"
        mock_config.model_name = "anthropic:claude-3-5-sonnet-20241022"
        mock_config_class.return_value = mock_config

        # Mock database connection with proper type
        mock_db = Mock(spec=PostgreSQLConnection)
        mock_db.database_type = "postgresql"

        # Create agent (this will trigger system prompt building)
        agent = AnthropicSQLAgent(mock_db, "test_db")

        # Verify API key client was created
        assert agent.client.use_oauth is False
        assert agent.client.api_key == "test-api-key"

        # Verify system prompt does NOT include Claude Code identity
        assert (
            "You are Claude Code, Anthropic's official CLI for Claude"
            not in agent.system_prompt
        )
        assert "You are also a helpful SQL assistant" in agent.system_prompt

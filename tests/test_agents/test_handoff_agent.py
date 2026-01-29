"""Tests for the HandoffAgent."""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from sqlsaber.agents.handoff_agent import HandoffAgent


def _create_agent_instance():
    """Create a HandoffAgent instance without full initialization."""
    agent = HandoffAgent.__new__(HandoffAgent)
    agent.config = None
    agent.agent = None
    return agent


class TestHandoffAgentFormatHistory:
    """Tests for HandoffAgent._format_history_for_prompt method."""

    def test_format_empty_history(self):
        """Test formatting empty history."""
        agent = _create_agent_instance()

        result = agent._format_history_for_prompt([])
        assert result == "(No conversation history)"

    def test_format_user_messages(self):
        """Test formatting user messages."""
        agent = _create_agent_instance()

        history = [ModelRequest(parts=[UserPromptPart(content="Show me all tables")])]

        result = agent._format_history_for_prompt(history)
        assert "User: Show me all tables" in result

    def test_format_assistant_text_response(self):
        """Test formatting assistant text responses."""
        agent = _create_agent_instance()

        history = [ModelResponse(parts=[TextPart(content="Here are the tables...")])]

        result = agent._format_history_for_prompt(history)
        assert "Assistant: Here are the tables..." in result

    def test_format_includes_full_long_responses(self):
        """Test that long assistant responses are included in full."""
        agent = _create_agent_instance()

        long_content = "x" * 600
        history = [ModelResponse(parts=[TextPart(content=long_content)])]

        result = agent._format_history_for_prompt(history)
        assert long_content in result

    def test_format_includes_all_messages(self):
        """Test that all messages are included."""
        agent = _create_agent_instance()

        history = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")])
            for i in range(50)
        ]

        result = agent._format_history_for_prompt(history)
        assert "Message 0" in result
        assert "Message 25" in result
        assert "Message 49" in result

    def test_format_includes_tool_calls_with_args(self):
        """Test that tool calls include their arguments."""
        agent = _create_agent_instance()

        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="execute_sql",
                        args={"query": "SELECT * FROM users"},
                        tool_call_id="call_123",
                    )
                ]
            )
        ]

        result = agent._format_history_for_prompt(history)
        assert "[Tool Call - execute_sql]" in result
        assert "SELECT * FROM users" in result

    def test_format_includes_tool_results(self):
        """Test that tool results are included."""
        agent = _create_agent_instance()

        history = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="execute_sql",
                        content='[{"id": 1, "name": "Alice"}]',
                        tool_call_id="call_123",
                    )
                ]
            )
        ]

        result = agent._format_history_for_prompt(history)
        assert "[Tool Result - execute_sql]" in result
        assert "Alice" in result

    def test_format_truncates_long_tool_results(self):
        """Test that long tool results are truncated."""
        agent = _create_agent_instance()

        long_result = "row " * 500
        history = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="execute_sql",
                        content=long_result,
                        tool_call_id="call_123",
                    )
                ]
            )
        ]

        result = agent._format_history_for_prompt(history)
        assert "...(truncated)" in result

    def test_format_handles_json_string_args(self):
        """Test that JSON string args are parsed and included."""
        agent = _create_agent_instance()

        history = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="execute_sql",
                        args='{"query": "SELECT COUNT(*) FROM orders"}',
                        tool_call_id="call_456",
                    )
                ]
            )
        ]

        result = agent._format_history_for_prompt(history)
        assert "[Tool Call - execute_sql]" in result
        assert "SELECT COUNT(*) FROM orders" in result

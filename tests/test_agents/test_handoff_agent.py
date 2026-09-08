"""Tests for the HandoffAgent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from sqlsaber.agents.handoff_agent import HandoffAgent
from sqlsaber.prompts.handoff import HANDOFF_INPUT_INSTRUCTIONS


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
        assert result == "[User]: Show me all tables"

    def test_format_assistant_text_response(self):
        """Test formatting assistant text responses."""
        agent = _create_agent_instance()

        history = [ModelResponse(parts=[TextPart(content="Here are the tables...")])]

        result = agent._format_history_for_prompt(history)
        assert result == "[Assistant]: Here are the tables..."

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
        assert result == (
            '[Assistant tool call - execute_sql]: query="SELECT * FROM users"'
        )

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
        assert result == '[Tool result - execute_sql]: [{"id": 1, "name": "Alice"}]'

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
        assert result == (
            '[Assistant tool call - execute_sql]: query="SELECT COUNT(*) FROM orders"'
        )


async def test_generate_draft_sends_labeled_transcript_and_instructions():
    agent = _create_agent_instance()
    agent.agent = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(output="  Draft handoff\n"))
    )
    history = [
        ModelRequest(parts=[UserPromptPart(content="Count active users")]),
        ModelResponse(
            parts=[
                TextPart(content="Checking active users."),
                ToolCallPart(
                    tool_name="execute_sql",
                    args={
                        "query": "SELECT COUNT(*) FROM users WHERE status = 'active'"
                    },
                ),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name="execute_sql", content="42")]),
    ]

    result = await agent.generate_draft(history, "Compare inactive users")

    agent.agent.run.assert_awaited_once_with(
        "<source_conversation>\n"
        "[User]: Count active users\n\n"
        "[Assistant]: Checking active users.\n\n"
        '[Assistant tool call - execute_sql]: query="SELECT COUNT(*) FROM users '
        "WHERE status = 'active'\"\n\n"
        "[Tool result - execute_sql]: 42\n"
        "</source_conversation>\n\n"
        "<handoff_goal>\nCompare inactive users\n</handoff_goal>\n\n"
        f"{HANDOFF_INPUT_INSTRUCTIONS}\n"
    )
    assert result == "Draft handoff"

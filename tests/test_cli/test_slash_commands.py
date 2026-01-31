from unittest.mock import AsyncMock, MagicMock

import pytest

from sqlsaber.cli.slash_commands import CommandContext, SlashCommandProcessor
from sqlsaber.config.settings import ThinkingLevel


@pytest.fixture
def mock_context():
    """Fixture for a mocked CommandContext."""
    return CommandContext(
        console=MagicMock(),
        agent=MagicMock(),
        thread_manager=AsyncMock(),
        on_clear_history=MagicMock(),
    )


@pytest.fixture
def processor():
    """Fixture for SlashCommandProcessor."""
    return SlashCommandProcessor()


@pytest.mark.asyncio
async def test_process_unknown_command(processor, mock_context):
    """Test processing an unknown command."""
    result = await processor.process("hello world", mock_context)
    assert result.handled is False
    assert result.should_exit is False


@pytest.mark.asyncio
async def test_process_exit_command(processor, mock_context):
    """Test processing exit commands."""
    # Setup thread manager to return a thread ID (simulating an active thread ending)
    mock_context.thread_manager.end_current_thread.return_value = "thread-123"

    for cmd in ["/exit", "/quit", "exit", "quit", "QUIT", "EXIT", "/EXIT", "/QUIT"]:
        result = await processor.process(cmd, mock_context)

        assert result.handled is True
        assert result.should_exit is True
        mock_context.thread_manager.end_current_thread.assert_called()

        # Verify hint is printed
        mock_context.console.print.assert_called()
        args, _ = mock_context.console.print.call_args
        assert "saber threads resume thread-123" in args[0]


@pytest.mark.asyncio
async def test_process_clear_command(processor, mock_context):
    """Test processing /clear command."""
    result = await processor.process("/clear", mock_context)

    assert result.handled is True
    assert result.should_exit is False

    # Verify actions
    mock_context.on_clear_history.assert_called_once()
    mock_context.thread_manager.clear_current_thread.assert_called_once()
    mock_context.console.print.assert_called()


@pytest.mark.asyncio
async def test_process_thinking_on(processor, mock_context):
    """Test processing /thinking on command."""
    result = await processor.process("/thinking on", mock_context)

    assert result.handled is True
    assert result.should_exit is False

    mock_context.agent.set_thinking.assert_called_once_with(enabled=True)
    mock_context.console.print.assert_called()


@pytest.mark.asyncio
async def test_process_thinking_off(processor, mock_context):
    """Test processing /thinking off command."""
    result = await processor.process("/thinking off", mock_context)

    assert result.handled is True
    assert result.should_exit is False

    mock_context.agent.set_thinking.assert_called_once_with(enabled=False)
    mock_context.console.print.assert_called()


@pytest.mark.asyncio
async def test_process_thinking_no_args_shows_status(processor, mock_context):
    """Test /thinking with no args shows current status."""
    mock_context.agent.thinking_enabled = True
    mock_context.agent.thinking_level = ThinkingLevel.HIGH

    result = await processor.process("/thinking", mock_context)

    assert result.handled is True
    assert result.should_exit is False
    mock_context.console.print.assert_called()
    # Verify status message is printed
    call_args = mock_context.console.print.call_args[0][0]
    assert "enabled" in call_args
    assert "high" in call_args


@pytest.mark.asyncio
async def test_process_thinking_level_argument(processor, mock_context):
    """Test /thinking with level argument sets the level."""
    result = await processor.process("/thinking high", mock_context)

    assert result.handled is True
    mock_context.agent.set_thinking.assert_called_once_with(
        enabled=True, level=ThinkingLevel.HIGH
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level_str,expected_level",
    [
        ("minimal", ThinkingLevel.MINIMAL),
        ("low", ThinkingLevel.LOW),
        ("medium", ThinkingLevel.MEDIUM),
        ("high", ThinkingLevel.HIGH),
        ("maximum", ThinkingLevel.MAXIMUM),
    ],
)
async def test_process_thinking_all_levels(
    processor, mock_context, level_str, expected_level
):
    """Test /thinking with various level arguments."""
    result = await processor.process(f"/thinking {level_str}", mock_context)

    assert result.handled is True
    mock_context.agent.set_thinking.assert_called_once_with(
        enabled=True, level=expected_level
    )


@pytest.mark.asyncio
async def test_process_thinking_invalid_argument(processor, mock_context):
    """Test /thinking with invalid argument shows error."""
    result = await processor.process("/thinking invalid", mock_context)

    assert result.handled is True
    mock_context.agent.set_thinking.assert_not_called()
    # Verify warning is printed
    call_args = mock_context.console.print.call_args[0][0]
    assert "Invalid" in call_args


@pytest.mark.asyncio
async def test_process_thinking_disabled_status(processor, mock_context):
    """Test /thinking shows disabled status correctly."""
    mock_context.agent.thinking_enabled = False
    mock_context.agent.thinking_level = ThinkingLevel.MEDIUM

    result = await processor.process("/thinking", mock_context)

    assert result.handled is True
    call_args = mock_context.console.print.call_args[0][0]
    assert "disabled" in call_args


@pytest.mark.asyncio
async def test_process_handoff_with_goal(processor, mock_context):
    """Test /handoff with a goal returns handoff_goal in result."""
    result = await processor.process("/handoff optimize this query", mock_context)

    assert result.handled is True
    assert result.should_exit is False
    assert result.handoff_goal == "optimize this query"


@pytest.mark.asyncio
async def test_process_handoff_without_goal_shows_usage(processor, mock_context):
    """Test /handoff without goal shows usage message."""
    result = await processor.process("/handoff", mock_context)

    assert result.handled is True
    assert result.handoff_goal is None
    mock_context.console.print.assert_called()
    call_args = mock_context.console.print.call_args[0][0]
    assert "Usage" in call_args


@pytest.mark.asyncio
async def test_process_handoff_preserves_goal_case(processor, mock_context):
    """Test /handoff preserves the original case of the goal."""
    result = await processor.process(
        "/handoff Check UPPER and lower Case", mock_context
    )

    assert result.handoff_goal == "Check UPPER and lower Case"

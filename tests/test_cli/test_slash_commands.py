from unittest.mock import AsyncMock, MagicMock

import pytest

from sqlsaber.cli.slash_commands import CommandContext, SlashCommandProcessor


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

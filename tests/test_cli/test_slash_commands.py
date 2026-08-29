from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sqlsaber.cli.slash_commands import CommandContext, SlashCommandProcessor
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.render.markdown_text import md_of


def _emitted_markdown(surface: MagicMock) -> str:
    parts = [md_of(call.args) for call in surface.emit.call_args_list]
    return "\n\n".join(parts)


@pytest.fixture
def mock_context():
    saber = MagicMock()
    saber.end_thread = AsyncMock()
    saber.new_thread = AsyncMock()
    saber.info = SimpleNamespace(
        thinking=SimpleNamespace(enabled=False, level=ThinkingLevel.MEDIUM)
    )
    saber.set_thinking.return_value = SimpleNamespace(
        enabled=True, level=ThinkingLevel.MEDIUM
    )
    return CommandContext(surface=MagicMock(), saber=saber)


@pytest.fixture
def processor():
    return SlashCommandProcessor()


@pytest.mark.asyncio
async def test_process_unknown_command(processor, mock_context):
    result = await processor.process("hello world", mock_context)
    assert result.handled is False
    assert result.should_exit is False


@pytest.mark.asyncio
async def test_process_exit_command(processor, mock_context):
    mock_context.saber.end_thread.return_value = "thread-123"

    for cmd in ["/exit", "/quit", "exit", "quit", "QUIT", "EXIT", "/EXIT", "/QUIT"]:
        mock_context.surface.reset_mock()
        result = await processor.process(cmd, mock_context)

        assert result.handled is True
        assert result.should_exit is True
        mock_context.saber.end_thread.assert_awaited()
        assert "saber threads resume thread-123" in _emitted_markdown(
            mock_context.surface
        )


@pytest.mark.asyncio
async def test_process_clear_command(processor, mock_context):
    result = await processor.process("/clear", mock_context)

    assert result.handled is True
    assert result.should_exit is False

    mock_context.saber.new_thread.assert_awaited_once()
    assert "Conversation history cleared." in _emitted_markdown(mock_context.surface)


@pytest.mark.asyncio
async def test_process_settings_is_not_a_slash_command(processor, mock_context):
    result = await processor.process("/settings", mock_context)

    assert result.handled is False


@pytest.mark.asyncio
async def test_process_thinking_on(processor, mock_context):
    result = await processor.process("/thinking on", mock_context)

    assert result.handled is True
    assert result.should_exit is False

    mock_context.saber.set_thinking.assert_called_once_with(enabled=True)
    mock_context.surface.emit.assert_called()


@pytest.mark.asyncio
async def test_process_thinking_off(processor, mock_context):
    result = await processor.process("/thinking off", mock_context)

    assert result.handled is True
    assert result.should_exit is False

    mock_context.saber.set_thinking.assert_called_once_with(enabled=False)
    mock_context.surface.emit.assert_called()


@pytest.mark.asyncio
async def test_process_thinking_no_args_shows_status(processor, mock_context):
    mock_context.saber.info.thinking = SimpleNamespace(
        enabled=True, level=ThinkingLevel.HIGH
    )

    result = await processor.process("/thinking", mock_context)

    assert result.handled is True
    assert result.should_exit is False
    output = _emitted_markdown(mock_context.surface)
    assert "enabled" in output
    assert "high" in output


@pytest.mark.asyncio
async def test_process_thinking_level_argument(processor, mock_context):
    result = await processor.process("/thinking high", mock_context)

    assert result.handled is True
    mock_context.saber.set_thinking.assert_called_once_with(
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
    result = await processor.process(f"/thinking {level_str}", mock_context)

    assert result.handled is True
    mock_context.saber.set_thinking.assert_called_once_with(
        enabled=True, level=expected_level
    )


@pytest.mark.asyncio
async def test_process_thinking_invalid_argument(processor, mock_context):
    result = await processor.process("/thinking invalid", mock_context)

    assert result.handled is True
    mock_context.saber.set_thinking.assert_not_called()
    assert "Invalid" in _emitted_markdown(mock_context.surface)


@pytest.mark.asyncio
async def test_process_thinking_disabled_status(processor, mock_context):
    mock_context.saber.info.thinking = SimpleNamespace(
        enabled=False, level=ThinkingLevel.MEDIUM
    )

    result = await processor.process("/thinking", mock_context)

    assert result.handled is True
    assert "disabled" in _emitted_markdown(mock_context.surface)


@pytest.mark.asyncio
async def test_process_handoff_with_goal(processor, mock_context):
    result = await processor.process("/handoff optimize this query", mock_context)

    assert result.handled is True
    assert result.should_exit is False
    assert result.handoff_goal == "optimize this query"


@pytest.mark.asyncio
async def test_process_handoff_without_goal_shows_usage(processor, mock_context):
    result = await processor.process("/handoff", mock_context)

    assert result.handled is True
    assert result.handoff_goal is None
    assert "Usage" in _emitted_markdown(mock_context.surface)


@pytest.mark.asyncio
async def test_process_handoff_preserves_goal_case(processor, mock_context):
    result = await processor.process(
        "/handoff Check UPPER and lower Case", mock_context
    )

    assert result.handoff_goal == "Check UPPER and lower Case"

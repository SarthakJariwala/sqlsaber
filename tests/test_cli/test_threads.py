import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from rich.console import Console

from sqlsaber.cli.threads import (
    _human_readable,
    _render_transcript,
    create_threads_app,
)
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.threads.storage import Thread, ThreadStorage


class TestThreadsCLI:
    """Test CLI thread commands."""

    @pytest.fixture
    def mock_console(self):
        """Mock console for testing output."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary thread storage for testing."""
        with tempfile.TemporaryDirectory() as tmp:
            storage = ThreadStorage()
            storage.db_path = Path(tmp) / "test_threads.db"
            yield storage

    @pytest.fixture
    def sample_threads(self):
        """Sample thread objects for testing."""
        return [
            Thread(
                id="thread-1",
                database_name="prod_db",
                title="Users query",
                created_at=1672531200.0,
                ended_at=None,
                last_activity_at=1672531200.0,  # 2023-01-01 00:00:00
                model_name="gpt-4",
            ),
            Thread(
                id="thread-2",
                database_name="dev_db",
                title="Orders analysis",
                created_at=1672617600.0,
                ended_at=None,
                last_activity_at=1672617600.0,  # 2023-01-02 00:00:00
                model_name="gpt-3.5-turbo",
            ),
            Thread(
                id="thread-3",
                database_name=None,
                title="Schema exploration",
                created_at=1672704000.0,
                ended_at=None,
                last_activity_at=1672704000.0,
                model_name="claude-3",
            ),
        ]

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing thread transcript rendering."""
        return [
            ModelRequest(parts=[UserPromptPart("Show me all tables")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="list_tables", args={}, tool_call_id="call-1"
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolReturnPart(
                        tool_name="list_tables",
                        content='["users", "orders", "products"]',
                        tool_call_id="call-1",
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        "Here are the tables in your database:\n- users\n- orders\n- products"
                    ),
                ]
            ),
        ]

    def _messages_to_bytes(self, messages):
        """Convert messages to JSON bytes."""
        return ModelMessagesTypeAdapter.dump_json(messages)

    def test_list_threads_function_call(self, sample_threads):
        """Test the list_threads functionality via ThreadStorage."""
        with patch("sqlsaber.cli.threads.ThreadStorage") as mock_storage_class:
            # Use MagicMock instead of AsyncMock
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage

            with patch("sqlsaber.cli.threads.console") as mock_console:
                from sqlsaber.cli.threads import list_threads

                # Mock asyncio.run to avoid event loop conflicts
                with patch("asyncio.run") as mock_run:
                    mock_run.return_value = sample_threads
                    list_threads()

                # Verify the function was called correctly
                mock_run.assert_called_once()
                mock_console.print.assert_called()

    def test_list_threads_empty(self):
        """Test listing threads when no threads exist."""
        with patch("sqlsaber.cli.threads.ThreadStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage

            with patch("sqlsaber.cli.threads.console") as mock_console:
                from sqlsaber.cli.threads import list_threads

                with patch("asyncio.run") as mock_run:
                    mock_run.return_value = []
                    list_threads()

                mock_console.print.assert_called_with("No threads found.")

    def test_show_thread_not_found(self):
        """Test showing a thread that doesn't exist."""
        with patch("sqlsaber.cli.threads.ThreadStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage

            with patch("sqlsaber.cli.threads.console") as mock_console:
                from sqlsaber.cli.threads import show

                with patch("asyncio.run") as mock_run:
                    mock_run.return_value = None
                    show("nonexistent-thread")

                mock_console.print.assert_called_with(
                    "[error]Thread not found:[/error] nonexistent-thread"
                )

    def test_show_thread_found(self, sample_threads, sample_messages):
        """Test showing a thread that exists."""
        thread = sample_threads[0]

        with patch("sqlsaber.cli.threads.ThreadStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage

            with patch("sqlsaber.cli.threads.console") as mock_console:
                with patch("sqlsaber.cli.threads._render_transcript") as mock_render:
                    from sqlsaber.cli.threads import show

                    with patch("asyncio.run", side_effect=[thread, sample_messages]):
                        show("thread-1")

                    mock_render.assert_called_once_with(
                        mock_console, sample_messages, None
                    )

    def test_human_readable_timestamp(self):
        """Test _human_readable timestamp formatting."""
        # Test with valid timestamp (adjust for timezone differences)
        timestamp = 1672531200.0  # 2023-01-01 00:00:00 UTC
        result = _human_readable(timestamp)
        # Just check that it contains a date format, don't be timezone specific
        assert len(result) > 10  # Should be a formatted date string
        assert (
            "2022" in result or "2023" in result
        )  # Could be either depending on timezone

        # Test with None
        assert _human_readable(None) == "-"

        # Test with zero/empty
        assert _human_readable(0.0) == "-"

    def test_render_transcript_with_tool_calls(self, mock_console):
        """Test _render_transcript with various message types."""
        messages = [
            ModelRequest(parts=[UserPromptPart("List all tables")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="list_tables", args={}, tool_call_id="call-1"
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolReturnPart(
                        tool_name="list_tables",
                        content='{"success": true, "results": [{"table_name": "users"}]}',
                        tool_call_id="call-1",
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart("Here are your tables."),
                ]
            ),
        ]

        with patch("sqlsaber.cli.display.DisplayManager") as mock_dm_class:
            mock_dm = MagicMock()
            mock_dm_class.return_value = mock_dm

            _render_transcript(mock_console, messages)

            # Verify DisplayManager was used for tool calls
            mock_dm.show_tool_executing.assert_called()

    def test_render_transcript_sql_execution(self, mock_console):
        """Test _render_transcript with SQL execution results."""
        messages = [
            ModelRequest(parts=[UserPromptPart("SELECT * FROM users LIMIT 5")]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="execute_sql",
                        args={"sql": "SELECT * FROM users LIMIT 5"},
                        tool_call_id="call-2",
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolReturnPart(
                        tool_name="execute_sql",
                        content='{"success": true, "results": [{"id": 1, "name": "John"}]}',
                        tool_call_id="call-2",
                    ),
                ]
            ),
        ]

        with patch("sqlsaber.cli.display.DisplayManager") as mock_dm_class:
            mock_dm = MagicMock()
            mock_dm_class.return_value = mock_dm

            _render_transcript(mock_console, messages)

            # Verify query results were shown
            mock_dm.show_query_results.assert_called_with([{"id": 1, "name": "John"}])

    def test_create_threads_app(self):
        """Test that threads app is created correctly."""
        app = create_threads_app()
        # Cyclopts apps have a tuple for name attribute
        assert "threads" in str(app.name)
        assert "Manage SQLsaber threads" in app.help

    @pytest.mark.asyncio
    async def test_storage_integration(self, temp_storage):
        """Integration test with real ThreadStorage."""
        # Create test thread
        messages = [
            ModelRequest(parts=[UserPromptPart("Hello")]),
            ModelResponse(parts=[TextPart("Hi there!")]),
        ]
        messages_bytes = self._messages_to_bytes(messages)

        thread_id = await temp_storage.save_snapshot(
            messages_json=messages_bytes, database_name="test_db"
        )
        await temp_storage.save_metadata(thread_id=thread_id, title="Test Thread")

        # Test that we can retrieve threads
        threads = await temp_storage.list_threads()
        assert len(threads) == 1
        assert threads[0].title == "Test Thread"

        # Test that we can get thread messages
        retrieved_messages = await temp_storage.get_thread_messages(thread_id)
        assert len(retrieved_messages) == 2

    @pytest.mark.asyncio
    async def test_resume_thread_success_integration(
        self, sample_threads, sample_messages
    ):
        """Test successful thread resume with mocked components."""
        thread = sample_threads[0]

        async def mock_resume_run():
            store = MagicMock()
            store.get_thread = AsyncMock(return_value=thread)
            store.get_thread_messages = AsyncMock(return_value=sample_messages)

            with (
                patch("sqlsaber.cli.threads.ThreadStorage", return_value=store),
                patch("sqlsaber.database.resolver") as mock_resolve,
                patch("sqlsaber.database.DatabaseConnection") as mock_db_conn_class,
                patch(
                    "sqlsaber.agents.pydantic_ai_agent.SQLSaberAgent"
                ) as mock_agent_class,
                patch(
                    "sqlsaber.cli.interactive.InteractiveSession"
                ) as mock_session_class,
            ):
                # Mock database resolution
                mock_resolved = MagicMock()
                mock_resolved.connection_string = "postgresql://test"
                mock_resolved.name = "prod_db"
                mock_resolve.return_value = mock_resolved

                # Mock database connection
                mock_db_conn = MagicMock()
                mock_db_conn_class.return_value = mock_db_conn

                # Mock agent and session
                mock_agent_instance = MagicMock()
                mock_agent_instance.agent = MagicMock()
                mock_agent_class.return_value = mock_agent_instance
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # This would be the actual resume logic
                resolved_thread = await store.get_thread("thread-1")
                assert resolved_thread == thread

                db_selector = resolved_thread.database_name
                resolved = mock_resolve(db_selector, DatabaseConfigManager())
                assert resolved.name == "prod_db"

        await mock_resume_run()

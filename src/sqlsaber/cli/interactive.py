"""Interactive mode handling for the CLI."""

import asyncio
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import platformdirs
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from sqlsaber.cli.completers import (
    CompositeCompleter,
    SlashCommandCompleter,
    TableNameCompleter,
)
from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.streaming import StreamingQueryHandler
from sqlsaber.database import (
    CSVConnection,
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.database.schema import SchemaManager
from sqlsaber.threads import ThreadStorage

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent


def bottom_toolbar():
    return [
        (
            "class:bottom-toolbar",
            " Use 'Esc-Enter' or 'Meta-Enter' to submit.",
        )
    ]


style = Style.from_dict(
    {
        "frame.border": "#ebbcba",
        "bottom-toolbar": "#ebbcba bg:#21202e",
    }
)


class InteractiveSession:
    """Manages interactive CLI sessions."""

    def __init__(
        self,
        console: Console,
        sqlsaber_agent: "SQLSaberAgent",
        db_conn,
        database_name: str,
        *,
        initial_thread_id: str | None = None,
        initial_history: list | None = None,
    ):
        self.console = console
        self.sqlsaber_agent = sqlsaber_agent
        self.db_conn = db_conn
        self.database_name = database_name
        self.display = DisplayManager(console)
        self.streaming_handler = StreamingQueryHandler(console)
        self.current_task: asyncio.Task | None = None
        self.cancellation_token: asyncio.Event | None = None
        self.table_completer = TableNameCompleter()
        self.message_history: list | None = initial_history or []
        # Conversation Thread persistence
        self._threads = ThreadStorage()
        self._thread_id: str | None = initial_thread_id
        self.first_message = not self._thread_id

    def show_welcome_message(self):
        """Display welcome message for interactive mode."""
        # Show database information
        db_name = self.database_name or "Unknown"
        db_type = (
            "PostgreSQL"
            if isinstance(self.db_conn, PostgreSQLConnection)
            else "MySQL"
            if isinstance(self.db_conn, MySQLConnection)
            else "DuckDB"
            if isinstance(self.db_conn, DuckDBConnection)
            else "DuckDB"
            if isinstance(self.db_conn, CSVConnection)
            else "SQLite"
            if isinstance(self.db_conn, SQLiteConnection)
            else "database"
        )

        if self.first_message:
            self.console.print(
                Panel.fit(
                    """
███████  ██████  ██      ███████  █████  ██████  ███████ ██████
██      ██    ██ ██      ██      ██   ██ ██   ██ ██      ██   ██
███████ ██    ██ ██      ███████ ███████ ██████  █████   ██████
     ██ ██ ▄▄ ██ ██           ██ ██   ██ ██   ██ ██      ██   ██
███████  ██████  ███████ ███████ ██   ██ ██████  ███████ ██   ██
            ▀▀
    """
                )
            )
            self.console.print(
                Markdown(
                    dedent("""
                    - Use `/` for slash commands
                    - Type `@` to get table name completions
                    - Start message with `#` to add something to agent's memory
                    - Use `Ctrl+C` to interrupt and `Ctrl+D` to exit
                    """)
                )
            )

        self.console.print(
            f"[bold blue]\n\nConnected to:[/bold blue] {db_name} ({db_type})\n"
        )
        # If resuming a thread, show a notice
        if self._thread_id:
            self.console.print(f"[dim]Resuming thread:[/dim] {self._thread_id}\n")

    async def _end_thread_and_display_resume_hint(self):
        """End thread and display command to resume thread"""
        # Print resume hint if there is an active thread
        if self._thread_id:
            await self._threads.end_thread(self._thread_id)
            self.console.print(
                f"[dim]You can continue this thread using:[/dim] saber threads resume {self._thread_id}"
            )

    async def _update_table_cache(self):
        """Update the table completer cache with fresh data."""
        try:
            tables_data = await SchemaManager(self.db_conn).list_tables()

            # Parse the table information
            table_list = []
            if isinstance(tables_data, dict) and "tables" in tables_data:
                for table in tables_data["tables"]:
                    if isinstance(table, dict):
                        name = table.get("name", "")
                        schema = table.get("schema", "")
                        full_name = table.get("full_name", "")

                        # Use full_name if available, otherwise construct it
                        if full_name:
                            table_name = full_name
                        elif schema and schema != "main":
                            table_name = f"{schema}.{name}"
                        else:
                            table_name = name

                        # No description needed - cleaner completions
                        table_list.append((table_name, ""))

            # Update the completer cache
            self.table_completer.update_cache(table_list)

        except Exception:
            # If there's an error, just use empty cache
            self.table_completer.update_cache([])

    async def _execute_query_with_cancellation(self, user_query: str):
        """Execute a query with cancellation support."""
        # Create cancellation token
        self.cancellation_token = asyncio.Event()

        # Create the query task
        query_task = asyncio.create_task(
            self.streaming_handler.execute_streaming_query(
                user_query,
                self.sqlsaber_agent,
                self.cancellation_token,
                self.message_history,
            )
        )
        self.current_task = query_task

        try:
            run_result = await query_task
            # Persist message history from this run using pydantic-ai API
            if run_result is not None:
                try:
                    # Use all_messages() so the system prompt and all prior turns are preserved
                    self.message_history = run_result.all_messages()

                    # Persist snapshot to thread storage (create or overwrite)
                    self._thread_id = await self._threads.save_snapshot(
                        messages_json=run_result.all_messages_json(),
                        database_name=self.database_name,
                        thread_id=self._thread_id,
                    )
                    # Save metadata separately (only if its the first message)
                    if self.first_message:
                        await self._threads.save_metadata(
                            thread_id=self._thread_id,
                            title=user_query,
                            model_name=self.sqlsaber_agent.agent.model.model_name,
                        )
                except Exception:
                    pass
                finally:
                    await self._threads.prune_threads()
        finally:
            self.current_task = None
            self.cancellation_token = None

    async def run(self):
        """Run the interactive session loop."""
        self.show_welcome_message()

        # Initialize table cache
        await self._update_table_cache()

        session = PromptSession(
            history=FileHistory(
                Path(platformdirs.user_config_dir("sqlsaber")) / "history"
            )
        )

        while True:
            try:
                with patch_stdout():
                    user_query = await session.prompt_async(
                        "",
                        multiline=True,
                        completer=CompositeCompleter(
                            SlashCommandCompleter(), self.table_completer
                        ),
                        show_frame=True,
                        bottom_toolbar=bottom_toolbar,
                        style=style,
                    )

                if not user_query:
                    continue

                if (
                    user_query in ["/exit", "/quit", "exit", "quit"]
                    or user_query.startswith("/exit")
                    or user_query.startswith("/quit")
                ):
                    await self._end_thread_and_display_resume_hint()
                    break

                if user_query == "/clear":
                    # Reset local history (pydantic-ai call will receive empty history on next run)
                    self.message_history = []
                    # End current thread (if any) so the next turn creates a fresh one
                    try:
                        if self._thread_id:
                            await self._threads.end_thread(self._thread_id)
                    except Exception:
                        pass
                    self.console.print("[green]Conversation history cleared.[/green]\n")
                    # Do not print resume hint when clearing; a new thread will be created on next turn
                    self._thread_id = None
                    continue

                # Thinking commands
                if user_query == "/thinking on":
                    self.sqlsaber_agent.set_thinking(enabled=True)
                    self.console.print("[green]✓ Thinking enabled[/green]\n")
                    continue

                if user_query == "/thinking off":
                    self.sqlsaber_agent.set_thinking(enabled=False)
                    self.console.print("[green]✓ Thinking disabled[/green]\n")
                    continue

                if memory_text := user_query.strip():
                    # Check if query starts with # for memory addition
                    if memory_text.startswith("#"):
                        memory_content = memory_text[1:].strip()  # Remove # and trim
                        if memory_content:
                            # Add memory via the agent's memory manager
                            try:
                                mm = self.sqlsaber_agent.memory_manager
                                if mm and self.database_name:
                                    memory = mm.add_memory(
                                        self.database_name, memory_content
                                    )
                                    self.console.print(
                                        f"[green]✓ Memory added:[/green] {memory_content}"
                                    )
                                    self.console.print(
                                        f"[dim]Memory ID: {memory.id}[/dim]\n"
                                    )
                                else:
                                    self.console.print(
                                        "[yellow]Could not add memory (no database context)[/yellow]\n"
                                    )
                            except Exception:
                                self.console.print(
                                    "[yellow]Could not add memory[/yellow]\n"
                                )
                        else:
                            self.console.print(
                                "[yellow]Empty memory content after '#'[/yellow]\n"
                            )
                        continue

                    # Execute query with cancellation support
                    await self._execute_query_with_cancellation(user_query)
                    self.display.show_newline()  # Empty line for readability

            except KeyboardInterrupt:
                # Handle Ctrl+C - cancel current task if running
                if self.current_task and not self.current_task.done():
                    if self.cancellation_token is not None:
                        self.cancellation_token.set()
                    self.current_task.cancel()
                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        pass
                    self.console.print("\n[yellow]Query interrupted[/yellow]")
                else:
                    self.console.print(
                        "\n[yellow]Press Ctrl+D to exit. Or use '/exit' or '/quit' slash command.[/yellow]"
                    )
            except EOFError:
                # Exit when Ctrl+D is pressed
                await self._end_thread_and_display_resume_hint()
                break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")

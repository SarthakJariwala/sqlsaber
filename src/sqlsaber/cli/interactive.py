"""Interactive mode handling for the CLI."""

import asyncio
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import platformdirs
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
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
from sqlsaber.theme.manager import get_theme_manager
from sqlsaber.threads import ThreadStorage
from sqlsaber.config.logging import get_logger

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent


class InteractiveSession:
    """Manages interactive CLI sessions."""

    exit_commands = {"/exit", "/quit", "exit", "quit"}
    resume_command_template = "saber threads resume {thread_id}"

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
        self.tm = get_theme_manager()
        # Conversation Thread persistence
        self._threads = ThreadStorage()
        self._thread_id: str | None = initial_thread_id
        self.first_message = not self._thread_id
        self.log = get_logger(__name__)

    def _history_path(self) -> Path:
        """Get the history file path, ensuring directory exists."""
        history_dir = Path(platformdirs.user_config_dir("sqlsaber"))
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir / "history"

    def _bottom_toolbar(self):
        """Get the bottom toolbar text."""
        return [
            (
                "class:bottom-toolbar",
                " Use 'Esc-Enter' or 'Meta-Enter' to submit.",
            )
        ]

    def _banner(self) -> str:
        """Get the ASCII banner."""
        return """[primary]
███████  ██████  ██      ███████  █████  ██████  ███████ ██████
██      ██    ██ ██      ██      ██   ██ ██   ██ ██      ██   ██
███████ ██    ██ ██      ███████ ███████ ██████  █████   ██████
     ██ ██ ▄▄ ██ ██           ██ ██   ██ ██   ██ ██      ██   ██
███████  ██████  ███████ ███████ ██   ██ ██████  ███████ ██   ██
            ▀▀
    [/primary]"""

    def _instructions(self) -> str:
        """Get the instruction text."""
        return dedent("""
                    - Use `/` for slash commands
                    - Type `@` to get table name completions
                    - Start message with `#` to add something to agent's memory
                    - Use `Ctrl+C` to interrupt and `Ctrl+D` to exit
                    """)

    def _db_type_name(self) -> str:
        """Get human-readable database type name."""
        mapping = {
            PostgreSQLConnection: "PostgreSQL",
            MySQLConnection: "MySQL",
            DuckDBConnection: "DuckDB",
            CSVConnection: "DuckDB",
            SQLiteConnection: "SQLite",
        }
        for cls, name in mapping.items():
            if isinstance(self.db_conn, cls):
                return name
        return "database"

    def _resume_hint(self, thread_id: str) -> str:
        """Build resume command hint."""
        return self.resume_command_template.format(thread_id=thread_id)

    def show_welcome_message(self):
        """Display welcome message for interactive mode."""
        if self.first_message:
            self.console.print(Panel.fit(self._banner(), border_style="primary"))
            self.console.print(
                Markdown(
                    self._instructions(),
                    code_theme=self.tm.pygments_style_name,
                    inline_code_theme=self.tm.pygments_style_name,
                )
            )

        db_name = self.database_name or "Unknown"
        self.console.print(
            f"[heading]\nConnected to {db_name} ({self._db_type_name()})[/heading]\n"
        )

        if self._thread_id:
            self.console.print(f"[muted]Resuming thread:[/muted] {self._thread_id}\n")

    async def _end_thread(self):
        """End thread and display resume hint."""
        if self._thread_id:
            await self._threads.end_thread(self._thread_id)
            self.console.print(
                f"[muted]You can continue this thread using:[/muted] {self._resume_hint(self._thread_id)}"
            )

    async def _handle_memory(self, content: str):
        """Handle memory addition command."""
        if not content:
            self.console.print("[warning]Empty memory content after '#'[/warning]\n")
            return

        try:
            mm = self.sqlsaber_agent.memory_manager
            if mm and self.database_name:
                memory = mm.add_memory(self.database_name, content)
                self.console.print(f"[success]✓ Memory added:[/success] {content}")
                self.console.print(f"[muted]Memory ID: {memory.id}[/muted]\n")
            else:
                self.console.print(
                    "[warning]Could not add memory (no database context)[/warning]\n"
                )
        except Exception as exc:
            self.console.print(f"[warning]Could not add memory:[/warning] {exc}\n")

    async def _cmd_clear(self):
        """Clear conversation history."""
        self.message_history = []
        try:
            if self._thread_id:
                await self._threads.end_thread(self._thread_id)
        except Exception:
            pass
        self.console.print("[success]Conversation history cleared.[/success]\n")
        self._thread_id = None
        self.first_message = True

    async def _cmd_thinking_on(self):
        """Enable thinking mode."""
        self.sqlsaber_agent.set_thinking(enabled=True)
        self.console.print("[success]✓ Thinking enabled[/success]\n")

    async def _cmd_thinking_off(self):
        """Disable thinking mode."""
        self.sqlsaber_agent.set_thinking(enabled=False)
        self.console.print("[success]✓ Thinking disabled[/success]\n")

    async def _handle_command(self, user_query: str) -> bool:
        """Handle slash commands. Returns True if command was handled."""
        if user_query == "/clear":
            await self._cmd_clear()
            return True
        if user_query == "/thinking on":
            await self._cmd_thinking_on()
            return True
        if user_query == "/thinking off":
            await self._cmd_thinking_off()
            return True
        return False

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

    async def before_prompt_loop(self):
        """Hook to refresh context before prompt loop."""
        await self._update_table_cache()

    async def _execute_query_with_cancellation(self, user_query: str):
        """Execute a query with cancellation support."""
        self.log.info("interactive.query.start", database=self.database_name)
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
                        self.first_message = False
                except Exception as e:
                    self.log.warning("interactive.thread.save_failed", error=str(e))
                finally:
                    await self._threads.prune_threads()
        finally:
            self.current_task = None
            self.cancellation_token = None
            self.log.info("interactive.query.end")

    async def run(self):
        """Run the interactive session loop."""
        self.log.info("interactive.start", database=self.database_name)
        self.show_welcome_message()
        await self.before_prompt_loop()

        session = PromptSession(history=FileHistory(self._history_path()))

        while True:
            try:
                with patch_stdout():
                    user_query = await session.prompt_async(
                        "> ",
                        multiline=True,
                        completer=CompositeCompleter(
                            SlashCommandCompleter(), self.table_completer
                        ),
                        bottom_toolbar=self._bottom_toolbar,
                        style=self.tm.pt_style(),
                    )

                if not user_query:
                    continue

                # Handle exit commands
                if user_query in self.exit_commands or any(
                    user_query.startswith(cmd) for cmd in self.exit_commands
                ):
                    await self._end_thread()
                    break

                # Handle slash commands
                if await self._handle_command(user_query):
                    continue

                # Handle memory addition
                if user_query.strip().startswith("#"):
                    await self._handle_memory(user_query.strip()[1:].strip())
                    continue

                # Execute query with cancellation support
                await self._execute_query_with_cancellation(user_query)
                self.display.show_newline()

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
                    self.console.print("\n[warning]Query interrupted[/warning]")
                else:
                    self.console.print(
                        "\n[warning]Press Ctrl+D to exit. Or use '/exit' or '/quit' slash command.[/warning]"
                    )
            except EOFError:
                # Exit when Ctrl+D is pressed
                await self._end_thread()
                break
            except Exception as exc:
                self.console.print(f"[error]Error:[/error] {exc}")
                self.log.exception("interactive.error", error=str(exc))

"""Interactive mode handling for the CLI."""

import asyncio
from collections.abc import Callable
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
from sqlsaber.cli.slash_commands import CommandContext, SlashCommandProcessor
from sqlsaber.cli.streaming import StreamingQueryHandler
from sqlsaber.cli.usage import SessionUsage
from sqlsaber.config.logging import get_logger
from sqlsaber.database import (
    CSVConnection,
    CSVsConnection,
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.database.schema import SchemaManager
from sqlsaber.theme.manager import get_theme_manager
from sqlsaber.threads.manager import ThreadManager

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent


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
        self.tm = get_theme_manager()

        # Component Managers
        self.thread_manager = ThreadManager(initial_thread_id)
        self.command_processor = SlashCommandProcessor()
        self.session_usage = SessionUsage()

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
            CSVsConnection: "DuckDB",
            SQLiteConnection: "SQLite",
        }
        for cls, name in mapping.items():
            if isinstance(self.db_conn, cls):
                return name
        return "database"

    def show_welcome_message(self):
        """Display welcome message for interactive mode."""
        if self.thread_manager.first_message:
            self.console.print(Panel.fit(self._banner(), border_style="primary"))
            self.console.print(
                Markdown(
                    self._instructions(),
                    code_theme=self.tm.pygments_style_name,
                    inline_code_theme=self.tm.pygments_style_name,
                )
            )

        db_name = self.database_name or "Unknown"
        model_name = getattr(self.sqlsaber_agent.agent.model, "model_name", "Unknown")
        self.console.print(
            f"[heading]\nConnected to {db_name} ({self._db_type_name()})[/heading]\n"
            f"[heading]Model: {model_name}[/heading]\n"
        )

        if self.thread_manager.current_thread_id:
            self.console.print(
                f"[muted]Resuming thread:[/muted] {self.thread_manager.current_thread_id}\n"
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

    async def _handle_handoff(
        self,
        session: PromptSession,
        goal: str,
        clear_history: Callable[[], None],
    ) -> None:
        """Handle the handoff flow: generate draft, let user edit, start new thread.

        Args:
            session: The prompt session for user input.
            goal: The user's goal for the new thread.
            clear_history: Callback to clear message history.
        """
        from sqlsaber.agents.handoff_agent import HandoffAgent

        self.display.live.start_status("Generating handoff prompt...")

        try:
            handoff_agent = HandoffAgent()

            draft = await handoff_agent.generate_draft(
                message_history=self.message_history or [],
                goal=goal,
            )
        except Exception as e:
            self.display.live.end_status()
            self.console.print(
                f"[error]Failed to generate handoff prompt:[/error] {e}\n"
            )
            return
        finally:
            self.display.live.end_status()

        self.console.print(
            "[success]Draft generated. Edit and submit to start new thread:[/success]\n"
        )

        try:
            with patch_stdout():
                edited = await session.prompt_async(
                    "handoff > ",
                    multiline=True,
                    default=draft,
                    bottom_toolbar=self._bottom_toolbar,
                    style=self.tm.pt_style(),
                )
        except KeyboardInterrupt:
            self.console.print("\n[muted]Handoff cancelled.[/muted]\n")
            return

        edited = edited.strip()
        if not edited:
            self.console.print("[warning]Empty handoff prompt; cancelled.[/warning]\n")
            return

        old_id = await self.thread_manager.end_current_thread()
        if old_id:
            self.console.print(f"[muted]Previous thread saved:[/muted] {old_id}")
            self.console.print(
                f"[muted]Resume with:[/muted] saber threads resume {old_id}\n"
            )

        clear_history()
        await self.thread_manager.clear_current_thread()

        self.console.print("[heading]Starting new thread...[/heading]\n")

        await self._execute_query_with_cancellation(edited)
        self.display.show_newline()

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
                self.message_history = await self.thread_manager.save_run(
                    run_result=run_result,
                    database_name=self.database_name,
                    user_query=user_query,
                    model_name=getattr(
                        self.sqlsaber_agent.agent.model, "model_name", "Unknown"
                    ),
                )
                # Track usage for session summary
                # Use result.response.usage for the FINAL request's context size
                final_context = run_result.response.usage.input_tokens
                self.session_usage.add_run(run_result.usage(), final_context)
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

        def clear_history():
            self.message_history = []

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

                user_query = user_query.strip()

                if not user_query:
                    continue

                # Process slash commands
                context = CommandContext(
                    console=self.console,
                    agent=self.sqlsaber_agent,
                    thread_manager=self.thread_manager,
                    on_clear_history=clear_history,
                    session_usage=self.session_usage,
                )

                cmd_result = await self.command_processor.process(user_query, context)
                if cmd_result.should_exit:
                    break

                # Handle handoff command
                if cmd_result.handoff_goal:
                    await self._handle_handoff(
                        session, cmd_result.handoff_goal, clear_history
                    )
                    continue

                if cmd_result.handled:
                    continue

                # Handle memory addition
                if user_query.strip().startswith("#"):
                    await self._handle_memory(user_query[1:].strip())
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
                self.display.show_session_summary(self.session_usage)
                ended_thread_id = await self.thread_manager.end_current_thread()
                if ended_thread_id:
                    hint = f"saber threads resume {ended_thread_id}"
                    self.console.print(
                        f"[muted]You can continue this thread using:[/muted] {hint}"
                    )
                break
            except Exception as exc:
                self.console.print(f"[error]Error:[/error] {exc}")
                self.log.exception("interactive.error", error=str(exc))

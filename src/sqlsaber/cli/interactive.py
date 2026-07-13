"""Interactive mode handling for the CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import platformdirs
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from sqlsaber.cli.completers import SQLSaberAutocompleteProvider
from sqlsaber.cli.slash_commands import CommandContext, SlashCommandProcessor
from sqlsaber.cli.tui_chat import (
    DANGEROUS_MODE_FOOTER_LABEL,
    ChatApp,
    ChatConsole,
    build_chat_app,
)
from sqlsaber.cli.tui_streaming import TUIStreamingQueryHandler
from sqlsaber.cli.usage import (
    SessionUsage,
    format_cost_usd,
    format_tokens,
    request_usages_from_run_result,
)
from sqlsaber.config.logging import get_logger
from sqlsaber.database import (
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.database.csv import CSVConnection
from sqlsaber.database.csvs import CSVsConnection
from sqlsaber.database.schema import SchemaManager
from sqlsaber.theme.manager import get_theme_manager

if TYPE_CHECKING:
    from sqlsaber.session import SQLSaberSession

QUERY_CANCEL_GRACE_SECONDS = 0.1


class InteractiveSession:
    """Manages interactive CLI sessions."""

    def __init__(
        self,
        console: Console,
        session: "SQLSaberSession",
        *,
        initial_history: list | None = None,
    ):
        self.console = console
        self.session = session
        self.sqlsaber_agent = session.agent
        self.allow_dangerous = bool(
            getattr(
                session.options,
                "allow_dangerous",
                getattr(self.sqlsaber_agent, "allow_dangerous", False),
            )
        )
        self.db_conn = session.connection
        self.database_name = session.db_name
        self.database_names = list(getattr(session, "db_names", [session.db_name]))
        self.streaming_handler: TUIStreamingQueryHandler | None = None
        self.current_task: asyncio.Task | None = None
        self.cancellation_token: asyncio.Event | None = None
        self._submit_pending = False
        self.autocomplete_provider = SQLSaberAutocompleteProvider()
        self.message_history: list | None = initial_history or []
        self.tm = get_theme_manager()
        self._handoff_mode = False
        self._exit_finalized = False

        if session.thread_manager is None:
            raise ValueError(
                "InteractiveSession requires SQLSaberSession with thread_manager set."
            )
        self.thread_manager = session.thread_manager
        self.command_processor = SlashCommandProcessor()
        self.session_usage = SessionUsage()

        self.log = get_logger(__name__)

    def _history_path(self) -> Path:
        """Get the history file path, ensuring directory exists."""
        history_dir = Path(platformdirs.user_config_dir("sqlsaber"))
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir / "history"

    def _load_history(self) -> list[str]:
        path = self._history_path()
        if not path.exists():
            return []
        try:
            return [
                line[1:] if line.startswith("+") else line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ][-100:]
        except OSError:
            return []

    def _append_history(self, text: str) -> None:
        try:
            with self._history_path().open("a", encoding="utf-8") as history_file:
                history_file.write(f"+{text.replace('\n', ' ')}\n")
        except OSError:
            return

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

    def _model_name(self) -> str:
        return getattr(self.sqlsaber_agent.agent.model, "model_name", "Unknown")

    def _model_id(self) -> str | None:
        model = self.sqlsaber_agent.agent.model
        model_id = getattr(model, "model_id", None)
        if model_id:
            return str(model_id)
        configured_model = getattr(
            getattr(self.sqlsaber_agent, "config", None), "model", None
        )
        configured_name = getattr(configured_model, "name", None)
        if configured_name:
            return str(configured_name)
        model_name = self._model_name()
        return None if model_name == "Unknown" else model_name

    def _database_footer_text(self) -> str:
        database_names = getattr(self, "database_names", None)
        if not database_names:
            database_names = [self.database_name or "Unknown"]

        if len(database_names) > 1:
            return f"DBs: {', '.join(database_names)}"

        db_name = database_names[0] or "Unknown"
        return f"DB: {db_name} ({self._db_type_name()})"

    def _footer_text(self) -> str:
        parts = [self._database_footer_text(), f"Model: {self._model_name()}"]
        if dangerous_mode := self._dangerous_mode_footer_text():
            parts.append(dangerous_mode)
        parts.append(self._usage_footer_text())
        return " | ".join(parts)

    def _dangerous_mode_footer_text(self) -> str | None:
        agent = getattr(self, "sqlsaber_agent", None)
        allow_dangerous = bool(
            getattr(self, "allow_dangerous", False)
            or getattr(agent, "allow_dangerous", False)
        )
        return DANGEROUS_MODE_FOOTER_LABEL if allow_dangerous else None

    def _usage_footer_text(self) -> str:
        session_usage = getattr(self, "session_usage", SessionUsage())
        return (
            f"Usage: ↑{format_tokens(session_usage.total_input_tokens)} "
            f"↓{format_tokens(session_usage.total_output_tokens)} | "
            f"Ctx: {format_tokens(session_usage.current_context_tokens)} | "
            f"Cost: {format_cost_usd(session_usage.total_cost_usd)}"
        )

    def _refresh_footer(self) -> None:
        if self.streaming_handler is not None:
            self.streaming_handler.app.set_footer(self._footer_text())

    def show_welcome_message(self, app: ChatApp) -> None:
        """Display welcome message for interactive mode."""

        def render(console: Console) -> None:
            if self.thread_manager.first_message:
                console.print(Panel.fit(self._banner(), border_style="primary"))
                console.print(
                    Markdown(
                        self._instructions(),
                        code_theme=self.tm.pygments_style_name,
                        inline_code_theme=self.tm.pygments_style_name,
                    )
                )

            if self.thread_manager.current_thread_id:
                console.print(
                    f"[muted]Resuming thread:[/muted] {self.thread_manager.current_thread_id}\n"
                )

        app.append_rich(render)

    async def _update_table_cache(self) -> None:
        """Update the table completer cache with fresh data."""
        try:
            tables_data = await SchemaManager(self.db_conn).list_tables()

            table_list = []
            if isinstance(tables_data, dict) and "tables" in tables_data:
                for table in tables_data["tables"]:
                    if isinstance(table, dict):
                        name = table.get("name", "")
                        schema = table.get("schema", "")
                        full_name = table.get("full_name", "")

                        if full_name:
                            table_name = full_name
                        elif schema and schema != "main":
                            table_name = f"{schema}.{name}"
                        else:
                            table_name = name

                        table_list.append((table_name, ""))

            self.autocomplete_provider.update_table_cache(table_list)

        except Exception:
            self.autocomplete_provider.update_table_cache([])

    async def before_prompt_loop(self) -> None:
        """Hook to refresh context before prompt loop."""
        await self._update_table_cache()

    async def _start_handoff(self, app: ChatApp, goal: str) -> None:
        """Generate a handoff draft and put it in the focused editor."""
        from sqlsaber.agents.handoff_agent import HandoffAgent

        app.set_loading("Generating handoff prompt...")
        try:
            handoff_agent = HandoffAgent()
            draft = await handoff_agent.generate_draft(
                message_history=self.message_history or [],
                goal=goal,
            )
        except Exception as exc:
            error_message = str(exc)
            app.append_rich(
                lambda console: console.print(
                    f"[error]Failed to generate handoff prompt:[/error] {error_message}\n"
                )
            )
            return
        finally:
            app.clear_status()

        self._handoff_mode = True
        app.editor.set_text(draft)
        app.set_status("Edit the handoff draft and press Enter to start a new thread.")

    async def _submit_handoff(
        self, app: ChatApp, edited: str, clear_history: Callable[[], None]
    ) -> None:
        self._handoff_mode = False
        app.clear_status()
        edited = edited.strip()
        if not edited:
            app.append_rich(
                lambda console: console.print(
                    "[warning]Empty handoff prompt; cancelled.[/warning]\n"
                )
            )
            return

        old_id = await self.thread_manager.end_current_thread()
        if old_id:
            app.append_rich(
                lambda console: console.print(
                    f"[muted]Previous thread saved:[/muted] {old_id}\n"
                    f"[muted]Resume with:[/muted] saber threads resume {old_id}\n"
                )
            )

        clear_history()
        await self.thread_manager.clear_current_thread()
        app.append_rich(
            lambda console: console.print("[heading]Starting new thread...[/heading]\n")
        )
        await self._execute_query_with_cancellation(edited)

    async def _execute_query_with_cancellation(self, user_query: str) -> None:
        """Execute a query with cancellation support."""
        if self.streaming_handler is None:
            raise RuntimeError("Streaming handler has not been initialized.")

        self.log.info("interactive.query.start", database=self.database_name)
        self.cancellation_token = asyncio.Event()
        query_task = asyncio.create_task(
            self.streaming_handler.execute_streaming_query(
                user_query,
                run_query=self.session.query,
                cancellation_token=self.cancellation_token,
                message_history=self.message_history,
            )
        )
        self.current_task = query_task

        try:
            run_result = await query_task
            if run_result is not None:
                self.message_history = run_result.all_messages()
                final_context = run_result.response.usage.input_tokens
                self.session_usage.add_run(
                    run_result.usage(),
                    final_context,
                    model_name=self._model_id(),
                    request_usages=request_usages_from_run_result(run_result),
                )
                self._refresh_footer()
        finally:
            self.current_task = None
            self.cancellation_token = None
            self.log.info("interactive.query.end")

    async def _cancel_current_task(self, app: ChatApp, chat_console: Console) -> None:
        if self.current_task and not self.current_task.done():
            task = self.current_task
            if self.cancellation_token is not None:
                self.cancellation_token.set()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=QUERY_CANCEL_GRACE_SECONDS
                    )
                    return
                except TimeoutError:
                    task.cancel()
                except asyncio.CancelledError:
                    return
            else:
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return

        if self._handoff_mode:
            self._cancel_handoff_editing(app, chat_console)
            return

        chat_console.print(
            "[warning]Press Ctrl+D to exit. Or use '/exit' or '/quit'.[/warning]"
        )
        app.tui.set_focus(app.editor)

    def _cancel_handoff_editing(self, app: ChatApp, chat_console: Console) -> None:
        self._handoff_mode = False
        app.editor.set_text("")
        app.clear_status()
        chat_console.print("[warning]Handoff cancelled.[/warning]\n")
        app.tui.set_focus(app.editor)

    async def _handle_submit(
        self,
        app: ChatApp,
        chat_console: Console,
        clear_history: Callable[[], None],
        user_query: str,
    ) -> None:
        try:
            if self.current_task and not self.current_task.done():
                chat_console.print(
                    "[warning]A query is already running. Press Ctrl+C to interrupt it.[/warning]"
                )
                return

            if self._handoff_mode:
                if user_query.strip():
                    self._append_history(user_query)
                await self._submit_handoff(app, user_query, clear_history)
                return

            self._append_history(user_query)

            context = CommandContext(
                console=chat_console,
                agent=self.sqlsaber_agent,
                thread_manager=self.thread_manager,
                on_clear_history=clear_history,
                session_usage=self.session_usage,
            )

            cmd_result = await self.command_processor.process(user_query, context)
            if cmd_result.should_exit:
                self._exit_finalized = True
                app.stop()
                return

            if cmd_result.handoff_goal:
                await self._start_handoff(app, cmd_result.handoff_goal)
                return

            if cmd_result.handled:
                return

            await self._execute_query_with_cancellation(user_query)
        except Exception as exc:
            chat_console.print(f"[error]Error:[/error] {exc}")
            self.log.exception("interactive.error", error=str(exc))
        finally:
            app.tui.set_focus(app.editor)

    async def _finalize_exit(self) -> None:
        if self._exit_finalized:
            return
        ended_thread_id = await self.thread_manager.end_current_thread()
        if ended_thread_id:
            hint = f"saber threads resume {ended_thread_id}"
            self.console.print(
                f"[muted]You can continue this thread using:[/muted] {hint}"
            )
        self._exit_finalized = True

    async def run(self) -> None:
        """Run the interactive session loop."""
        self.log.info("interactive.start", database=self.database_name)
        await self.before_prompt_loop()

        exit_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        app_ref: dict[str, ChatApp] = {}
        chat_console_ref: dict[str, ChatConsole] = {}

        def clear_history() -> None:
            self.message_history = []

        def on_submit(user_query: str) -> bool:
            app = app_ref["app"]
            chat_console = chat_console_ref["chat_console"]

            if self._submit_pending or (
                self.current_task and not self.current_task.done()
            ):
                chat_console.print(
                    "[warning]A query is already running. Press Ctrl+C to interrupt it.[/warning]"
                )
                app.tui.set_focus(app.editor)
                return False

            self._submit_pending = True

            async def submit_query() -> None:
                try:
                    await self._handle_submit(
                        app,
                        chat_console,
                        clear_history,
                        user_query,
                    )
                finally:
                    self._submit_pending = False

            loop.call_soon_threadsafe(lambda: asyncio.create_task(submit_query()))
            return True

        def open_command_palette(app: ChatApp) -> None:
            app.show_command_palette(
                thinking_enabled=self.sqlsaber_agent.thinking_enabled,
                thinking_level=self.sqlsaber_agent.thinking_level,
                on_thinking_change=self.sqlsaber_agent.set_thinking,
                model_name=self._model_name(),
                database_name=self.database_name,
            )

        def on_cancel() -> None:
            app = app_ref["app"]
            chat_console = chat_console_ref["chat_console"]
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._cancel_current_task(
                        app,
                        chat_console,
                    )
                )
            )

        def on_exit() -> None:
            loop.call_soon_threadsafe(exit_event.set)

        app = build_chat_app(
            on_submit=on_submit,
            on_exit=on_exit,
            on_cancel=on_cancel,
            should_submit_empty=lambda: self._handoff_mode,
            autocomplete_provider=self.autocomplete_provider,
            console=self.console,
            footer_text=self._footer_text(),
            on_open_command_palette=open_command_palette,
        )
        chat_console = ChatConsole(app)
        app_ref["app"] = app
        chat_console_ref["chat_console"] = chat_console
        app.editor.history = self._load_history()
        self.streaming_handler = TUIStreamingQueryHandler(
            app,
            self.console,
            display_registry_provider=lambda: getattr(
                getattr(self, "sqlsaber_agent", None), "display_registry", None
            ),
        )
        self.show_welcome_message(app)

        app.tui.start()
        try:
            await exit_event.wait()
        finally:
            if not app.tui.stopped:
                app.stop()
            await self._finalize_exit()

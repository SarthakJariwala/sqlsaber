"""Interactive mode handling for the CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import platformdirs

from sqlsaber.cli.completers import SQLSaberAutocompleteProvider
from sqlsaber.cli.slash_commands import (
    CommandContext,
    SlashCommandProcessor,
    ThreadResumeRequest,
)
from sqlsaber.cli.tui_chat import (
    DANGEROUS_MODE_FOOTER_LABEL,
    ChatApp,
    build_chat_app,
)
from sqlsaber.cli.tui_streaming import TUIStreamingQueryHandler
from sqlsaber.cli.update_check import bind_update_notice
from sqlsaber.cli.usage import SessionUsage, format_cost_usd, format_tokens
from sqlsaber.config.logging import get_logger
from sqlsaber.render import blocks as b

if TYPE_CHECKING:
    from sqlsaber import SQLSaber, SQLSaberResult

QUERY_CANCEL_GRACE_SECONDS = 0.1


class InteractiveSession:
    """Manages interactive CLI sessions."""

    def __init__(self, saber: "SQLSaber") -> None:
        self.saber = saber
        self.streaming_handler: TUIStreamingQueryHandler | None = None
        self.current_task: asyncio.Task[SQLSaberResult | None] | None = None
        self.cancellation_token: asyncio.Event | None = None
        self._submit_pending = False
        self.autocomplete_provider = SQLSaberAutocompleteProvider()
        self._handoff_mode = False
        self._exit_finalized = False
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
        return """
███████  ██████  ██      ███████  █████  ██████  ███████ ██████
██      ██    ██ ██      ██      ██   ██ ██   ██ ██      ██   ██
███████ ██    ██ ██      ███████ ███████ ██████  █████   ██████
     ██ ██ ▄▄ ██ ██           ██ ██   ██ ██   ██ ██      ██   ██
███████  ██████  ███████ ███████ ██   ██ ██████  ███████ ██   ██
            ▀▀
""".strip()

    def _instructions(self) -> str:
        """Get the instruction text."""
        return dedent("""
                    - Use `/` for slash commands
                    - Type `@` to get table name completions
                    - Use `Ctrl+C` to interrupt and `Ctrl+D` to exit
                    """)

    def _model_name(self) -> str:
        return self.saber.info.model_name

    def _model_id(self) -> str | None:
        info = self.saber.info
        return info.model_id or info.model_name

    def _database_footer_text(self) -> str:
        info = self.saber.info
        if len(info.database_names) > 1:
            return f"DBs: {', '.join(info.database_names)}"

        db_name = info.primary_database_name or "Unknown"
        return f"DB: {db_name} ({info.primary_database_type})"

    def _footer_text(self) -> str:
        parts = [self._database_footer_text(), f"Model: {self._model_name()}"]
        if dangerous_mode := self._dangerous_mode_footer_text():
            parts.append(dangerous_mode)
        parts.append(self._usage_footer_text())
        return " | ".join(parts)

    def _dangerous_mode_footer_text(self) -> str | None:
        return DANGEROUS_MODE_FOOTER_LABEL if self.saber.info.dangerous_mode else None

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
        from sqlsaber.cli.chat_surface import ChatSurface

        surface = ChatSurface(app)
        info = self.saber.info
        if info.is_new_thread:
            surface.emit(
                b.panel((b.md(f"```\n{self._banner()}\n```"),), role="primary"),
                b.md(self._instructions()),
            )

        if info.thread_id:
            surface.emit(b.md(f"Resuming thread: `{info.thread_id}`", role="muted"))

    async def _update_table_cache(self) -> None:
        """Update the table completer cache with fresh data."""
        try:
            tables = await self.saber.list_tables()
            self.autocomplete_provider.update_table_cache(
                [(table.completion_name, "") for table in tables]
            )
        except Exception:
            self.autocomplete_provider.update_table_cache([])

    async def before_prompt_loop(self) -> None:
        """Hook to refresh context before prompt loop."""
        await self._update_table_cache()

    async def _resume_thread(
        self,
        app: ChatApp,
        surface,
        request: ThreadResumeRequest,
    ) -> None:
        from sqlsaber.cli.threads import prepare_thread_resume, render_prepared_thread

        if self.saber.info.thread_id == request.thread_id:
            surface.emit(b.warn(f"Thread '{request.thread_id}' is already active."))
            return

        prepared = await prepare_thread_resume(
            request.thread_id,
            list(request.databases) or None,
        )
        prepared_saber = prepared.saber
        previous = self.saber
        try:
            await previous.end_thread()
        except BaseException:
            await prepared_saber.close()
            raise

        self.saber = prepared_saber
        try:
            await previous.close()
        except Exception as exc:
            self.log.warning("interactive.resume.cleanup_failed", error=str(exc))
            surface.emit(b.warn(f"Previous session cleanup failed: {exc}"))

        self.streaming_handler = TUIStreamingQueryHandler(
            app,
            display_registry_provider=lambda: self.saber.display_registry,
            query_result_store=self.saber.query_result_store,
        )
        self.session_usage = SessionUsage()
        app.clear_chat()
        render_prepared_thread(surface, prepared)
        await self._update_table_cache()
        self._refresh_footer()

    async def _start_handoff(self, app: ChatApp, goal: str) -> None:
        """Generate a handoff draft and put it in the focused editor."""
        app.set_loading("Generating handoff prompt...")
        try:
            draft = await self.saber.draft_handoff(goal)
        except Exception as exc:
            error_message = str(exc)
            from sqlsaber.cli.chat_surface import ChatSurface

            ChatSurface(app).emit(
                b.error(f"Failed to generate handoff prompt: {error_message}")
            )
            return
        finally:
            app.clear_status()

        self._handoff_mode = True
        app.editor.set_text(draft)
        app.set_status("Edit the handoff draft and press Enter to start a new thread.")

    async def _submit_handoff(self, app: ChatApp, edited: str) -> None:
        self._handoff_mode = False
        app.clear_status()
        edited = edited.strip()
        if not edited:
            from sqlsaber.cli.chat_surface import ChatSurface

            ChatSurface(app).emit(b.warn("Empty handoff prompt; cancelled."))
            return

        old_id = await self.saber.new_thread()
        if old_id:
            from sqlsaber.cli.chat_surface import ChatSurface

            ChatSurface(app).emit(
                b.md(
                    f"Previous thread saved: `{old_id}`\n"
                    f"Resume with: `saber threads resume {old_id}`",
                    role="muted",
                )
            )
        from sqlsaber.cli.chat_surface import ChatSurface

        ChatSurface(app).emit(b.md("**Starting new thread...**", role="primary"))
        await self._execute_query_with_cancellation(edited)

    async def _execute_query_with_cancellation(self, user_query: str) -> None:
        """Execute a query with cancellation support."""
        if self.streaming_handler is None:
            raise RuntimeError("Streaming handler has not been initialized.")

        self.log.info(
            "interactive.query.start",
            database=self.saber.info.primary_database_name,
        )
        self.cancellation_token = asyncio.Event()
        query_task = asyncio.create_task(
            self.streaming_handler.execute_streaming_query(
                user_query,
                run_query=self.saber.query,
                cancellation_token=self.cancellation_token,
            )
        )
        self.current_task = query_task

        try:
            result = await query_task
            if result is not None and result.usage is not None:
                self.session_usage.add_run(
                    result.usage,
                    result.final_context_tokens,
                    model_name=self._model_id(),
                    request_usages=result.request_usages,
                )
                self._refresh_footer()
        finally:
            self.current_task = None
            self.cancellation_token = None
            self.log.info("interactive.query.end")

    async def _cancel_current_task(self, app: ChatApp) -> None:
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
            self._cancel_handoff_editing(app)
            return

        from sqlsaber.cli.chat_surface import ChatSurface

        ChatSurface(app).emit(
            b.warn("Press Ctrl+D to exit. Or use '/exit' or '/quit'.")
        )
        app.tui.set_focus(app.editor)

    def _cancel_handoff_editing(self, app: ChatApp) -> None:
        self._handoff_mode = False
        app.editor.set_text("")
        app.clear_status()
        from sqlsaber.cli.chat_surface import ChatSurface

        ChatSurface(app).emit(b.warn("Handoff cancelled."))
        app.tui.set_focus(app.editor)

    async def _handle_submit(
        self,
        app: ChatApp,
        surface,
        user_query: str,
    ) -> None:
        try:
            if self.current_task and not self.current_task.done():
                surface.emit(
                    b.warn("A query is already running. Press Ctrl+C to interrupt it.")
                )
                return

            if self._handoff_mode:
                if user_query.strip():
                    self._append_history(user_query)
                await self._submit_handoff(app, user_query)
                return

            if not user_query.lstrip().startswith("/"):
                self._append_history(user_query)

            context = CommandContext(
                surface=surface,
                saber=self.saber,
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

            if cmd_result.resume_request is not None:
                await self._resume_thread(app, surface, cmd_result.resume_request)
                return

            if cmd_result.handled:
                return

            await self._execute_query_with_cancellation(user_query)
        except Exception as exc:
            surface.emit(b.error(str(exc)))
            self.log.exception("interactive.error", error=str(exc))
        finally:
            app.tui.set_focus(app.editor)

    async def _finalize_exit(self) -> None:
        if self._exit_finalized:
            return
        ended_thread_id = await self.saber.end_thread()
        if ended_thread_id:
            hint = f"saber threads resume {ended_thread_id}"
            from sqlsaber.cli.output import out

            out(b.md(f"You can continue this thread using: `{hint}`", role="muted"))
        self._exit_finalized = True

    async def run(self) -> None:
        """Run the interactive session loop."""
        self.log.info(
            "interactive.start", database=self.saber.info.primary_database_name
        )
        await self.before_prompt_loop()

        from sqlsaber.cli.chat_surface import ChatSurface

        exit_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        app_ref: dict[str, ChatApp] = {}
        surface_ref: dict[str, ChatSurface] = {}

        def on_submit(user_query: str) -> bool:
            app = app_ref["app"]
            surface = surface_ref["surface"]

            if self._submit_pending or (
                self.current_task and not self.current_task.done()
            ):
                surface.emit(
                    b.warn("A query is already running. Press Ctrl+C to interrupt it.")
                )
                app.tui.set_focus(app.editor)
                return False

            self._submit_pending = True

            async def submit_query() -> None:
                try:
                    await self._handle_submit(app, surface, user_query)
                finally:
                    self._submit_pending = False

            loop.call_soon_threadsafe(lambda: asyncio.create_task(submit_query()))
            return True

        def open_command_palette(app: ChatApp) -> None:
            info = self.saber.info
            app.show_command_palette(
                thinking_enabled=info.thinking.enabled,
                thinking_level=info.thinking.level,
                commands=self.command_processor.palette_commands(),
                model_name=info.model_name,
                database_name=info.primary_database_name,
            )

        def on_cancel() -> None:
            app = app_ref["app"]
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._cancel_current_task(app))
            )

        def on_exit() -> None:
            loop.call_soon_threadsafe(exit_event.set)

        app = build_chat_app(
            on_submit=on_submit,
            on_exit=on_exit,
            on_cancel=on_cancel,
            should_submit_empty=lambda: self._handoff_mode,
            autocomplete_provider=self.autocomplete_provider,
            footer_text=self._footer_text(),
            on_open_command_palette=open_command_palette,
        )
        surface = ChatSurface(app)
        app_ref["app"] = app
        surface_ref["surface"] = surface
        app.editor.history = self._load_history()
        self.streaming_handler = TUIStreamingQueryHandler(
            app,
            display_registry_provider=lambda: self.saber.display_registry,
            query_result_store=self.saber.query_result_store,
        )
        self.show_welcome_message(app)
        bind_update_notice(surface.emit)

        app.tui.start()
        try:
            await exit_event.wait()
        finally:
            bind_update_notice(None)
            if not app.tui.stopped:
                app.stop()
            await self._finalize_exit()

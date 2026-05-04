"""Display utilities for the CLI interface.

All rendering occurs on the event loop thread.
Streaming segments use Live Markdown; transient status and SQL blocks are also
rendered with Live.
"""

import json
from typing import TYPE_CHECKING, Type

from pydantic_ai.messages import ModelResponsePart, TextPart, ThinkingPart
from rich.columns import Columns
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from sqlsaber.theme.manager import get_theme_manager
from sqlsaber.tools.display import ResultConfig, SpecRenderer, ToolDisplaySpec
from sqlsaber.tools.registry import tool_registry

if TYPE_CHECKING:
    from sqlsaber.cli.usage import SessionUsage


class _SimpleCodeBlock(CodeBlock):
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            background_color="default",
            word_wrap=True,
        )


class LiveMarkdownRenderer:
    """Handles Live markdown rendering with segment separation.

    Supports different segment kinds: 'assistant', 'thinking', 'sql'.
    Adds visible paragraph breaks between segments and renders code fences
    with nicer formatting.
    """

    _patched_fences = False

    def __init__(self, console: Console):
        self.console = console
        self.tm = get_theme_manager()
        self._live: Live | None = None
        self._status_live: Live | None = None
        self._buffer: str = ""
        self._current_kind: Type[ModelResponsePart] | None = None

    def prepare_code_blocks(self) -> None:
        """Patch rich Markdown fence rendering once for nicer code blocks."""
        if LiveMarkdownRenderer._patched_fences:
            return
        # Guard with class check to avoid re-patching if already applied
        if Markdown.elements.get("fence") is not _SimpleCodeBlock:
            Markdown.elements["fence"] = _SimpleCodeBlock
        LiveMarkdownRenderer._patched_fences = True

    def ensure_segment(self, kind: Type[ModelResponsePart]) -> None:
        """
        Ensure a markdown Live segment is active for the given kind.

        When switching kinds, end the previous segment and add a paragraph break.
        """
        # If a transient status is showing, clear it first (no paragraph break)
        if self._status_live is not None:
            self.end_status()
        if self._live is not None and self._current_kind == kind:
            return
        if self._live is not None:
            self.end()
            self.paragraph_break()

        self._start(kind)
        self._current_kind = kind

    def append(self, text: str | None) -> None:
        """Append text to the current markdown segment and refresh."""
        if not text:
            return
        if self._live is None:
            # default to assistant if no segment was ensured
            self.ensure_segment(TextPart)

        self._buffer += text

        # Apply dim styling for thinking segments
        if self._live is not None:
            if self._current_kind == ThinkingPart:
                content = Markdown(
                    self._buffer, style="muted", code_theme=self.tm.pygments_style_name
                )
                self._live.update(content)
            else:
                self._live.update(
                    Markdown(self._buffer, code_theme=self.tm.pygments_style_name)
                )

    def end(self) -> None:
        """Finalize and stop the current Live segment, if any."""
        if self._live is None:
            return
        # Persist the *final* render exactly once, then shut Live down.
        buf = self._buffer
        kind = self._current_kind
        self._live.stop()
        self._live = None
        self._buffer = ""
        self._current_kind = None
        # Print the complete markdown to scroll-back for permanent reference
        if buf:
            if kind == ThinkingPart:
                self.console.print(
                    Markdown(buf, style="muted", code_theme=self.tm.pygments_style_name)
                )
            else:
                self.console.print(
                    Markdown(buf, code_theme=self.tm.pygments_style_name)
                )

    def end_if_active(self) -> None:
        self.end()

    def paragraph_break(self) -> None:
        self.console.print()

    def start_sql_block(self, sql: str) -> None:
        """Render a SQL block using a transient Live markdown segment."""
        if not sql or not isinstance(sql, str) or not sql.strip():
            return
        # Separate from surrounding content
        self.end_if_active()
        self.paragraph_break()
        self._buffer = f"```sql\n{sql}\n```"
        # Use context manager to auto-stop and persist final render
        with Live(
            Markdown(self._buffer, code_theme=self.tm.pygments_style_name),
            console=self.console,
            vertical_overflow="visible",
            refresh_per_second=12,
        ):
            pass

    def start_status(self, message: str = "Crunching data...") -> None:
        """Show a transient status line with a spinner until streaming starts."""
        if self._status_live is not None:
            # Update existing status text
            self._status_live.update(self._status_renderable(message))
            return
        live = Live(
            self._status_renderable(message),
            console=self.console,
            transient=True,  # disappear when stopped
            refresh_per_second=12,
        )
        self._status_live = live
        live.start()

    def end_status(self) -> None:
        live = self._status_live
        if live is None:
            return
        live.stop()
        self._status_live = None

    def _status_renderable(self, message: str):
        spinner = Spinner("dots", style=self.tm.style("spinner"))
        text = Text(f" {message}", style=self.tm.style("status"))
        return Columns([spinner, text], expand=False)

    def _start(
        self, kind: Type[ModelResponsePart] | None = None, initial_markdown: str = ""
    ) -> None:
        if self._live is not None:
            self.end()
        self._buffer = initial_markdown or ""

        # Add visual styling for thinking segments
        if kind == ThinkingPart:
            if self.console.is_terminal:
                self.console.print("[muted]💭 Thinking...[/muted]")
            else:
                self.console.print("*Thinking...*\n")

        # NOTE: Use transient=True so the live widget disappears on exit,
        # giving a clean transition to the final printed result.
        live = Live(
            Markdown(self._buffer, code_theme=self.tm.pygments_style_name),
            console=self.console,
            transient=True,
            refresh_per_second=12,
        )
        self._live = live
        live.start()


class DisplayManager:
    """Manages display formatting and output for the CLI."""

    def __init__(self, console: Console):
        self.console = console
        self.live = LiveMarkdownRenderer(console)
        self.tm = get_theme_manager()
        self._spec_renderer = SpecRenderer(self.tm)
        self._replay_messages: list | None = None
        self._pending_ask_database_context: dict[str, str] | None = None
        self._pending_ask_database_contexts: dict[str, dict[str, str]] = {}

    def set_replay_messages(self, messages: list) -> None:
        """Set message history for replay scenarios (e.g., threads show)."""
        self._replay_messages = messages

    def show_tool_executing(
        self,
        tool_name: str,
        tool_input: dict,
        *,
        tool_call_id: str | None = None,
    ):
        """Display tool execution details."""
        self.show_newline()
        if self._render_coordinator_tool_executing(
            tool_name, tool_input, tool_call_id=tool_call_id
        ):
            return

        tool = self._get_tool(tool_name)
        if tool and tool.render_executing(self.console, tool_input):
            return

        spec = tool.display_spec if tool else None
        if spec:
            self._spec_renderer.render_executing(
                self.console, tool_name, tool_input, spec
            )
            return

        self._render_fallback_result(tool_input)

    def show_text_stream(self, text: str):
        """Display streaming text."""
        if text is not None:  # Extra safety check
            self.console.print(text, end="", markup=False)

    def show_tool_result(
        self,
        tool_name: str,
        result: object,
        *,
        tool_call_id: str | None = None,
    ) -> None:
        """Display tool result using override/spec/fallback resolution."""
        if self._render_coordinator_tool_result(
            tool_name, result, tool_call_id=tool_call_id
        ):
            return

        tool = self._get_tool(tool_name)
        if tool:
            if self._replay_messages is not None and hasattr(
                tool, "set_replay_messages"
            ):
                tool.set_replay_messages(self._replay_messages)
            if tool.render_result(self.console, result):
                return

        spec = tool.display_spec if tool else None
        if spec:
            self._spec_renderer.render_result(self.console, tool_name, result, spec)
            return

        self._render_fallback_result(result)

    def render_tool_result_html(
        self, tool_name: str, result: object, args: dict | None = None
    ) -> str:
        tool = self._get_tool(tool_name)
        if tool:
            if self._replay_messages is not None and hasattr(
                tool, "set_replay_messages"
            ):
                tool.set_replay_messages(self._replay_messages)
            html = tool.render_result_html(result)
            if html is not None:
                return html
        spec = tool.display_spec if tool else None
        if spec:
            return self._spec_renderer.render_result_html(
                tool_name, result, spec, args=args
            )
        return self._render_fallback_result_html(result)

    def show_error(self, error_message: str):
        """Display error message."""
        self.console.print(f"\n[error]Error:[/error] {error_message}")

    def show_processing(self, message: str):
        """Display processing message."""
        self.console.print()  # Add newline
        return self.console.status(
            f"[status]{message}[/status]", spinner="bouncingBall"
        )

    def show_newline(self):
        """Display a newline for spacing."""
        self.console.print()

    def _get_tool(self, tool_name: str):
        try:
            return tool_registry.get_tool(tool_name)
        except KeyError:
            return None

    def _render_coordinator_tool_executing(
        self, tool_name: str, tool_input: dict, *, tool_call_id: str | None
    ) -> bool:
        if tool_name == "ask_database":
            database_id = str(tool_input.get("database_id") or "database")
            question = str(tool_input.get("question") or "").strip()
            context = {
                "database_id": database_id,
                "question": question,
            }
            if tool_call_id:
                self._pending_ask_database_contexts[tool_call_id] = context
            else:
                self._pending_ask_database_context = context

            if self.console.is_terminal:
                line = Text("Asking database ", style=self.tm.style("muted"))
                line.append(database_id, style=self.tm.style("info"))
                self.console.print(line)
                if question:
                    question_text = Text("Question: ", style=self.tm.style("muted"))
                    question_text.append(question)
                    self.console.print(question_text)
            else:
                self.console.print(f"**Asking database {database_id}**")
                if question:
                    self.console.print(f"Question: {question}")
            return True

        if tool_name == "list_connected_databases":
            if self.console.is_terminal:
                self.console.print(
                    "[muted bold]Checking connected databases[/muted bold]"
                )
            else:
                self.console.print("**Checking connected databases**")
            return True

        return False

    def _render_coordinator_tool_result(
        self, tool_name: str, result: object, *, tool_call_id: str | None
    ) -> bool:
        if tool_name == "ask_database":
            self._render_ask_database_result(result, tool_call_id=tool_call_id)
            return True

        if tool_name == "list_connected_databases":
            self._render_connected_databases(result)
            return True

        return False

    def _render_ask_database_result(
        self, result: object, *, tool_call_id: str | None
    ) -> None:
        database_name, database_id, thread_id, answer_text = (
            self._split_child_answer_result(str(result))
        )
        context = self._pop_ask_database_context(tool_call_id)
        if not database_id:
            database_id = context.get("database_id")
        question = context.get("question", "").strip()

        title_target = database_name or database_id or "database"
        panel_title = f"Subagent answer: {title_target}"
        if database_id and database_id != title_target:
            panel_title = f"{panel_title} ({database_id})"

        renderables = []
        if question:
            question_text = Text()
            question_text.append("Question: ", style=self.tm.style("muted"))
            question_text.append(question)
            renderables.append(question_text)
        if thread_id:
            thread_text = Text()
            thread_text.append("Thread: ", style=self.tm.style("muted"))
            thread_text.append(thread_id, style=self.tm.style("muted"))
            renderables.append(thread_text)
        if renderables:
            renderables.append(Text(""))
        renderables.append(
            Markdown(answer_text, code_theme=self.tm.pygments_style_name)
        )

        self.console.print(
            Panel(
                Group(*renderables),
                title=panel_title,
                border_style=self.tm.style("panel.border.assistant"),
                expand=True,
            )
        )

    def _pop_ask_database_context(self, tool_call_id: str | None) -> dict[str, str]:
        if tool_call_id:
            context = self._pending_ask_database_contexts.pop(tool_call_id, None)
            if context is not None:
                return context

        context = self._pending_ask_database_context or {}
        self._pending_ask_database_context = None
        return context

    def _split_child_answer_result(
        self, result: str
    ) -> tuple[str | None, str | None, str | None, str]:
        lines = result.strip().splitlines()
        if not lines:
            return None, None, None, ""

        database_name: str | None = None
        database_id: str | None = None
        thread_id: str | None = None
        body_start = 0

        first = lines[0].strip()
        if first.startswith("Database: "):
            database_label = first.removeprefix("Database: ").strip()
            if database_label.endswith(")") and " (id: " in database_label:
                database_name, raw_id = database_label.rsplit(" (id: ", 1)
                database_id = raw_id[:-1]
            else:
                database_name = database_label
            body_start = 1

        if len(lines) > body_start:
            candidate = lines[body_start].strip()
            if candidate.startswith("Child thread ID: "):
                thread_id = candidate.removeprefix("Child thread ID: ").strip()
                body_start += 1

        answer_text = "\n".join(lines[body_start:]).strip()
        return database_name, database_id, thread_id, answer_text

    def _render_connected_databases(self, result: object) -> None:
        rows = self._coerce_database_rows(result)
        if not rows:
            self.console.print("[muted]No connected databases.[/muted]")
            return

        table = Table(title="Connected databases")
        table.add_column("ID", style=self.tm.style("info"))
        table.add_column("Name")
        table.add_column("Type", style=self.tm.style("muted"))
        table.add_column("Description", style=self.tm.style("muted"))
        for row in rows:
            table.add_row(
                str(row.get("id") or ""),
                str(row.get("name") or ""),
                str(row.get("type") or ""),
                str(row.get("description") or row.get("summary") or ""),
            )
        self.console.print(table)

    def _coerce_database_rows(self, result: object) -> list[dict[str, object]]:
        data = result
        if isinstance(result, str):
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                return []

        if not isinstance(data, list):
            return []

        rows: list[dict[str, object]] = []
        for item in data:
            if isinstance(item, dict):
                rows.append({str(key): value for key, value in item.items()})
                continue
            model_dump = getattr(item, "model_dump", None)
            if callable(model_dump):
                dumped = model_dump()
                if isinstance(dumped, dict):
                    rows.append({str(key): value for key, value in dumped.items()})
        return rows

    def _render_fallback_result(self, result: object) -> None:
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                self._render_fallback_result(parsed)
                return
            except json.JSONDecodeError:
                if self.console.is_terminal:
                    self.console.print(result)
                else:
                    self.console.print(f"```\n{result}\n```\n")
                return

        if isinstance(result, (dict, list)):
            if self.console.is_terminal:
                self.console.print_json(json.dumps(result, ensure_ascii=False))
            else:
                self.console.print(
                    f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```\n"
                )
            return

        if self.console.is_terminal:
            self.console.print(str(result))
        else:
            self.console.print(f"```\n{result}\n```\n")

    def _render_fallback_result_html(self, result: object) -> str:
        spec = ToolDisplaySpec(result=ResultConfig(format="json"))
        return self._spec_renderer.render_result_html("tool", result, spec)

    def show_markdown_response(self, content: list):
        """Display the assistant's response as rich markdown in a panel."""
        if not content:
            return

        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)

        # Join all text parts and display as markdown in a panel
        full_text = "".join(text_parts).strip()
        if full_text:
            self.console.print()  # Add spacing before panel
            markdown = Markdown(full_text, code_theme=self.tm.pygments_style_name)
            panel = Panel.fit(
                markdown, border_style=self.tm.style("panel.border.assistant")
            )
            self.console.print(panel)
            self.console.print()  # Add spacing after panel

    def show_session_summary(self, session_usage: "SessionUsage") -> None:
        """Display session summary on exit.

        Shows final context size, total output tokens generated, and request/tool counts.
        """
        if not self.console.is_terminal:
            return

        if session_usage.requests == 0:
            return

        self.console.print()
        self.console.print("[muted]Session Summary[/muted]")
        self.console.print("[muted]" + "─" * 40 + "[/muted]")

        tokens_line = Text()
        tokens_line.append("Input: ", style="muted")
        tokens_line.append(
            f"{session_usage.current_context_tokens:,} tokens",
            style="muted bold",
        )
        self.console.print(tokens_line)

        output_line = Text()
        output_line.append("Output (total): ", style="muted")
        output_line.append(
            f"{session_usage.total_output_tokens:,} tokens",
            style="muted bold",
        )
        self.console.print(output_line)

        stats_line = Text()
        stats_line.append("Requests: ", style="muted")
        stats_line.append(str(session_usage.requests), style="muted bold")
        stats_line.append(" │ ", style="muted")
        stats_line.append("Tool calls: ", style="muted")
        stats_line.append(str(session_usage.tool_calls), style="muted bold")
        self.console.print(stats_line)

        self.console.print("[muted]" + "─" * 40 + "[/muted]")

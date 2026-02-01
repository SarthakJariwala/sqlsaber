"""Display utilities for the CLI interface.

All rendering occurs on the event loop thread.
Streaming segments use Live Markdown; transient status and SQL blocks are also
rendered with Live.
"""

import json
from typing import TYPE_CHECKING, Type

from pydantic_ai.messages import ModelResponsePart, TextPart, ThinkingPart
from rich.columns import Columns
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
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

    def show_tool_executing(self, tool_name: str, tool_input: dict):
        """Display tool execution details."""
        self.show_newline()
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

    def show_tool_result(self, tool_name: str, result: object) -> None:
        """Display tool result using override/spec/fallback resolution."""
        tool = self._get_tool(tool_name)
        if tool and tool.render_result(self.console, result):
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

"""Display utilities for the CLI interface.

All rendering occurs on the event loop thread.
Streaming segments use Live Markdown; transient status and SQL blocks are also
rendered with Live.
"""

import json
from typing import Sequence, Type

from pydantic_ai.messages import ModelResponsePart, TextPart, ThinkingPart
from rich.columns import Columns
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from tabulate import tabulate

from sqlsaber.theme.manager import get_theme_manager


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

    def _tables_to_markdown(self, data: dict) -> str:
        """Convert tables list JSON to markdown table."""
        rows = data.get("tables", [])
        cols = ["name", "schema", "type", "table_comment"]
        table_data = [[row.get(c, "") for c in cols] for row in rows]
        return tabulate(table_data, headers=cols, tablefmt="github")

    def _schema_to_markdown(self, data: dict) -> str:
        """Convert schema JSON to markdown tables."""
        parts = []
        for table_name, table_info in data.items():
            columns = table_info.get("columns", {})
            table_data = [
                [
                    col_name,
                    info["type"],
                    info["nullable"],
                    info["default"],
                    info.get("comment", ""),
                ]
                for col_name, info in columns.items()
            ]
            md = f"**Table: {table_name}**\n\n"
            md += tabulate(
                table_data,
                headers=["Column", "Type", "Nullable", "Default", "Comments"],
                tablefmt="github",
            )
            if table_info.get("primary_keys"):
                md += f"\n\n**Primary Keys:** {', '.join(table_info['primary_keys'])}"
            if table_info.get("foreign_keys"):
                md += f"\n\n**Foreign Keys:** {', '.join(table_info['foreign_keys'])}"
            parts.append(md)
        return "\n\n".join(parts)

    def _results_to_markdown(self, results: list) -> str:
        """Convert query results to markdown table."""
        if not results:
            return "*No results*"
        return tabulate(results, headers="keys", tablefmt="github")

    def _create_table(
        self,
        columns: Sequence[str | dict[str, str]],
        header_style: str | None = None,
        title: str | None = None,
    ) -> Table:
        """Create a Rich table with specified columns."""
        header_style = header_style or self.tm.style("table.header")
        table = Table(show_header=True, header_style=header_style, title=title)
        for col in columns:
            if isinstance(col, dict):
                table.add_column(col["name"], style=col.get("style"))
            else:
                table.add_column(col)
        return table

    def show_tool_executing(self, tool_name: str, tool_input: dict):
        """Display tool execution details."""
        # Normalized leading blank line before tool headers
        self.show_newline()
        if tool_name == "list_tables":
            if self.console.is_terminal:
                self.console.print(
                    "[muted bold]:gear: Discovering available tables[/muted bold]"
                )
            else:
                self.console.print("**Discovering available tables**\n")
        elif tool_name == "introspect_schema":
            pattern = tool_input.get("table_pattern", "all tables")
            if self.console.is_terminal:
                self.console.print(
                    f"[muted bold]:gear: Examining schema for: {pattern}[/muted bold]"
                )
            else:
                self.console.print(f"**Examining schema for:** {pattern}\n")
        elif tool_name == "execute_sql":
            # For streaming, we render SQL via LiveMarkdownRenderer; keep Syntax
            # rendering for threads show/resume. Controlled by include_sql flag.
            query = tool_input.get("query", "")
            if self.console.is_terminal:
                self.console.print("[muted bold]:gear: Executing SQL:[/muted bold]")
                self.show_newline()
                syntax = Syntax(
                    query,
                    "sql",
                    theme=self.tm.pygments_style_name,
                    background_color="default",
                    word_wrap=True,
                )
                self.console.print(syntax)
            else:
                self.console.print("**Executing SQL:**\n")
                self.console.print(f"```sql\n{query}\n```\n")

    def show_text_stream(self, text: str):
        """Display streaming text."""
        if text is not None:  # Extra safety check
            self.console.print(text, end="", markup=False)

    def show_query_results(self, results: list):
        """Display query results in a formatted table."""
        if not results:
            return

        if not self.console.is_terminal:
            # Markdown output for redirected/piped output
            self.console.print(f"\n**Results ({len(results)} rows):**\n")
            self.console.print(self._results_to_markdown(results))
            self.console.print()
            return

        # Rich table for terminal
        self.console.print(f"\n[section]Results ({len(results)} rows):[/section]")

        all_columns = list(results[0].keys())
        display_columns = all_columns[:15]

        if len(all_columns) > 15:
            self.console.print(
                f"[warning]Note: Showing first 15 of {len(all_columns)} columns[/warning]"
            )

        table = self._create_table(display_columns)

        for row in results[:20]:
            table.add_row(*[str(row[key]) for key in display_columns])

        self.console.print(table)

        if len(results) > 20:
            self.console.print(
                f"[warning]... and {len(results) - 20} more rows[/warning]"
            )

    def show_error(self, error_message: str):
        """Display error message."""
        self.console.print(f"\n[error]Error:[/error] {error_message}")

    def show_sql_error(self, error_message: str, suggestions: list[str] | None = None):
        """Display SQL-specific error with optional suggestions."""
        self.show_newline()
        self.console.print(f"[error]SQL error:[/error] {error_message}")
        if suggestions:
            self.console.print("[warning]Hints:[/warning]")
            for suggestion in suggestions:
                self.console.print(f"  • {suggestion}")

    def show_processing(self, message: str):
        """Display processing message."""
        self.console.print()  # Add newline
        return self.console.status(
            f"[status]{message}[/status]", spinner="bouncingBall"
        )

    def show_newline(self):
        """Display a newline for spacing."""
        self.console.print()

    def show_table_list(self, tables_data: str | dict):
        """Display the results from list_tables tool."""
        try:
            data = (
                json.loads(tables_data) if isinstance(tables_data, str) else tables_data
            )

            if "error" in data:
                self.show_error(data["error"])
                return

            tables = data.get("tables", [])
            total_tables = data.get("total_tables", 0)

            if not tables:
                if self.console.is_terminal:
                    self.console.print(
                        "[warning]No tables found in the database.[/warning]"
                    )
                else:
                    self.console.print("*No tables found in the database.*\n")
                return

            if not self.console.is_terminal:
                # Markdown output for redirected/piped output
                self.console.print(f"\n**Database Tables ({total_tables} total):**\n")
                self.console.print(self._tables_to_markdown(data))
                self.console.print()
                return

            # Rich table for terminal
            self.console.print(
                f"\n[title]Database Tables ({total_tables} total):[/title]"
            )

            columns = [
                {"name": "Schema", "style": "column.schema"},
                {"name": "Table Name", "style": "column.name"},
                {"name": "Type", "style": "column.type"},
            ]
            table = self._create_table(columns)

            for table_info in tables:
                schema = table_info.get("schema", "")
                name = table_info.get("name", "")
                table_type = table_info.get("type", "")
                table.add_row(schema, name, table_type)

            self.console.print(table)

        except json.JSONDecodeError:
            self.show_error("Failed to parse table list data")
        except Exception as e:
            self.show_error(f"Error displaying table list: {str(e)}")

    def show_schema_info(self, schema_data: str | dict):
        """Display the results from introspect_schema tool."""
        try:
            data = (
                json.loads(schema_data) if isinstance(schema_data, str) else schema_data
            )

            if "error" in data:
                self.show_error(data["error"])
                return

            if not data:
                if self.console.is_terminal:
                    self.console.print(
                        "[warning]No schema information found.[/warning]"
                    )
                else:
                    self.console.print("*No schema information found.*\n")
                return

            if not self.console.is_terminal:
                # Markdown output for redirected/piped output
                self.console.print(f"\n**Schema Information ({len(data)} tables):**\n")
                self.console.print(self._schema_to_markdown(data))
                self.console.print()
                return

            # Rich tables for terminal
            self.console.print(
                f"\n[title]Schema Information ({len(data)} tables):[/title]"
            )

            for table_name, table_info in data.items():
                self.console.print(f"\n[heading]Table: {table_name}[/heading]")

                table_comment = table_info.get("comment")
                if table_comment:
                    self.console.print(f"[muted]Comment: {table_comment}[/muted]")

                table_columns = table_info.get("columns", {})
                if table_columns:
                    include_column_comments = any(
                        col_info.get("comment") for col_info in table_columns.values()
                    )

                    columns = [
                        {"name": "Column Name", "style": "column.name"},
                        {"name": "Type", "style": "column.type"},
                        {"name": "Nullable", "style": "info"},
                        {"name": "Default", "style": "muted"},
                    ]
                    if include_column_comments:
                        columns.append({"name": "Comment", "style": "muted"})
                    col_table = self._create_table(columns, title="Columns")

                    for col_name, col_info in table_columns.items():
                        nullable = "✓" if col_info.get("nullable", False) else "✗"
                        default = (
                            str(col_info.get("default", ""))
                            if col_info.get("default")
                            else ""
                        )
                        row = [
                            col_name,
                            col_info.get("type", ""),
                            nullable,
                            default,
                        ]
                        if include_column_comments:
                            row.append(col_info.get("comment") or "")
                        col_table.add_row(*row)

                    self.console.print(col_table)

                primary_keys = table_info.get("primary_keys", [])
                if primary_keys:
                    self.console.print(
                        f"[key.primary]Primary Keys:[/key.primary] {', '.join(primary_keys)}"
                    )

                foreign_keys = table_info.get("foreign_keys", [])
                if foreign_keys:
                    self.console.print("[key.foreign]Foreign Keys:[/key.foreign]")
                    for fk in foreign_keys:
                        self.console.print(f"  • {fk}")

                indexes = table_info.get("indexes", [])
                if indexes:
                    self.console.print("[key.index]Indexes:[/key.index]")
                    for idx in indexes:
                        self.console.print(f"  • {idx}")

        except json.JSONDecodeError:
            self.show_error("Failed to parse schema data")
        except Exception as e:
            self.show_error(f"Error displaying schema information: {str(e)}")

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

"""Tool display specifications and rendering helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html import escape
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from tabulate import tabulate

from sqlsaber.theme.manager import ThemeManager, get_theme_manager

ResultFormat = Literal["auto", "json", "panel", "code", "table", "key_value"]
ShowArgs = Literal["all", "none"]


@dataclass(frozen=True)
class ColumnDef:
    """Definition for a table column."""

    field: str
    header: str
    style: str | None = None


@dataclass(frozen=True)
class TableConfig:
    """Configuration for table format rendering."""

    columns: list[ColumnDef]
    max_rows: int = 20


@dataclass(frozen=True)
class FieldMappings:
    """Maps JSON fields to semantic roles."""

    output: str = "output"
    error: str = "error"
    success: str | None = "success"
    items: str | None = None


@dataclass(frozen=True)
class ExecutingConfig:
    """Configuration for 'tool executing' display."""

    message: str = "{tool_name}"
    icon: str | None = "⚙️"
    show_args: list[str] | ShowArgs = "none"


@dataclass(frozen=True)
class ResultConfig:
    """Configuration for tool result display."""

    format: ResultFormat = "auto"
    title: str | None = None
    success_style: str = "green"
    error_style: str = "red"
    code_language: str | None = None
    fields: FieldMappings = field(default_factory=FieldMappings)
    table: TableConfig | None = None


@dataclass(frozen=True)
class DisplayMetadata:
    """Metadata about the tool for display purposes."""

    display_name: str = ""
    description: str | None = None


@dataclass(frozen=True)
class ToolDisplaySpec:
    """Complete display specification for a tool."""

    executing: ExecutingConfig = field(default_factory=ExecutingConfig)
    result: ResultConfig = field(default_factory=ResultConfig)
    metadata: DisplayMetadata = field(default_factory=DisplayMetadata)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class SpecRenderer:
    """Render tool display specs for terminal and HTML contexts."""

    def __init__(self, theme_manager: ThemeManager | None = None):
        self.tm = theme_manager or get_theme_manager()

    def render_executing(
        self,
        console: Console,
        tool_name: str,
        tool_args: dict[str, Any],
        spec: ToolDisplaySpec,
    ) -> None:
        config = spec.executing
        message = self._format_template(
            config.message, {"tool_name": tool_name, **tool_args}
        )
        icon = f"{config.icon} " if config.icon else ""
        line = f"{icon}{message}".strip()

        if console.is_terminal:
            console.print(f"[muted bold]{line}[/muted bold]")
        else:
            console.print(f"**{line}**\n")

        args_to_show = self._resolve_args_to_show(config.show_args, tool_args)
        if not args_to_show:
            return

        if console.is_terminal:
            table = Table(show_header=True, header_style=self.tm.style("table.header"))
            table.add_column("Arg", style=self.tm.style("info"))
            table.add_column("Value", style=self.tm.style("muted"))
            for key, value in args_to_show.items():
                table.add_row(str(key), json.dumps(value, ensure_ascii=False))
            console.print(table)
        else:
            for key, value in args_to_show.items():
                console.print(f"- **{key}**: {value}")
            console.print()

    def render_result(
        self,
        console: Console,
        tool_name: str,
        result: object,
        spec: ToolDisplaySpec,
    ) -> None:
        data, raw = self._parse_result(result)
        config = spec.result

        error_text = self._extract_error(data, config.fields)
        success_value = self._extract_success(data, config.fields)

        if error_text:
            self._render_error(console, error_text, config.error_style)
            return

        fmt = self._resolve_format(config.format, data, config.fields)
        title = self._format_title(config.title, data)
        output = self._extract_output(data, config.fields)

        if fmt == "json":
            self._render_json(console, data, raw)
        elif fmt == "code":
            self._render_code(console, output, config.code_language or "")
        elif fmt == "panel":
            self._render_panel(console, output, title, config.success_style)
        elif fmt == "table":
            self._render_table(console, output, config.table, title)
        elif fmt == "key_value":
            self._render_key_value(console, output)
        else:
            self._render_json(console, data, raw)

        if success_value is False and not error_text:
            self._render_error(
                console, "Operation reported failure.", config.error_style
            )

    def render_result_html(
        self,
        tool_name: str,
        result: object,
        spec: ToolDisplaySpec,
        args: dict[str, Any] | None = None,
    ) -> str:
        data, raw = self._parse_result(result)
        config = spec.result

        error_text = self._extract_error(data, config.fields)
        if error_text:
            return f'<div class="sql-error"><strong>Error:</strong> {escape(error_text)}</div>'

        fmt = self._resolve_format(config.format, data, config.fields)
        title = self._format_title(config.title, data)
        output = self._extract_output(data, config.fields)

        if fmt == "json":
            return self._render_json_html(data, raw)
        if fmt == "code":
            language = config.code_language or ""
            return self._render_code_html(output, language)
        if fmt == "panel":
            return self._render_panel_html(output, title)
        if fmt == "table":
            return self._render_table_html(output, config.table, title)
        if fmt == "key_value":
            return self._render_key_value_html(output)

        return self._render_json_html(data, raw)

    def _format_template(self, template: str, values: dict[str, Any]) -> str:
        try:
            return template.format_map(_SafeFormatDict(values))
        except Exception:
            return template

    def _format_title(self, title: str | None, data: object) -> str | None:
        if not title:
            return None
        values = self._coerce_mapping(data) or {}
        return self._format_template(title, {"result": data, **values})

    def _resolve_args_to_show(
        self, show_args: list[str] | ShowArgs, tool_args: dict[str, Any]
    ) -> dict[str, Any]:
        if show_args == "none":
            return {}
        if show_args == "all":
            return tool_args
        return {key: tool_args[key] for key in show_args if key in tool_args}

    def _parse_result(self, result: object) -> tuple[object, str | None]:
        if isinstance(result, (dict, list)):
            return result, None
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                return result, result
            return parsed, result
        return {"output": str(result)}, None

    def _extract_output(self, data: object, fields: FieldMappings) -> object:
        mapping = self._coerce_mapping(data)
        if mapping is not None:
            if fields.items and fields.items in mapping:
                return mapping[fields.items]
            if fields.output in mapping:
                return mapping[fields.output]
            return mapping
        return data

    def _extract_error(self, data: object, fields: FieldMappings) -> str | None:
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return None
        if fields.error in mapping and mapping[fields.error]:
            return str(mapping[fields.error])
        if "error" in mapping and mapping["error"]:
            return str(mapping["error"])
        if "stderr" in mapping and mapping["stderr"]:
            return str(mapping["stderr"])
        return None

    def _extract_success(self, data: object, fields: FieldMappings) -> bool | None:
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return None
        if fields.success and fields.success in mapping:
            value = mapping[fields.success]
            if isinstance(value, bool):
                return value
        return None

    def _resolve_format(
        self, fmt: ResultFormat, data: object, fields: FieldMappings
    ) -> ResultFormat:
        if fmt != "auto":
            return fmt
        output = self._extract_output(data, fields)
        if isinstance(output, list):
            return "table"
        if isinstance(output, dict):
            return "key_value"
        if isinstance(output, str):
            return "panel"
        if isinstance(data, list):
            return "table"
        if isinstance(data, dict):
            return "key_value"
        return "json"

    def _render_error(self, console: Console, message: str, style: str) -> None:
        if console.is_terminal:
            console.print(f"[{style}]Error:[/{style}] {message}")
        else:
            console.print(f"**Error:** {message}\n")

    def _render_json(self, console: Console, data: object, raw: str | None) -> None:
        if console.is_terminal:
            if isinstance(data, (dict, list)):
                console.print_json(json.dumps(data, ensure_ascii=False))
            else:
                console.print(str(data))
            return
        if raw is not None:
            console.print(f"```json\n{raw}\n```\n")
        else:
            console.print(
                f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```\n"
            )

    def _render_code(self, console: Console, output: object, language: str) -> None:
        code = "" if output is None else str(output)
        if console.is_terminal:
            console.print(
                Syntax(
                    code,
                    language or "text",
                    theme=self.tm.pygments_style_name,
                    background_color="default",
                    word_wrap=True,
                )
            )
        else:
            console.print(f"```{language}\n{code}\n```\n")

    def _render_panel(
        self, console: Console, output: object, title: str | None, style: str
    ) -> None:
        body = "" if output is None else str(output)
        if console.is_terminal:
            panel = Panel(body, title=title, border_style=style)
            console.print(panel)
        else:
            if title:
                console.print(f"**{title}**\n")
            console.print(body)
            console.print()

    def _render_table(
        self,
        console: Console,
        output: object,
        config: TableConfig | None,
        title: str | None,
    ) -> None:
        rows = self._normalize_rows(output)
        if not rows:
            if console.is_terminal:
                console.print("[warning]No results[/warning]")
            else:
                console.print("*No results*\n")
            return

        if console.is_terminal:
            table = Table(
                show_header=True,
                header_style=self.tm.style("table.header"),
                title=title,
            )
            columns = self._resolve_columns(rows, config)
            for col in columns:
                table.add_column(col.header, style=col.style)
            max_rows = config.max_rows if config else 20
            for row in rows[:max_rows]:
                table.add_row(
                    *(self._stringify_value(row.get(col.field)) for col in columns)
                )
            console.print(table)
            if len(rows) > max_rows:
                console.print(
                    f"[warning]... and {len(rows) - max_rows} more rows[/warning]"
                )
            return

        columns = self._resolve_columns(rows, config)
        headers = [col.header for col in columns]
        table_data = [
            [self._stringify_value(row.get(col.field)) for col in columns]
            for row in rows
        ]
        if title:
            console.print(f"**{title}**\n")
        console.print(tabulate(table_data, headers=headers, tablefmt="github"))
        console.print()

    def _render_key_value(self, console: Console, output: object) -> None:
        if not isinstance(output, dict):
            console.print(str(output))
            return
        if console.is_terminal:
            table = Table(show_header=True, header_style=self.tm.style("table.header"))
            table.add_column("Key", style=self.tm.style("info"))
            table.add_column("Value", style=self.tm.style("muted"))
            for key, value in output.items():
                table.add_row(str(key), self._stringify_value(value))
            console.print(table)
        else:
            for key, value in output.items():
                console.print(f"- **{key}**: {self._stringify_value(value)}")
            console.print()

    def _normalize_rows(self, output: object) -> list[dict[str, Any]]:
        if output is None:
            return []
        if isinstance(output, list):
            rows: list[dict[str, Any]] = []
            for item in output:
                mapping = self._coerce_mapping(item)
                if mapping is not None:
                    rows.append(mapping)
                else:
                    rows.append({"value": item})
            return rows
        mapping = self._coerce_mapping(output)
        if mapping is not None:
            return [mapping]
        return [{"value": output}]

    def _coerce_mapping(self, data: object) -> dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        return {str(key): value for key, value in data.items()}

    def _resolve_columns(
        self, rows: list[dict[str, Any]], config: TableConfig | None
    ) -> list[ColumnDef]:
        if config and config.columns:
            return config.columns
        if not rows:
            return [ColumnDef(field="value", header="Value")]
        keys = list(rows[0].keys())
        return [ColumnDef(field=key, header=key) for key in keys]

    def _stringify_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _render_json_html(self, data: object, raw: str | None) -> str:
        if raw is not None:
            return f"<pre><code>{escape(raw)}</code></pre>"
        return f"<pre><code>{escape(json.dumps(data, ensure_ascii=False, indent=2))}</code></pre>"

    def _render_code_html(self, output: object, language: str) -> str:
        code = "" if output is None else str(output)
        class_attr = f' class="language-{escape(language)}"' if language else ""
        return f"<pre><code{class_attr}>{escape(code)}</code></pre>"

    def _render_panel_html(self, output: object, title: str | None) -> str:
        body = "" if output is None else str(output)
        title_html = f'<p class="result-count">{escape(title)}</p>' if title else ""
        return f"{title_html}<pre><code>{escape(body)}</code></pre>"

    def _render_table_html(
        self, output: object, config: TableConfig | None, title: str | None
    ) -> str:
        rows = self._normalize_rows(output)
        if not rows:
            return '<p class="result-count">No results</p>'
        columns = self._resolve_columns(rows, config)
        header_html = "".join(f"<th>{escape(col.header)}</th>" for col in columns)
        max_rows = config.max_rows if config else 100
        row_html = []
        for row in rows[:max_rows]:
            cells = "".join(
                f"<td>{escape(self._stringify_value(row.get(col.field)))}</td>"
                for col in columns
            )
            row_html.append(f"<tr>{cells}</tr>")
        title_html = f'<p class="result-count">{escape(title)}</p>' if title else ""
        count_note = ""
        if len(rows) > max_rows:
            count_note = (
                f'<p class="result-count">Showing {max_rows} of {len(rows)} rows</p>'
            )
        return (
            f'{title_html}{count_note}<div class="table-wrapper">'
            f'<table class="sql-results"><thead><tr>{header_html}</tr></thead>'
            f"<tbody>{''.join(row_html)}</tbody></table></div>"
        )

    def _render_key_value_html(self, output: object) -> str:
        if not isinstance(output, dict):
            return f"<pre><code>{escape(str(output))}</code></pre>"
        rows = "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(self._stringify_value(value))}</td></tr>"
            for key, value in output.items()
        )
        return (
            '<div class="table-wrapper"><table class="sql-results">'
            f"<tbody>{rows}</tbody></table></div>"
        )

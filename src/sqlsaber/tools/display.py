"""Tool display specifications and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rich.console import Console

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


class SpecRenderer:
    """Render tool display specs. Delegates to spec_blocks plus md_of/html_of."""

    def __init__(self, theme_manager: ThemeManager | None = None):
        self.tm = theme_manager or get_theme_manager()

    def render_executing(
        self,
        console: Console,
        tool_name: str,
        tool_args: dict[str, Any],
        spec: ToolDisplaySpec,
    ) -> None:
        from sqlsaber.render.markdown_text import md_of
        from sqlsaber.tools.spec_blocks import blocks_from_spec_executing

        text = md_of(blocks_from_spec_executing(tool_name, tool_args, spec))
        if text:
            console.print(text, markup=False)

    def render_result(
        self,
        console: Console,
        tool_name: str,
        result: object,
        spec: ToolDisplaySpec,
    ) -> None:
        from sqlsaber.render.markdown_text import md_of
        from sqlsaber.tools.spec_blocks import blocks_from_spec_result

        text = md_of(blocks_from_spec_result(tool_name, result, spec))
        if text:
            console.print(text, markup=False)

    def render_result_html(
        self,
        tool_name: str,
        result: object,
        spec: ToolDisplaySpec,
        args: dict[str, Any] | None = None,
    ) -> str:
        del args
        from sqlsaber.render.html import html_of
        from sqlsaber.tools.spec_blocks import blocks_from_spec_result

        return html_of(blocks_from_spec_result(tool_name, result, spec))

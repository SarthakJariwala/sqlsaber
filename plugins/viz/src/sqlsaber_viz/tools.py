"""Visualization tool implementation."""

from __future__ import annotations

import asyncio
import json
import re
from html import escape

from pydantic import ValidationError
from pydantic_ai import RunContext
from rich.console import Console
from rich.text import Text

from sqlsaber.tools.base import Tool
from sqlsaber.utils.json_utils import json_dumps

from .data_loader import (
    extract_data_summary,
    find_tool_output_in_messages,
    find_tool_output_payload,
)
from .renderers.plotext_renderer import PlotextRenderer
from .spec import BarChart, LimitTransform, SortItem, SortTransform, VizSpec
from .transforms import apply_transforms

TOOL_OUTPUT_FILE_PATTERN = re.compile(r"^result_[A-Za-z0-9._-]+\.json$")
SPEC_TIMEOUT_SECONDS = 300


class VizTool(Tool):
    """Terminal visualization tool for SQL results."""

    requires_ctx = True

    def __init__(self):
        super().__init__()
        self._last_ctx: RunContext | None = None
        self._last_rows: list[dict] | None = None
        self._last_file: str | None = None
        self._replay_messages: list | None = None

    def set_replay_messages(self, messages: list) -> None:
        """Set message history for replay scenarios (e.g., threads show)."""
        self._replay_messages = messages

    @property
    def name(self) -> str:
        return "viz"

    def render_executing(self, console: Console, args: dict) -> bool:
        """Suppress default JSON rendering during execution."""
        return True

    async def execute(
        self,
        ctx: RunContext,
        request: str,
        file: str,
        chart_type: str | None = None,
    ) -> str:
        """Generate a visualization spec for SQL results.

        Args:
            request: Natural language description of the desired visualization.
            file: Result file key from execute_sql (e.g., "result_abc123.json").
            chart_type: Optional hint for chart type (bar, line, scatter, boxplot, histogram).

        Returns:
            JSON string containing the visualization spec.
        """
        self._last_ctx = ctx

        if not file or not TOOL_OUTPUT_FILE_PATTERN.match(file):
            return json_dumps({"error": "Invalid result file key format."})

        tool_call_id = file.removeprefix("result_").removesuffix(".json")
        payload = find_tool_output_payload(ctx, tool_call_id)
        if payload is None:
            return json_dumps({"error": "Tool output not found in message history."})

        summary = extract_data_summary(payload)
        columns = summary.get("columns", [])
        row_count = summary.get("row_count", 0)
        rows = summary.get("rows", [])

        self._last_rows = rows
        self._last_file = file

        agent = _get_spec_agent_cls()()

        try:
            spec = await asyncio.wait_for(
                agent.generate_spec(
                    request=request,
                    columns=columns,
                    row_count=row_count,
                    file=file,
                    chart_type_hint=chart_type,
                ),
                timeout=SPEC_TIMEOUT_SECONDS,
            )
            spec = self._ensure_bar_defaults(spec, row_count)
            return json_dumps(spec.model_dump())
        except asyncio.TimeoutError:
            return json_dumps(
                {
                    "error": "Spec generation timed out.",
                    "details": f"Timed out after {SPEC_TIMEOUT_SECONDS} seconds.",
                }
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            return json_dumps(
                {
                    "error": "Failed to generate a valid visualization spec.",
                    "details": str(exc),
                }
            )

    def render_result(self, console: Console, result: object) -> bool:
        """Render the spec as a terminal chart using plotext."""
        spec = self._parse_spec(result)
        if spec is None:
            return False

        rows = self._resolve_rows(spec)
        if rows is None:
            if console.is_terminal:
                console.print("[warning]No data available for visualization.[/warning]")
            else:
                console.print("*No data available for visualization.*\n")
            return True

        rows = apply_transforms(rows, spec.transform)

        renderer = PlotextRenderer()
        chart = renderer.render(spec, rows)
        if console.is_terminal:
            console.print(Text.from_ansi(chart))
        else:
            console.print(f"```\n{self._strip_ansi(chart)}\n```\n", markup=False)
        return True

    def render_result_html(self, result: object) -> str | None:
        """Render the spec as an HTML chart."""
        spec = self._parse_spec(result)
        if spec is None:
            return None

        rows = self._resolve_rows(spec)
        if rows is None:
            return '<div class="viz-error">No data available for visualization.</div>'

        rows = apply_transforms(rows, spec.transform)
        from .renderers.plotext_renderer import PlotextRenderer

        renderer = PlotextRenderer()
        chart = renderer.render(spec, rows)
        return f'<pre class="viz-chart">{escape(self._strip_ansi(chart))}</pre>'

    def _parse_spec(self, result: object) -> VizSpec | None:
        data = self._parse_result(result)
        if not isinstance(data, dict):
            return None
        if "error" in data and data["error"]:
            return None
        try:
            return VizSpec.model_validate(data)
        except ValidationError:
            return None

    def _parse_result(self, result: object) -> object:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": result}
        return {"error": str(result)}

    def _strip_ansi(self, text: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def _resolve_rows(self, spec: VizSpec) -> list[dict] | None:
        if self._last_rows is not None and self._last_file == spec.data.source.file:
            return self._last_rows

        tool_call_id = spec.data.source.file.removeprefix("result_").removesuffix(
            ".json"
        )

        payload: dict | None = None
        if self._last_ctx is not None:
            payload = find_tool_output_payload(self._last_ctx, tool_call_id)
        elif self._replay_messages is not None:
            payload = find_tool_output_in_messages(self._replay_messages, tool_call_id)

        if payload is None:
            return None
        summary = extract_data_summary(payload)
        rows = summary.get("rows")
        if isinstance(rows, list):
            return rows
        return None

    def _ensure_bar_defaults(self, spec: VizSpec, row_count: int) -> VizSpec:
        if not isinstance(spec.chart, BarChart):
            return spec

        transforms = list(spec.transform)
        has_limit = any(isinstance(t, LimitTransform) for t in transforms)
        has_sort = any(isinstance(t, SortTransform) for t in transforms)

        if not has_sort:
            transforms.append(
                SortTransform(
                    sort=[SortItem(field=spec.chart.encoding.y.field, dir="desc")]
                )
            )

        if not has_limit and row_count > 20:
            transforms.append(LimitTransform(limit=20))

        if transforms != spec.transform:
            return spec.model_copy(update={"transform": transforms})

        return spec


def _get_spec_agent_cls():
    from .spec_agent import SpecAgent

    return SpecAgent

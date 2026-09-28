"""Internal agent for generating visualization specs."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from sqlsaber.agents.model_factory import resolve_model
from sqlsaber.config.settings import Config

from .prompts import VIZ_SYSTEM_PROMPT
from .renderers.plotext_renderer import PlotextRenderer
from .spec import (
    BoxplotChart,
    FilterTransform,
    HistogramChart,
    LineChart,
    ScatterChart,
    SortTransform,
    VizSpec,
)
from .transforms import apply_transforms

MAX_RETRIES = 2


@dataclass
class SpecData:
    file: str
    rows: list[dict]


def validate_spec(ctx: RunContext[SpecData], spec: VizSpec) -> VizSpec:
    """Check references and renderable values without sending full rows to the model."""
    if spec.data.source.file != ctx.deps.file:
        raise ModelRetry(f"Use the requested source file: {ctx.deps.file}")

    chart = spec.chart
    if isinstance(chart, HistogramChart):
        fields = [chart.histogram.field]
        numeric_fields = fields.copy()
    elif isinstance(chart, BoxplotChart):
        fields = [chart.boxplot.label_field, chart.boxplot.value_field]
        numeric_fields = [chart.boxplot.value_field]
    else:
        fields = [chart.encoding.x.field, chart.encoding.y.field]
        numeric_fields = (
            fields.copy()
            if isinstance(chart, (LineChart, ScatterChart))
            else [chart.encoding.y.field]
        )
        if chart.encoding.series:
            fields.append(chart.encoding.series.field)

    for transform in spec.transform:
        if isinstance(transform, SortTransform):
            fields.extend(item.field for item in transform.sort)
        elif isinstance(transform, FilterTransform):
            fields.append(transform.filter.field)

    available = {key for row in ctx.deps.rows for key in row}
    missing = sorted(set(fields) - available)
    if missing:
        raise ModelRetry(
            f"Unknown fields: {missing}. Available fields: {sorted(available)}"
        )

    rows = apply_transforms(ctx.deps.rows, spec.transform)
    renderer = PlotextRenderer()
    if not any(
        all(renderer._to_number(row.get(field)) is not None for field in numeric_fields)
        for row in rows
    ):
        raise ModelRetry(
            f"No plottable values for {numeric_fields} after transforms. "
            "Check field choices and transforms without changing the user's intent."
        )
    return spec


class SpecAgent:
    """Internal agent for generating visualization specs."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        *,
        model: Model | str | None = None,
    ):
        self.config = Config()
        self._model_name_override = model_name
        self._api_key_override = api_key
        self._model = model
        self.agent = self._build_agent()

    def _build_agent(self):
        model = self._model
        if model is None:
            resolved = resolve_model(
                self.config.auth,
                self._model_name_override or self.config.model.name,
                api_key_override=self._api_key_override,
            )
            model = resolved.model
        agent = Agent(
            model,
            instructions=VIZ_SYSTEM_PROMPT,
            output_type=VizSpec,
            deps_type=SpecData,
            retries=MAX_RETRIES,
        )
        agent.output_validator(validate_spec)
        return agent

    async def generate_spec(
        self,
        request: str,
        columns: list[dict],
        row_count: int,
        file: str,
        chart_type_hint: str | None = None,
        *,
        rows: list[dict],
    ) -> VizSpec:
        """Generate a VizSpec from user request and data summary.

        Pydantic AI retries schema and data validation failures in the same run.
        Full rows are local validation dependencies, not part of the prompt.

        Args:
            request: Natural language viz request.
            columns: Column metadata from the data summary.
            row_count: Number of rows in the result set.
            file: Result file key.
            chart_type_hint: Optional chart type hint.
            rows: Complete local result rows for reference/value checks.

        Returns:
            A validated VizSpec.
        """

        prompt = self._build_prompt(
            request=request,
            columns=columns,
            row_count=row_count,
            file=file,
            chart_type_hint=chart_type_hint,
        )

        result = await self.agent.run(prompt, deps=SpecData(file=file, rows=rows))
        return result.output

    def _build_prompt(
        self,
        request: str,
        columns: list[dict],
        row_count: int,
        file: str,
        chart_type_hint: str | None,
    ) -> str:
        columns_json = json.dumps(columns, ensure_ascii=False, indent=2)
        hint_text = f"Chart type hint: {chart_type_hint}" if chart_type_hint else ""

        return (
            "## User Request\n"
            f"{request.strip()}\n\n"
            "## Data Summary\n"
            f"Row count: {row_count}\n"
            f"File: {file}\n"
            f"Columns:\n{columns_json}\n\n"
            f"{hint_text}\n\n"
            "Return a VizSpec using the provided output schema."
        ).strip()

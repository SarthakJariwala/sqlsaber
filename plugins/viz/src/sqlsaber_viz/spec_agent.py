"""Internal agent for generating visualization specs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from sqlsaber.agents.provider_factory import ProviderFactory
from sqlsaber.config import providers
from sqlsaber.config.logging import get_logger
from sqlsaber.config.settings import Config

from .prompts import VIZ_SYSTEM_PROMPT
from .spec import VizSpec
from .templates import ChartType, list_chart_types, vizspec_template

logger = get_logger(__name__)

MAX_RETRIES = 2


class SpecAgent:
    """Internal agent for generating visualization specs."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.config = Config()
        self._model_name_override = model_name
        self._api_key_override = api_key
        self.agent = self._build_agent()

    def _build_agent(self):
        model_name = (
            self._model_name_override
            or self.config.model.get_subagent_model("viz")
            or self.config.model.name
        )
        model_name_only = (
            model_name.split(":", 1)[1] if ":" in model_name else model_name
        )

        if not (self._model_name_override and self._api_key_override):
            self.config.auth.validate(model_name)

        provider = providers.provider_from_model(model_name) or ""
        api_key = self._api_key_override or self.config.auth.get_api_key(model_name)

        factory = ProviderFactory()
        agent = factory.create_agent(
            provider=provider,
            model_name=model_name_only,
            full_model_str=model_name,
            api_key=api_key,
            thinking_enabled=False,
        )

        @agent.system_prompt
        def viz_system_prompt() -> str:
            return VIZ_SYSTEM_PROMPT

        self._register_tools(agent)

        return agent

    def _register_tools(self, agent) -> None:
        """Register visualization helper tools on the agent."""

        @agent.tool_plain
        def get_vizspec_template(chart_type: ChartType, file: str) -> dict:
            """Get the complete VizSpec template for a chart type.

            Call this FIRST to get the correct JSON structure, then fill in
            the placeholder field names with actual column names from your data.

            Args:
                chart_type: One of "bar", "line", "scatter", "boxplot", "histogram"
                file: The result file key (e.g., "result_abc123.json")

            Returns:
                A complete VizSpec template with placeholders for field names.
            """
            return vizspec_template(chart_type, file)

        @agent.tool_plain
        def get_available_chart_types() -> list[dict]:
            """List available chart types with descriptions.

            Call this if you're unsure which chart type to use for the data.

            Returns:
                List of chart types with descriptions and use cases.
            """
            return list_chart_types()

    async def generate_spec(
        self,
        request: str,
        columns: list[dict],
        row_count: int,
        file: str,
        chart_type_hint: str | None = None,
    ) -> VizSpec:
        """Generate a VizSpec from user request and data summary.

        Uses a retry loop that feeds validation errors back into the
        agent conversation so it can self-correct without losing context
        (e.g. the template it fetched and the chart type it chose).

        Args:
            request: Natural language viz request.
            columns: Column metadata from the data summary.
            row_count: Number of rows in the result set.
            file: Result file key.
            chart_type_hint: Optional chart type hint.

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

        message_history = None

        for attempt in range(MAX_RETRIES + 1):
            result = await self.agent.run(prompt, message_history=message_history)
            output = str(result.output).strip()

            try:
                parsed = _parse_json(output)
                return VizSpec.model_validate(parsed)
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                if attempt == MAX_RETRIES:
                    raise
                logger.debug(
                    "Spec validation failed (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    exc,
                )
                # Preserve the full conversation so the agent sees its
                # prior tool calls, reasoning, and failed output.
                message_history = result.all_messages()
                prompt = (
                    f"The spec you returned failed validation:\n{exc}\n\n"
                    "Fix the JSON and return ONLY the corrected spec."
                )

        # Unreachable, but satisfies type checkers.
        raise RuntimeError("Exhausted retries without raising")

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
            "Use `get_vizspec_template` to get the correct spec structure, "
            "then fill in the placeholders with actual column names.\n"
            "Return ONLY the final JSON."
        ).strip()


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected JSON object", text, 0)
    return parsed

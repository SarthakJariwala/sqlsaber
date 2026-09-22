"""Runtime for the bundled, model-callable handoff capability."""

import json
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.run_usage import current_usage_limits

from .prompts import HANDOFF_INPUT_INSTRUCTIONS, HANDOFF_SYSTEM_PROMPT


class Handoff(SqlSaberCapability):
    """Draft a continuation prompt without changing conversation or thread state."""

    id = "handoff"
    description = "Generate a context-aware draft for continuing in a fresh thread."

    def __init__(
        self,
        context: PluginContext,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.context = context
        self._model_name_override = model_name
        self._api_key_override = api_key
        self._toolset = FunctionToolset[Any](id=self.id)
        self._toolset.add_function(self.draft_handoff, takes_ctx=True)

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def update_context(self, context: PluginContext) -> None:
        self.context = context

    async def draft_handoff(self, ctx: RunContext[Any], goal: str) -> str:
        """Draft a prompt to continue this conversation in a fresh thread.

        Returns a draft only: it does not start a thread or clear history.
        The user decides whether to use it.

        Args:
            goal: What the user wants to accomplish in the new thread.
        """
        return await self.generate_draft(list(ctx.messages), goal, usage=ctx.usage)

    def _build_agent(self) -> Agent:
        """Create the pydantic-ai Agent with no tools."""
        _, model, _ = self.context.resolve_subagent_model(
            "handoff",
            model_name=self._model_name_override,
            api_key=self._api_key_override,
        )
        return Agent(
            model,
            instructions=HANDOFF_SYSTEM_PROMPT,
        )

    def _format_history_for_prompt(
        self,
        message_history: list[ModelMessage],
    ) -> str:
        """Format message history into a readable string for the LLM.

        Includes full transparency: user messages, assistant responses,
        tool calls with arguments, and tool results (including SQL and output).

        Args:
            message_history: The pydantic-ai message history.

        Returns:
            Formatted conversation string.
        """
        if not message_history:
            return "(No conversation history)"

        lines: list[str] = []

        for msg in message_history:
            if msg.kind == "request":
                for part in msg.parts:
                    if part.part_kind == "user-prompt":
                        content = getattr(part, "content", "")
                        lines.append(f"[User]: {content}")
                    elif part.part_kind == "tool-return":
                        tool_name = getattr(part, "tool_name", "tool")
                        content = str(getattr(part, "content", ""))
                        if len(content) > 1000:
                            content = content[:1000] + "...(truncated)"
                        lines.append(f"[Tool result - {tool_name}]: {content}")
            elif msg.kind == "response":
                for part in msg.parts:
                    if part.part_kind == "text":
                        content = getattr(part, "content", "")
                        lines.append(f"[Assistant]: {content}")
                    elif part.part_kind == "tool-call":
                        tool_name = getattr(part, "tool_name", "unknown")
                        args = getattr(part, "args", {})
                        # args can be dict or JSON string (ArgsJson type)
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        if isinstance(args, dict) and args:
                            args_str = ", ".join(
                                f"{k}={json.dumps(v, ensure_ascii=False)}"
                                for k, v in args.items()
                            )
                            lines.append(
                                f"[Assistant tool call - {tool_name}]: {args_str}"
                            )
                        else:
                            lines.append(f"[Assistant tool call - {tool_name}]:")

        return "\n\n".join(lines) if lines else "(No readable messages)"

    async def generate_draft(
        self,
        message_history: list[ModelMessage],
        goal: str,
        *,
        usage: RunUsage | None = None,
    ) -> str:
        """Generate a handoff prompt draft.

        Args:
            message_history: The current conversation history.
            goal: The user's goal for the new thread.

        Returns:
            Generated handoff prompt text.
        """
        formatted_history = self._format_history_for_prompt(message_history)

        prompt = f"""<source_conversation>
{formatted_history}
</source_conversation>

<handoff_goal>
{goal}
</handoff_goal>

{HANDOFF_INPUT_INSTRUCTIONS}
"""

        result = await self._build_agent().run(
            prompt,
            usage=usage,
            usage_limits=current_usage_limits() or UsageLimits(),
        )
        return str(result.output).strip()

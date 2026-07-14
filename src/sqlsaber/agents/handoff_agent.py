"""Handoff Agent for generating context-aware handoff prompts.

This is a dedicated agent with no tools, used solely for summarizing
conversations and generating handoff prompts for fresh threads.
"""

import json

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from sqlsaber.agents.model_factory import build_model
from sqlsaber.config.settings import Config
from sqlsaber.prompts.handoff import HANDOFF_SYSTEM_PROMPT


class HandoffAgent:
    """Dedicated agent for generating handoff prompts.

    This agent has no tools registered and uses a specialized system prompt
    focused on summarizing conversations and extracting key context.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the handoff agent.

        Args:
            model_name: Optional model override. Defaults to configured model.
            api_key: Optional API key override.
        """
        self.config = Config()
        self._model_name_override = model_name
        self._api_key_override = api_key
        self.agent = self._build_agent()

    def _build_agent(self) -> Agent:
        """Create the pydantic-ai Agent with no tools."""
        model_name = (
            self._model_name_override
            or self.config.model.get_subagent_model("handoff")
            or self.config.model.name
        )
        if not (self._model_name_override and self._api_key_override):
            self.config.auth.validate(model_name)

        api_key = self._api_key_override or self.config.auth.get_api_key(model_name)
        return Agent(
            build_model(model_name, api_key),
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
                        lines.append(f"User: {content}")
                    elif part.part_kind == "tool-return":
                        tool_name = getattr(part, "tool_name", "tool")
                        content = str(getattr(part, "content", ""))
                        if len(content) > 1000:
                            content = content[:1000] + "...(truncated)"
                        lines.append(f"[Tool Result - {tool_name}]:\n{content}")
            elif msg.kind == "response":
                for part in msg.parts:
                    if part.part_kind == "text":
                        content = getattr(part, "content", "")
                        lines.append(f"Assistant: {content}")
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
                            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                            lines.append(f"[Tool Call - {tool_name}]: {args_str}")
                        else:
                            lines.append(f"[Tool Call - {tool_name}]")

        return "\n\n".join(lines) if lines else "(No readable messages)"

    async def generate_draft(
        self,
        message_history: list[ModelMessage],
        goal: str,
    ) -> str:
        """Generate a handoff prompt draft.

        Args:
            message_history: The current conversation history.
            goal: The user's goal for the new thread.

        Returns:
            Generated handoff prompt text.
        """
        formatted_history = self._format_history_for_prompt(message_history)

        prompt = f"""
## Conversation History
{formatted_history}

## User's Goal for New Thread
{goal}
"""

        result = await self.agent.run(prompt)
        return str(result.output).strip()

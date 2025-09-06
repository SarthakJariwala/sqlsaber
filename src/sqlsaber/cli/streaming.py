"""Streaming query handling for the CLI (pydantic-ai based)."""

import asyncio
import json
from typing import AsyncIterable

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from rich.console import Console

from sqlsaber.cli.display import DisplayManager


class StreamingQueryHandler:
    """Handles streaming query execution and display using pydantic-ai events."""

    def __init__(self, console: Console):
        self.console = console
        self.display = DisplayManager(console)

        self.status = self.console.status(
            "[yellow]Crunching data...[/yellow]", spinner="bouncingBall"
        )

    async def _event_stream_handler(
        self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        async for event in event_stream:
            if isinstance(event, PartStartEvent):
                if isinstance(event.part, (TextPart, ThinkingPart)):
                    self.status.stop()
                    self.display.show_text_stream(event.part.content)

            elif isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, (TextPartDelta, ThinkingPartDelta)):
                    delta = event.delta.content_delta or ""
                    if delta:
                        self.status.stop()
                        self.display.show_text_stream(delta)

            elif isinstance(event, FunctionToolCallEvent):
                # Show tool execution start
                self.status.stop()
                args = event.part.args_as_dict()
                self.display.show_newline()
                self.display.show_tool_executing(event.part.tool_name, args)

            elif isinstance(event, FunctionToolResultEvent):
                self.status.stop()
                # Route tool result to appropriate display
                tool_name = event.result.tool_name
                content = event.result.content
                if tool_name == "list_tables":
                    self.display.show_table_list(content)
                elif tool_name == "introspect_schema":
                    self.display.show_schema_info(content)
                elif tool_name == "execute_sql":
                    try:
                        data = json.loads(content)
                        if data.get("success") and data.get("results"):
                            self.display.show_query_results(data["results"])  # type: ignore[arg-type]
                    except json.JSONDecodeError:
                        # If not JSON, ignore here
                        pass

    async def execute_streaming_query(
        self,
        user_query: str,
        agent: Agent,
        cancellation_token: asyncio.Event | None = None,
        message_history: list | None = None,
    ):
        self.status.start()
        try:
            # If Anthropic OAuth, inject SQLsaber instructions before the first user prompt
            prepared_prompt: str | list[str] = user_query
            is_oauth = bool(getattr(agent, "_sqlsaber_is_oauth", False))
            no_history = not message_history
            if is_oauth and no_history:
                ib = getattr(agent, "_sqlsaber_instruction_builder", None)
                mm = getattr(agent, "_sqlsaber_memory_manager", None)
                db_type = getattr(agent, "_sqlsaber_db_type", "database")
                db_name = getattr(agent, "_sqlsaber_database_name", None)
                instructions = (
                    ib.build_instructions(db_type=db_type) if ib is not None else ""
                )
                mem = (
                    mm.format_memories_for_prompt(db_name)
                    if (mm is not None and db_name)
                    else ""
                )
                parts = [p for p in (instructions, mem) if p and str(p).strip()]
                if parts:
                    injected = "\n\n".join(parts)
                    prepared_prompt = [injected, user_query]

            # Run the agent with our event stream handler
            run = await agent.run(
                prepared_prompt,
                message_history=message_history,
                event_stream_handler=self._event_stream_handler,
            )
            # After the run completes, show the assistant's final text as markdown if available
            try:
                output = run.output
                if isinstance(output, str) and output.strip():
                    self.display.show_newline()
                    self.display.show_markdown_response(
                        [{"type": "text", "text": output}]
                    )
            except Exception as e:
                self.display.show_error(str(e))
                self.display.show_newline()
            return run
        except asyncio.CancelledError:
            self.display.show_newline()
            self.console.print("[yellow]Query interrupted[/yellow]")
            return None
        finally:
            try:
                self.status.stop()
            except Exception:
                pass

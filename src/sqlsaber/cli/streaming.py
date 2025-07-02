"""Streaming query handling for the CLI."""

import asyncio

from rich.console import Console

from sqlsaber.agents.base import BaseSQLAgent
from sqlsaber.cli.display import DisplayManager


class StreamingQueryHandler:
    """Handles streaming query execution and display."""

    def __init__(self, console: Console):
        self.console = console
        self.display = DisplayManager(console)

    async def execute_streaming_query(
        self,
        user_query: str,
        agent: BaseSQLAgent,
        cancellation_token: asyncio.Event | None = None,
    ):
        """Execute a query with streaming display."""

        status = self.console.status(
            "[yellow]Crunching data...[/yellow]", spinner="bouncingBall"
        )
        status.start()

        try:
            async for event in agent.query_stream(
                user_query, cancellation_token=cancellation_token
            ):
                if cancellation_token is not None and cancellation_token.is_set():
                    break

                if event.type == "tool_use":
                    self._stop_status(status)

                    if event.data["status"] == "executing":
                        self.display.show_newline()
                        self.display.show_tool_executing(
                            event.data["name"], event.data["input"]
                        )

                elif event.type == "text":
                    # Always stop status when text streaming starts
                    self._stop_status(status)
                    self.display.show_text_stream(event.data)

                elif event.type == "query_result":
                    if event.data["results"]:
                        self.display.show_query_results(event.data["results"])

                elif event.type == "tool_result":
                    # Handle tool results - particularly list_tables and introspect_schema
                    if event.data.get("tool_name") == "list_tables":
                        self.display.show_table_list(event.data["result"])
                    elif event.data.get("tool_name") == "introspect_schema":
                        self.display.show_schema_info(event.data["result"])

                elif event.type == "plot_result":
                    # Handle plot results
                    self.display.show_plot(event.data)

                elif event.type == "processing":
                    self.display.show_newline()  # Add newline after explanation text
                    self._stop_status(status)
                    status = self.display.show_processing(event.data)
                    status.start()

                elif event.type == "error":
                    self._stop_status(status)
                    self.display.show_error(event.data)

        except asyncio.CancelledError:
            # Handle cancellation gracefully
            self._stop_status(status)
            self.display.show_newline()
            self.console.print("[yellow]Query interrupted[/yellow]")
            return
        finally:
            # Make sure status is stopped
            self._stop_status(status)

            # Display the last assistant response as markdown
            if hasattr(agent, "conversation_history") and agent.conversation_history:
                last_message = agent.conversation_history[-1]
                if last_message.get("role") == "assistant" and last_message.get(
                    "content"
                ):
                    self.display.show_markdown_response(last_message["content"])

    def _stop_status(self, status):
        """Safely stop a status spinner."""
        try:
            status.stop()
        except Exception:
            pass  # Status might already be stopped

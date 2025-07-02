"""Event models for streaming and responses."""

from typing import Any


class StreamEvent:
    """Event emitted during streaming processing."""

    def __init__(self, event_type: str, data: Any = None):
        # 'tool_use', 'text', 'query_result', 'plot_result', 'error', 'processing'
        self.type = event_type
        self.data = data


class SQLResponse:
    """Response from the SQL agent."""

    def __init__(
        self,
        query: str | None = None,
        explanation: str = "",
        results: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ):
        self.query = query
        self.explanation = explanation
        self.results = results
        self.error = error

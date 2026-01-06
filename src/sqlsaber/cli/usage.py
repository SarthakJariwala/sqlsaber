"""Session usage tracking for the CLI.

IMPORTANT: Token accounting in multi-turn conversations:
- input_tokens on each request includes the FULL context (message history)
- We track the LATEST context size, not cumulative input tokens
- output_tokens are cumulative since each response is new content
"""

from dataclasses import dataclass

from pydantic_ai.usage import RunUsage


@dataclass
class SessionUsage:
    """Tracks usage across a session with correct token accounting.

    Input tokens are NOT accumulated because each request's input_tokens
    includes the full message history. We track the current context size
    (latest request's input tokens) instead.

    Output tokens ARE accumulated since each response is genuinely new.
    """

    requests: int = 0
    tool_calls: int = 0

    # Current context window size (latest request's input tokens)
    current_context_tokens: int = 0

    # Cumulative output tokens (new content each turn)
    total_output_tokens: int = 0

    # Cache tokens (cumulative for reporting)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add_run(self, usage: RunUsage, final_context_tokens: int) -> None:
        """Add usage from a single agent run.

        Args:
            usage: The RunUsage from the agent run (cumulative across all requests).
            final_context_tokens: The input tokens for the FINAL request only
                (from result.response.usage.input_tokens), representing the
                actual context window size.
        """
        self.requests += usage.requests
        self.tool_calls += usage.tool_calls

        # Use the final request's input tokens as context size (not cumulative)
        self.current_context_tokens = final_context_tokens

        # Accumulate output tokens (these are genuinely new each turn)
        self.total_output_tokens += usage.output_tokens

        # Accumulate cache tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens


def format_tokens(count: int) -> str:
    """Format token count with K/M suffixes for readability."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)

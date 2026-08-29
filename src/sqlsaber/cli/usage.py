"""Session usage tracking for the CLI."""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.usage import RequestUsage, RunUsage


@dataclass
class SessionUsage:
    """Tracks cumulative model usage and the current context size."""

    requests: int = 0
    tool_calls: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    current_context_tokens: int = 0

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    total_cost_usd: float | None = 0.0

    def add_run(
        self,
        usage: RunUsage,
        final_context_tokens: int,
        *,
        model_name: str | None = None,
        request_usages: Sequence[RequestUsage] | None = None,
    ) -> None:
        """Add usage from a single agent run.

        Args:
            usage: The RunUsage from the agent run, summed across model requests.
            final_context_tokens: Input tokens for the final request only, representing
                the current context window size.
            model_name: Optional provider-prefixed model name for cost calculation.
            request_usages: Optional per-model-request usage entries for accurate
                pricing of multi-request agent runs.
        """
        self.requests += usage.requests
        self.tool_calls += usage.tool_calls

        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.current_context_tokens = final_context_tokens

        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens

        run_cost = calculate_usages_cost_usd(
            request_usages if request_usages else [usage],
            model_name,
        )
        if run_cost is None:
            self.total_cost_usd = None
            return
        if self.total_cost_usd is not None:
            self.total_cost_usd += run_cost


def calculate_usages_cost_usd(
    usages: Sequence[RunUsage | RequestUsage], model_name: str | None
) -> float | None:
    """Calculate a cache-aware USD cost estimate from one or more request usages."""
    if not model_name:
        return None

    total = 0.0
    for usage in usages:
        usage_cost = calculate_run_cost_usd(usage, model_name)
        if usage_cost is None:
            return None
        total += usage_cost
    return total


def calculate_run_cost_usd(
    usage: RunUsage | RequestUsage, model_name: str | None
) -> float | None:
    """Calculate a cache-aware USD cost estimate for one model request."""
    if not model_name:
        return None
    if isinstance(usage, RunUsage) and usage.requests > 1:
        return None

    try:
        from genai_prices import calc_price
        from genai_prices.types import Usage as GenAIUsage

        provider_id: str | None = None
        model_ref = model_name
        if ":" in model_name:
            provider_id, model_ref = model_name.split(":", 1)

        result = calc_price(
            usage=GenAIUsage(
                input_tokens=usage.input_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                output_tokens=usage.output_tokens,
                input_audio_tokens=usage.input_audio_tokens,
                cache_audio_read_tokens=usage.cache_audio_read_tokens,
                output_audio_tokens=usage.output_audio_tokens,
            ),
            model_ref=model_ref,
            provider_id=provider_id,
        )
        return float(result.total_price)
    except Exception:
        return None


def format_cost_usd(cost_usd: float | None) -> str:
    """Format a USD cost estimate for compact terminal display."""
    if cost_usd is None:
        return "n/a"
    if 0 < cost_usd < 0.0001:
        return "<$0.0001"
    return f"${cost_usd:.4f}"


def format_tokens(count: int) -> str:
    """Format token count with K/M suffixes for readability."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def session_summary_blocks(session_usage: SessionUsage):
    """Blocks for the TTY session summary. Empty when there were no requests.

    Args:
        session_usage: Accumulated usage for the session.

    Returns:
        Summary blocks, or an empty tuple.
    """
    from sqlsaber.render import blocks as b

    if session_usage.requests == 0:
        return ()
    return (
        b.md("**Session Summary**", role="muted"),
        b.md(
            f"Usage: {format_tokens(session_usage.total_input_tokens)} in / "
            f"{format_tokens(session_usage.total_output_tokens)} out │ "
            f"Cost: {format_cost_usd(session_usage.total_cost_usd)}",
            role="muted",
        ),
        b.md(
            f"Current context: {session_usage.current_context_tokens:,} tokens",
            role="muted",
        ),
        b.md(
            f"Requests: {session_usage.requests} │ "
            f"Tool calls: {session_usage.tool_calls}",
            role="muted",
        ),
    )

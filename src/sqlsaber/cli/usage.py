"""Session usage tracking for the CLI.

Token accounting in multi-turn conversations has two useful views:
- cumulative input/output usage: total model traffic for cost/session accounting
- current context tokens: latest request size, useful for context-window awareness
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.usage import RequestUsage, RunUsage


@dataclass
class SessionUsage:
    """Tracks cumulative model usage and the current context size."""

    requests: int = 0
    tool_calls: int = 0

    # Cumulative input/output tokens across model requests.
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Current context window size (latest request's input tokens).
    current_context_tokens: int = 0

    # Cache tokens (cumulative for reporting and cache-aware pricing).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Estimated cumulative USD cost. None means pricing was unavailable.
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


def request_usages_from_messages(messages: Sequence[ModelMessage]) -> list[RequestUsage]:
    """Extract per-request usage entries from model response messages."""
    return [message.usage for message in messages if isinstance(message, ModelResponse)]


def request_usages_from_run_result(run_result: Any) -> list[RequestUsage]:
    """Extract per-request usage entries from a Pydantic AI run result."""
    new_messages = getattr(run_result, "new_messages", None)
    if not callable(new_messages):
        return []
    return request_usages_from_messages(new_messages())


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

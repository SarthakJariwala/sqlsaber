"""Session usage tracking for the CLI."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage

from sqlsaber.config.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import AgentStreamEvent, ModelMessage

    from sqlsaber import SQLSaberResult

log = get_logger(__name__)

type EventStreamHandler = Callable[
    [RunContext[Any], AsyncIterable[AgentStreamEvent]], Awaitable[None]
]


class StreamingQuery(Protocol):
    """The slice of ``SQLSaber.query`` that the stream presenter calls."""

    async def __call__(
        self,
        prompt: str,
        /,
        *,
        event_stream_handler: EventStreamHandler | None = ...,
    ) -> SQLSaberResult: ...


@dataclass(frozen=True, slots=True)
class SessionUsage:
    """Cumulative model usage and current context size for one session.

    Frozen so that ``UsageMeter`` is the only thing able to produce a new one.
    """

    requests: int = 0
    tool_calls: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    current_context_tokens: int = 0

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    total_cost_usd: float | None = 0.0
    """``None`` means not priceable and is absorbing until :meth:`UsageMeter.reset`."""


@dataclass(frozen=True, slots=True)
class _PricedResponse:
    """One complete model message: what it consumed and what it cost."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None

    @classmethod
    def from_request(cls, usage: RequestUsage, model_id: str | None) -> _PricedResponse:
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=_price(usage, model_id),
        )


@dataclass(frozen=True, slots=True)
class _TurnCursor:
    """High-water mark for the agent run currently in flight.

    ``responses`` is how many of this run's model messages have been folded and
    ``tool_calls`` is the run's tool-call counter as of the last fold.
    """

    responses: int = 0
    tool_calls: int = 0


def _add_cost(total: float | None, addition: float | None) -> float | None:
    if total is None or addition is None:
        return None
    return total + addition


def _fold(
    base: SessionUsage, responses: Sequence[_PricedResponse], *, tool_calls: int
) -> SessionUsage:
    folded = replace(
        base,
        requests=base.requests + len(responses),
        tool_calls=base.tool_calls + tool_calls,
    )
    for response in responses:
        folded = replace(
            folded,
            total_input_tokens=folded.total_input_tokens + response.input_tokens,
            total_output_tokens=folded.total_output_tokens + response.output_tokens,
            cache_read_tokens=folded.cache_read_tokens + response.cache_read_tokens,
            cache_write_tokens=folded.cache_write_tokens + response.cache_write_tokens,
            current_context_tokens=response.input_tokens,
            total_cost_usd=_add_cost(folded.total_cost_usd, response.cost_usd),
        )
    return folded


def _price(usage: RequestUsage, model_id: str | None) -> float | None:
    """Cache-aware USD estimate for one model request, or ``None`` if unknown."""
    if not model_id:
        return None

    try:
        from genai_prices import calc_price
        from genai_prices.types import Usage as GenAIUsage

        provider_id: str | None = None
        model_ref = model_id
        if ":" in model_id:
            provider_id, model_ref = model_id.split(":", 1)

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


def _run_request_usages(
    messages: Sequence[ModelMessage], run_id: str
) -> list[RequestUsage]:
    """Usage of every complete model message this run produced, in order.

    ``ctx.messages`` is the whole conversation; pydantic-ai stamps ``run_id`` on
    each response as it appends it, which isolates this run without index math.
    """
    return [
        message.usage
        for message in messages
        if isinstance(message, ModelResponse)
        and message.run_id == run_id
        and message.state == "complete"
    ]


class UsageMeter:
    """Keeps one ``SessionUsage`` current across agent runs.

    Wrap the query callable with :meth:`metered` and the meter folds usage into
    the session after every graph node and once more from the run result. Each
    observation is a cumulative snapshot of the run, so overlapping observations
    cannot double-count.
    """

    def __init__(
        self,
        *,
        model_id: Callable[[], str | None],
        on_change: Callable[[], None] | None = None,
    ) -> None:
        """
        Args:
            model_id: Read on each fold, so swapping the underlying ``SQLSaber``
                prices later responses with the new model.
            on_change: Called after any fold that changed the session.
        """
        self._model_id = model_id
        self._on_change = on_change
        self._session = SessionUsage()
        self._turn = _TurnCursor()
        self._turn_base = SessionUsage()
        self._ctx: RunContext[Any] | None = None

    @property
    def session(self) -> SessionUsage:
        """Current totals; immutable and safe to hand to renderers and commands."""
        return self._session

    def metered(self, query: StreamingQuery) -> StreamingQuery:
        """Wrap a query callable so its usage lands in this meter.

        Args:
            query: The SDK query to observe, typically ``SQLSaber.query``.

        Returns:
            A callable with the same signature. Each call is one turn: it opens a
            fresh cursor, wraps the caller's event-stream handler, and folds the
            run result on the way out. If the run raises, whatever pydantic-ai
            had already applied to the last seen ``RunContext`` is folded first.
        """

        async def metered_query(
            prompt: str,
            /,
            *,
            event_stream_handler: EventStreamHandler | None = None,
            **kwargs: Any,
        ) -> SQLSaberResult:
            self._turn = _TurnCursor()
            self._turn_base = self._session
            self._ctx = None
            result: SQLSaberResult | None = None
            try:
                result = await query(
                    prompt,
                    event_stream_handler=self._observing(event_stream_handler),
                    **kwargs,
                )
            finally:
                if result is None and self._ctx is not None:
                    self._absorb_context(self._ctx)
            run_usage = result.usage
            self._absorb(
                result.request_usages,
                tool_calls=run_usage.tool_calls if run_usage is not None else 0,
            )
            return result

        return metered_query

    def reset(self) -> None:
        """Start a new accounting epoch, notifying if the session was non-empty."""
        changed = self._session != SessionUsage()
        self._session = SessionUsage()
        self._turn = _TurnCursor()
        self._turn_base = SessionUsage()
        self._ctx = None
        if changed:
            self._notify()

    def _observing(self, inner: EventStreamHandler | None) -> EventStreamHandler:
        async def observing_handler(
            ctx: RunContext[Any], event_stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            self._ctx = ctx
            if inner is not None:
                await inner(ctx, event_stream)
            else:
                async for _ in event_stream:
                    pass
            self._absorb_context(ctx)

        return observing_handler

    def _absorb_context(self, ctx: RunContext[Any]) -> None:
        run_id = ctx.run_id
        if run_id is None:
            return
        self._absorb(
            _run_request_usages(ctx.messages, run_id), tool_calls=ctx.usage.tool_calls
        )

    def _absorb(
        self, request_usages: Iterable[RequestUsage], *, tool_calls: int
    ) -> None:
        usages = list(request_usages)
        cursor = self._turn
        if len(usages) < cursor.responses:
            log.warning(
                "usage.meter.cursor_reset",
                observed=len(usages),
                folded=cursor.responses,
            )
            self._session = self._turn_base
            cursor = _TurnCursor()
        fresh = usages[cursor.responses :]
        new_tool_calls = max(0, tool_calls - cursor.tool_calls)
        self._turn = _TurnCursor(
            responses=len(usages), tool_calls=max(cursor.tool_calls, tool_calls)
        )
        if not fresh and not new_tool_calls:
            return
        model_id = self._model_id()
        self._session = _fold(
            self._session,
            [_PricedResponse.from_request(usage, model_id) for usage in fresh],
            tool_calls=new_tool_calls,
        )
        self._notify()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()


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

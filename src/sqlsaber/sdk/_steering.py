"""Per-run enqueue port used by SQLSaber.steer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.run import AgentRunResult, PendingMessage

from .errors import SteerIdleError


@dataclass(slots=True)
class _Open:
    queue: list[PendingMessage]
    texts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _Staged(_Open):
    """Before wrap_run. `queue` is a private list this port owns; pydantic-ai has no queue yet."""


@dataclass(slots=True)
class _Live(_Open):
    """After wrap_run. `queue` IS `ctx.pending_messages`, the very list object the
    PendingMessageDrainCapability drains with `queue[:] = remaining` between graph nodes.
    """


@dataclass(frozen=True, slots=True)
class _Ended:
    pass


type _State = _Staged | _Live | _Ended


class Steering(AbstractCapability[Any]):
    def __init__(self, *, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop
        self._state: _State = _Staged(queue=[]) if loop is not None else _Ended()

    @classmethod
    def ended(cls) -> Steering:
        return cls(loop=None)

    def steer(self, text: str) -> str:
        self._assert_on_loop()
        match self._state:
            case _Open(queue, texts):
                pending = PendingMessage.from_content(text)
                assert pending is not None, (
                    "SQLSaber.steer rejects blank text at the boundary"
                )
                queue.append(pending)
                texts[pending.enqueue_id] = text
                return pending.enqueue_id
            case _Ended():
                raise SteerIdleError()

    @property
    def pending(self) -> tuple[str, ...]:
        match self._state:
            case _Open(queue, texts):
                return tuple(
                    texts[pm.enqueue_id] for pm in queue if pm.enqueue_id in texts
                )
            case _Ended():
                return ()

    def clear(self) -> list[str]:
        self._assert_on_loop()
        match self._state:
            case _Open(queue, texts):
                mine = [pm for pm in queue if pm.enqueue_id in texts]
                queue[:] = [pm for pm in queue if pm.enqueue_id not in texts]
                return [texts.pop(pm.enqueue_id) for pm in mine]
            case _Ended():
                return []

    def end(self) -> None:
        self._state = _Ended()

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        match self._state:
            case _Staged(queue, texts):
                live = ctx.pending_messages
                assert live is not None, "wrap_run ctx always carries the run queue"
                live.extend(queue)
                self._state = _Live(queue=live, texts=texts)
            case _Live() | _Ended():
                raise RuntimeError("Steering is single-use: one instance per query()")
        try:
            return await handler()
        finally:
            self.end()

    def _assert_on_loop(self) -> None:
        if self._loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not self._loop:
            raise RuntimeError(
                "SQLSaber.steer / clear_steers must run on the event loop that runs query(). "
                "Use loop.call_soon_threadsafe from other threads."
            )

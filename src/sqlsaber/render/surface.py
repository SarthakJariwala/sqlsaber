"""Surface protocol, prompt types, and PromptUnavailable."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .blocks import Block, Role


@dataclass(frozen=True, slots=True)
class Choice[T]:
    """One selectable option. ``value`` is the typed result, not a string id."""

    label: str
    value: T
    description: str | None = None


@dataclass(frozen=True)
class Ask[T]:
    """Base prompt. ``unavailable_hint`` is the command to suggest on a non-TTY."""

    message: str
    unavailable_hint: str | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class AskText(Ask[str]):
    default: str = ""
    validate: Callable[[str], str | None] | None = None


@dataclass(frozen=True)
class AskSecret(Ask[str]):
    """Hosted by ``getpass``. Never mounted inside a live TUI."""


@dataclass(frozen=True)
class AskPath(Ask[str]):
    default: str = ""
    only_directories: bool = False


@dataclass(frozen=True)
class AskConfirm(Ask[bool]):
    default: bool = False
    assume_yes: bool = False


@dataclass(frozen=True)
class AskChoice[T](Ask[T]):
    choices: Sequence[Choice[T]] = ()
    default: T | None = None
    searchable: bool = True


class PromptUnavailable(Exception):
    """Raised when a prompt is asked of a surface that cannot interact."""

    def __init__(self, prompt: Ask[Any]) -> None:
        self.prompt = prompt
        hint = prompt.unavailable_hint
        if hint:
            message = (
                f"confirmation requires an interactive terminal.\n  Re-run with: {hint}"
            )
        else:
            message = "this command requires an interactive terminal."
        super().__init__(message)


@runtime_checkable
class TextStream(Protocol):
    """A block of markdown that grows in place, then settles into scrollback."""

    def append(self, chunk: str) -> None: ...

    def set(self, text: str) -> None: ...

    def close(self) -> None: ...

    def discard(self) -> None: ...


@runtime_checkable
class Surface(Protocol):
    """Print, stream, status, and prompt on one host."""

    def emit(self, *blocks: Block) -> None:
        """Append finished blocks. Spacing between blocks belongs to the surface."""
        ...

    def stream(
        self,
        *,
        role: Role | None = None,
        replace: TextStream | None = None,
        before: TextStream | None = None,
    ) -> TextStream:
        """Open a growing markdown region.

        ``replace`` swaps an existing stream in place. ``before`` inserts
        ahead of another open stream so out-of-order SQL previews keep order.
        """
        ...

    def status(
        self, text: str | None, *, on_cancel: Callable[[], None] | None = None
    ) -> None:
        """Transient progress line. ``None`` clears. No-op on a pipe."""
        ...

    async def ask[T](self, prompt: Ask[T]) -> T | None:
        """None means the user cancelled. Raises ``PromptUnavailable`` on a pipe."""
        ...

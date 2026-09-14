"""Headless JSONL RPC mode for embedding SQLSaber."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import RpcSession, serve

__all__ = [
    "RpcSession",
    "serve",
]


def __getattr__(name: str):
    """Lazy import so ``saber --help`` never loads pydantic-ai via RPC."""
    if name in {"RpcSession", "serve"}:
        from . import session

        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

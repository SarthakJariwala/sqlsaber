"""SQLSaber CLI - Agentic SQL assistant like Claude Code but for SQL."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import SQLSaber

__all__ = ["SQLSaber"]


def __getattr__(name: str):
    """Lazy import for SQLSaber to avoid heavy startup imports."""
    if name == "SQLSaber":
        from .api import SQLSaber

        return SQLSaber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

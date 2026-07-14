"""SQLSaber CLI - Agentic SQL assistant like Claude Code but for SQL."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import SQLSaber
    from .capabilities import Knowledge, SqlTools
    from .options import SQLSaberOptions
    from .overrides import ModelOverides

__all__ = ["SQLSaber", "SQLSaberOptions", "ModelOverides", "SqlTools", "Knowledge"]


def __getattr__(name: str):
    """Lazy import for SQLSaber to avoid heavy startup imports."""
    if name == "SQLSaber":
        from .api import SQLSaber

        return SQLSaber
    if name == "SQLSaberOptions":
        from .options import SQLSaberOptions

        return SQLSaberOptions
    if name == "ModelOverides":
        from .overrides import ModelOverides

        return ModelOverides
    if name == "SqlTools":
        from .capabilities import SqlTools

        return SqlTools
    if name == "Knowledge":
        from .capabilities import Knowledge

        return Knowledge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

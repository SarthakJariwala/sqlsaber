"""Configuration module for SQLSaber."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import Config

__all__ = [
    "Config",
]


def __getattr__(name: str):
    """Lazy import for Config to avoid heavy startup imports."""
    if name == "Config":
        from .settings import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

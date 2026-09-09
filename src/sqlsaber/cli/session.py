"""Shared SQLSaberOptions construction for CLI sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlsaber import SQLSaberOptions


def cli_sqlsaber_options(**kwargs: Any) -> SQLSaberOptions:
    """Build CLI session options with installed plugin factories."""
    from sqlsaber import SQLSaberOptions
    from sqlsaber.capabilities.plugins import load_capability_factories

    kwargs.setdefault("capabilities", load_capability_factories())
    return SQLSaberOptions(**kwargs)

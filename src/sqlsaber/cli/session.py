"""Shared SQLSaberOptions construction for CLI sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlsaber import SQLSaberOptions


def cli_sqlsaber_options(**kwargs: Any) -> SQLSaberOptions:
    """Build session options, including installed plugin factories.

    CLI sessions opt into ``sqlsaber.capabilities`` entry points explicitly.
    The SDK default is an empty capabilities list.
    """
    from sqlsaber import SQLSaberOptions
    from sqlsaber.capabilities.plugins import load_capability_factories

    kwargs.setdefault("capabilities", load_capability_factories())
    return SQLSaberOptions(**kwargs)

"""Lazy CLI emit helpers. No saber-tui at import."""

from __future__ import annotations

import asyncio
from typing import NoReturn

from sqlsaber.render import PromptUnavailable, blocks as b, cli_err, cli_out


def out(*blocks: b.Block) -> None:
    """Emit finished blocks on stdout."""
    cli_out().emit(*blocks)


def err(*blocks: b.Block) -> None:
    """Emit finished blocks on stderr."""
    cli_err().emit(*blocks)


def fail(message: str, code: int = 1) -> NoReturn:
    """Print ``**Error:**`` on stderr and exit.

    Args:
        message: Error body after the label.
        code: Process exit code.
    """
    err(b.error(message))
    raise SystemExit(code)


def fail_usage(message: str) -> NoReturn:
    """Print ``**Error:**`` on stderr and exit 2."""
    fail(message, code=2)


async def confirm(*, yes: bool, prompt: str, hint: str) -> bool:
    """Ask ``AskConfirm``. ``yes`` is ``--yes``.

    Args:
        yes: Skip the prompt.
        prompt: Confirm message.
        hint: Command suggested when stdin is not a TTY.

    Returns:
        True if confirmed.

    Raises:
        SystemExit: Exit 2 when the prompt cannot be shown.
    """
    from sqlsaber.render import AskConfirm

    try:
        result = await cli_out().ask(
            AskConfirm(prompt, assume_yes=yes, unavailable_hint=hint)
        )
    except PromptUnavailable as exc:
        err(b.error(str(exc)))
        raise SystemExit(2) from exc
    return bool(result)


def confirm_sync(*, yes: bool, prompt: str, hint: str) -> bool:
    """Sync wrapper around ``confirm`` for cyclopts commands."""
    return asyncio.run(confirm(yes=yes, prompt=prompt, hint=hint))

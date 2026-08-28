"""Shared safety checks for destructive CLI commands."""

import sys

import questionary
from rich.console import Console


def confirm_action(
    *,
    yes: bool,
    prompt: str,
    non_interactive_command: str,
    error_console: Console,
) -> bool:
    """Confirm an action or require an explicit flag when stdin is not a TTY."""
    if yes:
        return True
    if not sys.stdin.isatty():
        error_console.print(
            "[error]Error: confirmation requires an interactive terminal.[/error]\n"
            f"  Re-run with: {non_interactive_command}"
        )
        raise SystemExit(2)
    return bool(questionary.confirm(prompt, default=False).ask())

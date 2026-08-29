"""Shared safety checks for destructive CLI commands."""

from sqlsaber.cli.output import confirm_sync


def confirm_action(
    *,
    yes: bool,
    prompt: str,
    non_interactive_command: str,
) -> bool:
    """Confirm an action or require ``--yes`` when stdin is not a TTY.

    Args:
        yes: ``--yes`` short-circuit.
        prompt: Confirm message.
        non_interactive_command: Hint shown on a pipe.

    Returns:
        True if the user confirmed.
    """
    return confirm_sync(yes=yes, prompt=prompt, hint=non_interactive_command)

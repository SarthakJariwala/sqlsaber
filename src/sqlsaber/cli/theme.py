"""Theme management CLI commands."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from platformdirs import user_config_dir
from pygments.styles import get_all_styles

from sqlsaber.application.prompts import AsyncPrompter, Choice
from sqlsaber.cli.output import err, fail, fail_usage, out
from sqlsaber.cli.safety import confirm_action
from sqlsaber.config.logging import get_logger
from sqlsaber.render import blocks as b
from sqlsaber.theme.manager import DEFAULT_THEME_NAME

logger = get_logger(__name__)

theme_app = cyclopts.App(
    name="theme",
    help="Manage theme settings",
    help_epilogue=("Examples:\n\nsaber theme set dracula\n\nsaber theme reset --yes"),
)


class ThemeManager:
    """Manages theme configuration persistence."""

    def __init__(self):
        self.config_dir = Path(user_config_dir("sqlsaber"))
        self.config_file = self.config_dir / "theme.json"

    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load theme configuration from file."""
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_config(self, config: dict) -> None:
        """Save theme configuration to file."""
        self._ensure_config_dir()

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    def get_current_theme(self) -> str:
        """Get the currently configured theme."""
        config = self._load_config()
        env_theme = os.getenv("SQLSABER_THEME")
        if env_theme:
            return env_theme
        return config.get("theme", {}).get("pygments_style") or DEFAULT_THEME_NAME

    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme."""
        try:
            config = self._load_config()
            if "theme" not in config:
                config["theme"] = {}
            config["theme"]["name"] = theme_name
            config["theme"]["pygments_style"] = theme_name
            self._save_config(config)
            return True
        except Exception as e:
            logger.error("theme.set.error", theme=theme_name, error=str(e))
            err(b.error(f"Error setting theme: {e}"))
            return False

    def reset_theme(self) -> bool:
        """Reset to default theme."""
        try:
            if self.config_file.exists():
                self.config_file.unlink()
            return True
        except Exception as e:
            logger.error("theme.reset.error", error=str(e))
            err(b.error(f"Error resetting theme: {e}"))
            return False

    def get_available_themes(self) -> list[str]:
        """Get list of available Pygments themes."""
        return sorted(get_all_styles())


theme_manager = ThemeManager()


@theme_app.command(
    help_epilogue=("Examples:\n\nsaber theme set\n\nsaber theme set dracula")
)
def set(
    theme_name: Annotated[
        str | None,
        cyclopts.Parameter(help="Pygments theme name (omit to select interactively)"),
    ] = None,
):
    """Set the theme to use for syntax highlighting.

    Examples:
        saber theme set
        saber theme set dracula
    """
    logger.info("theme.set.start")

    themes = theme_manager.get_available_themes()
    if theme_name is not None:
        theme_name = theme_name.strip().lower()
        if theme_name not in themes:
            fail_usage(
                f"unknown theme '{theme_name}'.\n"
                "  Run 'saber theme set' in a terminal to browse available themes.\n"
                "  Example: saber theme set dracula"
            )
        if not theme_manager.set_theme(theme_name):
            raise SystemExit(1)
        out(b.success(f"Theme set to: {theme_name}"))
        logger.info("theme.set.done", theme=theme_name)
        return

    if not sys.stdin.isatty():
        fail_usage(
            "THEME is required when stdin is not a terminal.\n"
            "  Example: saber theme set dracula"
        )

    async def interactive_set():
        current_theme = theme_manager.get_current_theme()
        choices = [
            Choice(
                title=f"{theme} (current)" if theme == current_theme else theme,
                value=theme,
            )
            for theme in themes
        ]
        selected_theme = await AsyncPrompter().select(
            "Select a theme:",
            choices=choices,
            default=current_theme,
            use_search_filter=True,
        )

        if selected_theme:
            if theme_manager.set_theme(selected_theme):
                out(b.success(f"Theme set to: {selected_theme}"))
                logger.info("theme.set.done", theme=selected_theme)
            else:
                fail("failed to set theme.")
        else:
            out(b.warn("Operation cancelled"))
            logger.info("theme.set.cancelled")

    asyncio.run(interactive_set())


@theme_app.command(
    help_epilogue=("Examples:\n\nsaber theme reset\n\nsaber theme reset --yes")
)
def reset(
    yes: Annotated[
        bool,
        cyclopts.Parameter(["--yes"], help="Skip confirmation prompt"),
    ] = False,
):
    """Reset to the default theme.

    Examples:
        saber theme reset
        saber theme reset --yes
    """

    if not confirm_action(
        yes=yes,
        prompt=f"Reset theme to {DEFAULT_THEME_NAME}?",
        non_interactive_command="saber theme reset --yes",
    ):
        out(b.warn("Operation cancelled"))
        logger.info("theme.reset.cancelled")
        return

    if theme_manager.reset_theme():
        out(b.success(f"Theme reset to default: {DEFAULT_THEME_NAME}"))
        logger.info("theme.reset.done", theme=DEFAULT_THEME_NAME)
    else:
        fail("failed to reset theme.")


def create_theme_app() -> cyclopts.App:
    """Return the theme management CLI app."""
    return theme_app

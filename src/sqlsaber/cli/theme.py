"""Theme management CLI commands."""

import asyncio
import json
import os
import sys
from pathlib import Path

import cyclopts
import questionary
from platformdirs import user_config_dir
from pygments.styles import get_all_styles

from sqlsaber.theme.manager import DEFAULT_THEME_NAME, create_console
from sqlsaber.config.logging import get_logger

console = create_console()
logger = get_logger(__name__)

# Create the theme management CLI app
theme_app = cyclopts.App(
    name="theme",
    help="Manage theme settings",
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
            console.print(f"[error]Error setting theme: {e}[/error]")
            logger.error("theme.set.error", theme=theme_name, error=str(e))
            return False

    def reset_theme(self) -> bool:
        """Reset to default theme."""
        try:
            if self.config_file.exists():
                self.config_file.unlink()
            return True
        except Exception as e:
            console.print(f"[error]Error resetting theme: {e}[/error]")
            logger.error("theme.reset.error", error=str(e))
            return False

    def get_available_themes(self) -> list[str]:
        """Get list of available Pygments themes."""
        return sorted(get_all_styles())


theme_manager = ThemeManager()


@theme_app.command
def set():
    """Set the theme to use for syntax highlighting."""
    logger.info("theme.set.start")

    async def interactive_set():
        themes = theme_manager.get_available_themes()
        current_theme = theme_manager.get_current_theme()

        # Create choices with current theme highlighted
        choices = [
            questionary.Choice(
                title=f"{theme} (current)" if theme == current_theme else theme,
                value=theme,
            )
            for theme in themes
        ]

        selected_theme = await questionary.select(
            "Select a theme:",
            choices=choices,
            default=current_theme,
            use_search_filter=True,
            use_jk_keys=False,
        ).ask_async()

        if selected_theme:
            if theme_manager.set_theme(selected_theme):
                console.print(f"[success]✓ Theme set to: {selected_theme}[/success]")
                logger.info("theme.set.done", theme=selected_theme)
            else:
                console.print("[error]✗ Failed to set theme[/error]")
                sys.exit(1)
        else:
            console.print("[warning]Operation cancelled[/warning]")
            logger.info("theme.set.cancelled")

    asyncio.run(interactive_set())


@theme_app.command
def reset():
    """Reset to the default theme."""

    if theme_manager.reset_theme():
        console.print(
            f"[success]✓ Theme reset to default: {DEFAULT_THEME_NAME}[/success]"
        )
        logger.info("theme.reset.done", theme=DEFAULT_THEME_NAME)
    else:
        console.print("[error]✗ Failed to reset theme[/error]")
        sys.exit(1)


def create_theme_app() -> cyclopts.App:
    """Return the theme management CLI app."""
    return theme_app

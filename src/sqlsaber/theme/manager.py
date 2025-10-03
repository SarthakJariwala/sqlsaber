"""Theme management for unified theming across Rich and prompt_toolkit."""

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

from platformdirs import user_config_dir
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.styles import get_style_by_name
from rich.console import Console
from rich.theme import Theme

DEFAULT_THEME_NAME = "nord"

DEFAULT_ROLE_PALETTE = {
    # base roles
    "primary": "cyan",
    "accent": "magenta",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "cyan",
    "muted": "dim",
    # components
    "table.header": "bold $primary",
    "panel.border.user": "$info",
    "panel.border.assistant": "$success",
    "panel.border.thread": "$primary",
    "spinner": "$warning",
    "status": "$warning",
    # domain-specific
    "key.primary": "bold $warning",
    "key.foreign": "bold $accent",
    "key.index": "bold $primary",
    "column.schema": "$info",
    "column.name": "white",
    "column.type": "$warning",
    "heading": "bold $primary",
    "section": "bold $accent",
    "title": "bold $success",
}

# Theme presets using exact Pygments colors
THEME_PRESETS = {
    # Nord - exact colors from pygments nord theme
    "nord": {
        "primary": "#81a1c1",  # Keyword (frost)
        "accent": "#b48ead",  # Number (aurora purple)
        "success": "#a3be8c",  # String (aurora green)
        "warning": "#ebcb8b",  # String.Escape (aurora yellow)
        "error": "#bf616a",  # Error/Generic.Error (aurora red)
        "info": "#88c0d0",  # Name.Function (frost cyan)
        "muted": "dim",
    },
    # Dracula - exact colors from pygments dracula theme
    "dracula": {
        "primary": "#bd93f9",  # purple
        "accent": "#ff79c6",  # pink
        "success": "#50fa7b",  # green
        "warning": "#f1fa8c",  # yellow
        "error": "#ff5555",  # red
        "info": "#8be9fd",  # cyan
        "muted": "dim",
    },
    # Solarized Light - exact colors from pygments solarized-light theme
    "solarized-light": {
        "primary": "#268bd2",  # blue
        "accent": "#d33682",  # magenta
        "success": "#859900",  # green
        "warning": "#b58900",  # yellow
        "error": "#dc322f",  # red
        "info": "#2aa198",  # cyan
        "muted": "dim",
    },
    # VS (Visual Studio Light) - exact colors from pygments vs theme
    "vs": {
        "primary": "#0000ff",  # Keyword (blue)
        "accent": "#2b91af",  # Keyword.Type/Name.Class
        "success": "#008000",  # Comment (green)
        "warning": "#b58900",  # (using solarized yellow as fallback)
        "error": "#dc322f",  # (using solarized red as fallback)
        "info": "#2aa198",  # (using solarized cyan as fallback)
        "muted": "dim",
    },
    # Material (approximation based on material design colors)
    "material": {
        "primary": "#89ddff",  # cyan
        "accent": "#f07178",  # pink/red
        "success": "#c3e88d",  # green
        "warning": "#ffcb6b",  # yellow
        "error": "#ff5370",  # red
        "info": "#82aaff",  # blue
        "muted": "dim",
    },
    # One Dark - exact colors from pygments one-dark theme
    "one-dark": {
        "primary": "#c678dd",  # Keyword (purple)
        "accent": "#e06c75",  # Name (red)
        "success": "#98c379",  # String (green)
        "warning": "#e5c07b",  # Keyword.Type (yellow)
        "error": "#e06c75",  # Name (red, used for errors)
        "info": "#61afef",  # Name.Function (blue)
        "muted": "dim",
    },
    # Lightbulb - exact colors from pygments lightbulb theme (minimal dark)
    "lightbulb": {
        "primary": "#73d0ff",  # Keyword.Type/Name.Class (blue_1)
        "accent": "#dfbfff",  # Number (magenta_1)
        "success": "#d5ff80",  # String (green_1)
        "warning": "#ffd173",  # Name.Function (yellow_1)
        "error": "#f88f7f",  # Error (red_1)
        "info": "#95e6cb",  # Name.Entity (cyan_1)
        "muted": "dim",
    },
}


def _load_user_theme_config() -> dict:
    """Load theme configuration from user config directory."""
    cfg_dir = user_config_dir("sqlsaber")
    path = os.path.join(cfg_dir, "theme.toml")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_refs(palette: dict[str, str]) -> dict[str, str]:
    """Resolve $var references in palette values."""
    out = {}
    for k, v in palette.items():
        if isinstance(v, str) and "$" in v:
            parts = v.split()
            resolved = []
            for part in parts:
                if part.startswith("$"):
                    ref = part[1:]
                    resolved.append(palette.get(ref, ""))
                else:
                    resolved.append(part)
            out[k] = " ".join(p for p in resolved if p)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class ThemeConfig:
    """Theme configuration."""

    name: str
    pygments_style: str
    roles: Dict[str, str]


class ThemeManager:
    """Manages theme configuration and provides themed components."""

    def __init__(self, cfg: ThemeConfig):
        self._cfg = cfg
        self._roles = _resolve_refs({**DEFAULT_ROLE_PALETTE, **cfg.roles})
        self._rich_theme = Theme(self._roles)
        self._pt_style = None

    @property
    def rich_theme(self) -> Theme:
        """Get Rich theme with semantic role mappings."""
        return self._rich_theme

    @property
    def pygments_style_name(self) -> str:
        """Get pygments style name for syntax highlighting."""
        return self._cfg.pygments_style

    def pt_style(self) -> PTStyle:
        """Get prompt_toolkit style derived from Pygments theme."""
        if self._pt_style is None:
            try:
                # Try to use Pygments style directly
                pygments_style = get_style_by_name(self._cfg.pygments_style)
                self._pt_style = style_from_pygments_cls(pygments_style)
            except Exception:
                # Fallback to basic style if Pygments theme not found
                self._pt_style = PTStyle.from_dict({})
        return self._pt_style

    def style(self, role: str) -> str:
        """Get style string for a semantic role."""
        return self._roles.get(role, "")


@lru_cache(maxsize=1)
def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    user_cfg = _load_user_theme_config()
    env_name = os.getenv("SQLSABER_THEME")

    name = (
        env_name or user_cfg.get("theme", {}).get("name") or DEFAULT_THEME_NAME
    ).lower()
    pygments_style = user_cfg.get("theme", {}).get("pygments_style") or name

    roles = dict(DEFAULT_ROLE_PALETTE)
    roles.update(THEME_PRESETS.get(name, {}))
    roles.update(user_cfg.get("roles", {}))

    cfg = ThemeConfig(name=name, pygments_style=pygments_style, roles=roles)
    return ThemeManager(cfg)


def create_console(**kwargs):
    """Create a Rich Console with theme applied."""
    # from rich.console import Console

    tm = get_theme_manager()
    return Console(theme=tm.rich_theme, **kwargs)

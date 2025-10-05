"""Theme management for unified theming across Rich and prompt_toolkit."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

from platformdirs import user_config_dir
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.styles import get_all_styles, get_style_by_name
from pygments.token import Token
from pygments.util import ClassNotFound
from rich.console import Console
from rich.theme import Theme

DEFAULT_THEME_NAME = "nord"

DEFAULT_ROLE_PALETTE = {
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

ROLE_TOKEN_PREFERENCES: dict[str, tuple] = {
    "primary": (
        Token.Keyword,
        Token.Keyword.Namespace,
        Token.Name.Tag,
    ),
    "accent": (
        Token.Name.Tag,
        Token.Keyword.Type,
        Token.Literal.Number,
        Token.Operator.Word,
    ),
    "success": (
        Token.Literal.String,
        Token.Generic.Inserted,
        Token.Name.Attribute,
    ),
    "warning": (
        Token.Literal.String.Escape,
        Token.Name.Constant,
        Token.Generic.Emph,
    ),
    "error": (
        Token.Error,
        Token.Generic.Error,
        Token.Generic.Deleted,
        Token.Name.Exception,
    ),
    "info": (
        Token.Name.Function,
        Token.Name.Builtin,
        Token.Keyword.Type,
    ),
    "muted": (
        Token.Comment,
        Token.Generic.Subheading,
        Token.Text,
    ),
}


def _normalize_hex(color: str | None) -> str | None:
    if not color:
        return None
    color = color.strip()
    if not color:
        return None
    if color.startswith("#"):
        color = color[1:]
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6:
        return None
    return f"#{color.lower()}"


def _build_role_palette_from_style(style_name: str) -> dict[str, str]:
    try:
        style_cls = get_style_by_name(style_name)
    except ClassNotFound:
        return {}

    palette: dict[str, str] = {}
    try:
        base_color = _normalize_hex(style_cls.style_for_token(Token.Text).get("color"))
    except KeyError:
        base_color = None
    for role, tokens in ROLE_TOKEN_PREFERENCES.items():
        for token in tokens:
            try:
                style_def = style_cls.style_for_token(token)
            except KeyError:
                continue
            color = _normalize_hex(style_def.get("color"))
            if not color or color == base_color:
                continue
            if role == "accent" and color == palette.get("primary"):
                continue
            palette[role] = color
            break
    return palette


def _load_user_theme_config() -> dict:
    """Load theme configuration from user config directory."""
    cfg_dir = user_config_dir("sqlsaber")
    path = os.path.join(cfg_dir, "theme.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


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

    if env_name and env_name.lower() not in get_all_styles():
        env_name = None

    name = (
        env_name or user_cfg.get("theme", {}).get("name") or DEFAULT_THEME_NAME
    ).lower()
    pygments_style = user_cfg.get("theme", {}).get("pygments_style") or name

    roles = dict(DEFAULT_ROLE_PALETTE)
    roles.update(_build_role_palette_from_style(pygments_style))
    roles.update(user_cfg.get("roles", {}))

    cfg = ThemeConfig(name=name, pygments_style=pygments_style, roles=roles)
    return ThemeManager(cfg)


def create_console(**kwargs):
    """Create a Rich Console with theme applied."""
    # from rich.console import Console

    tm = get_theme_manager()
    return Console(theme=tm.rich_theme, **kwargs)

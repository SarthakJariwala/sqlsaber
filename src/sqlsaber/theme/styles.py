"""Role to ANSI callables."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pygments import highlight
from pygments.formatters.terminal256 import TerminalTrueColorFormatter
from pygments.lexers import get_lexer_by_name
from pygments.lexers.special import TextLexer
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound

from sqlsaber.render.blocks import Role
from sqlsaber.theme.manager import get_theme_manager

type StyleFn = Callable[[str], str]

_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")

INFO_FALLBACK = (125, 211, 252)
SUCCESS_FALLBACK = (134, 239, 172)
WARNING_FALLBACK = (251, 191, 36)
MUTED_FALLBACK = (148, 163, 184)
ERROR_FALLBACK = (248, 113, 113)
USER_BG_FALLBACK = (30, 41, 59)
ACCENT_FALLBACK = (167, 139, 250)
PRIMARY_FALLBACK = INFO_FALLBACK

_ROLE_FALLBACKS: dict[Role, tuple[int, int, int]] = {
    "primary": PRIMARY_FALLBACK,
    "accent": ACCENT_FALLBACK,
    "success": SUCCESS_FALLBACK,
    "warning": WARNING_FALLBACK,
    "error": ERROR_FALLBACK,
    "info": INFO_FALLBACK,
    "muted": MUTED_FALLBACK,
}

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "red": (205, 49, 49),
    "green": (13, 188, 121),
    "yellow": (229, 229, 16),
    "blue": (36, 114, 200),
    "magenta": (188, 63, 188),
    "cyan": (17, 168, 205),
    "white": (229, 229, 229),
    "bright_black": (102, 102, 102),
    "bright_red": (241, 76, 76),
    "bright_green": (35, 209, 139),
    "bright_yellow": (245, 245, 67),
    "bright_blue": (59, 142, 234),
    "bright_magenta": (214, 112, 214),
    "bright_cyan": (41, 184, 219),
    "bright_white": (255, 255, 255),
}


def fg(r: int, g: int, b: int) -> StyleFn:
    """Truecolor foreground wrapper.

    Args:
        r: Red 0-255.
        g: Green 0-255.
        b: Blue 0-255.

    Returns:
        A function that wraps text in an SGR foreground sequence.
    """
    code = f"\x1b[38;2;{r};{g};{b}m"
    return lambda text: f"{code}{text}\x1b[39m"


def bg(r: int, g: int, b: int) -> StyleFn:
    """Truecolor background wrapper that restores after SGR resets.

    Args:
        r: Red 0-255.
        g: Green 0-255.
        b: Blue 0-255.

    Returns:
        A function that wraps text in an SGR background sequence.
    """
    code = f"\x1b[48;2;{r};{g};{b}m"

    def apply(text: str) -> str:
        def restore(match: re.Match[str]) -> str:
            parameters = match.group(1).split(";")
            clears_background = not match.group(1) or any(
                parameter in {"0", "00", "49"} for parameter in parameters
            )
            return match.group(0) + code if clears_background else match.group(0)

        return f"{code}{_SGR_RE.sub(restore, text)}\x1b[49m"

    return apply


def bold(text: str) -> str:
    """Wrap text in a bold SGR pair."""
    return f"\x1b[1m{text}\x1b[22m"


def italic(text: str) -> str:
    """Wrap text in an italic SGR pair."""
    return f"\x1b[3m{text}\x1b[23m"


def strikethrough(text: str) -> str:
    """Wrap text in a strikethrough SGR pair."""
    return f"\x1b[9m{text}\x1b[29m"


def underline(text: str) -> str:
    """Wrap text in an underline SGR pair."""
    return f"\x1b[4m{text}\x1b[24m"


def identity(text: str) -> str:
    """Return text unchanged."""
    return text


def hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    """Parse ``#rgb`` / ``#rrggbb`` into an RGB tuple.

    Args:
        color: Hex colour or None.

    Returns:
        ``(r, g, b)`` or None when the value is not a hex colour.
    """
    if not color:
        return None
    normalized = color.strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if len(normalized) == 3:
        normalized = "".join(ch * 2 for ch in normalized)
    if len(normalized) != 6:
        return None
    try:
        return (
            int(normalized[0:2], 16),
            int(normalized[2:4], 16),
            int(normalized[4:6], 16),
        )
    except ValueError:
        return None


def blend(
    base: tuple[int, int, int], accent: tuple[int, int, int], ratio: float
) -> tuple[int, int, int]:
    """Mix two RGB colours.

    Args:
        base: Background colour.
        accent: Colour mixed in.
        ratio: Fraction of accent, 0 to 1.

    Returns:
        Blended RGB tuple.
    """
    return (
        round(base[0] * (1 - ratio) + accent[0] * ratio),
        round(base[1] * (1 - ratio) + accent[1] * ratio),
        round(base[2] * (1 - ratio) + accent[2] * ratio),
    )


@dataclass(frozen=True, slots=True)
class StyleSpec:
    """Parsed role string: colour plus attribute flags."""

    rgb: tuple[int, int, int] | None = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False


def parse_role_style(value: str) -> StyleSpec:
    """Parse a theme.json / ThemeManager role string without Rich.

    Exotic values such as ``white on blue`` keep the foreground and drop
    the background.

    Args:
        value: Role string after ``$ref`` resolution.

    Returns:
        A ``StyleSpec``.
    """
    is_bold = False
    is_italic = False
    is_underline = False
    is_strike = False
    rgb: tuple[int, int, int] | None = None
    tokens = value.split()
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        lowered = token.lower()
        if lowered == "on":
            skip_next = True
            continue
        if lowered in {"bold", "b"}:
            is_bold = True
            continue
        if lowered in {"italic", "i", "dim"}:
            if lowered != "dim":
                is_italic = True
            continue
        if lowered in {"underline", "u"}:
            is_underline = True
            continue
        if lowered in {"strike", "strikethrough"}:
            is_strike = True
            continue
        parsed = hex_to_rgb(token) or _NAMED_COLORS.get(lowered)
        if parsed is not None:
            rgb = parsed
    return StyleSpec(
        rgb=rgb,
        bold=is_bold,
        italic=is_italic,
        underline=is_underline,
        strikethrough=is_strike,
    )


def compile_style(spec: StyleSpec) -> StyleFn:
    """Turn a ``StyleSpec`` into an SGR wrapper.

    Args:
        spec: Parsed style.

    Returns:
        A function that styles a string.
    """
    fns: list[StyleFn] = []
    if spec.rgb is not None:
        fns.append(fg(*spec.rgb))
    if spec.bold:
        fns.append(bold)
    if spec.italic:
        fns.append(italic)
    if spec.underline:
        fns.append(underline)
    if spec.strikethrough:
        fns.append(strikethrough)
    if not fns:
        return identity

    def apply(text: str) -> str:
        styled = text
        for fn in fns:
            styled = fn(styled)
        return styled

    return apply


@dataclass(frozen=True, slots=True)
class Styles:
    """Resolved theme: ANSI callables per role plus saber-tui markdown theme."""

    pygments_style_name: str
    roles: Mapping[Role, StyleFn]
    panel_bg: StyleFn
    user_bg: StyleFn
    user_fg: StyleFn
    assistant_fg: StyleFn
    system_fg: StyleFn
    spinner_fg: StyleFn
    status_fg: StyleFn
    markdown: Any

    def role(self, role: Role | None) -> StyleFn:
        """Identity when ``role`` is None.

        Args:
            role: Semantic role or None.

        Returns:
            The matching style function.
        """
        if role is None:
            return identity
        return self.roles.get(role, identity)

    @property
    def muted_fg(self) -> StyleFn:
        """Muted foreground, used by chat chrome."""
        return self.roles["muted"]


def _role_rgb(
    roles: Mapping[str, str], name: str, fallback: tuple[int, int, int]
) -> tuple[int, int, int]:
    spec = parse_role_style(roles.get(name, ""))
    return spec.rgb or fallback


def _role_fn(
    roles: Mapping[str, str], name: str, fallback: tuple[int, int, int]
) -> StyleFn:
    spec = parse_role_style(roles.get(name, ""))
    if spec.rgb is None:
        spec = StyleSpec(
            rgb=fallback,
            bold=spec.bold,
            italic=spec.italic,
            underline=spec.underline,
            strikethrough=spec.strikethrough,
        )
    return compile_style(spec)


def _markdown_theme(roles: Mapping[str, str], pygments_style_name: str) -> Any:
    from saber_tui.components.markdown import MarkdownTheme

    primary = _role_fn(roles, "primary", PRIMARY_FALLBACK)
    accent = _role_fn(roles, "accent", ACCENT_FALLBACK)
    info = _role_fn(roles, "info", INFO_FALLBACK)
    warning = _role_fn(roles, "warning", WARNING_FALLBACK)
    muted = _role_fn(roles, "muted", MUTED_FALLBACK)
    formatter = TerminalTrueColorFormatter(style=pygments_style_name)

    def highlight_code(code: str, language: str | None) -> list[str]:
        try:
            lexer = get_lexer_by_name(language) if language else TextLexer()
        except ClassNotFound:
            lexer = TextLexer()
        rendered = highlight(code, lexer, formatter).rstrip("\n")
        return rendered.split("\n") if rendered else []

    return MarkdownTheme(
        heading=primary,
        link=info,
        link_url=muted,
        code=warning,
        code_block_border=muted,
        quote=muted,
        quote_border=muted,
        hr=muted,
        list_bullet=accent,
        bold=bold,
        italic=italic,
        strikethrough=strikethrough,
        underline=underline,
        highlight_code=highlight_code,
    )


def _user_bg(roles: Mapping[str, str], pygments_style_name: str) -> StyleFn:
    try:
        style = get_style_by_name(pygments_style_name)
    except ClassNotFound:
        return bg(*USER_BG_FALLBACK)
    highlight_color = hex_to_rgb(getattr(style, "highlight_color", None))
    if highlight_color is not None:
        return bg(*highlight_color)
    background = hex_to_rgb(getattr(style, "background_color", None))
    if background is None:
        return bg(*USER_BG_FALLBACK)
    accent = _role_rgb(roles, "panel.border.user", INFO_FALLBACK)
    return bg(*blend(background, accent, 0.16))


def _panel_bg(roles: Mapping[str, str], pygments_style_name: str) -> StyleFn:
    try:
        style = get_style_by_name(pygments_style_name)
    except ClassNotFound:
        return bg(*USER_BG_FALLBACK)
    background = (
        hex_to_rgb(getattr(style, "background_color", None)) or USER_BG_FALLBACK
    )
    accent = _role_rgb(roles, "panel.border.assistant", SUCCESS_FALLBACK)
    return bg(*blend(background, accent, 0.10))


@lru_cache(maxsize=1)
def get_styles() -> Styles:
    """Cached styles built from the pygments theme plus theme.json overrides.

    Returns:
        A ``Styles`` value. Imports saber-tui ``MarkdownTheme`` on first call.
    """
    tm = get_theme_manager()
    roles = tm._roles
    pygments_style_name = tm.pygments_style_name
    role_fns: dict[Role, StyleFn] = {
        name: _role_fn(roles, name, fallback)
        for name, fallback in _ROLE_FALLBACKS.items()
    }
    return Styles(
        pygments_style_name=pygments_style_name,
        roles=role_fns,
        panel_bg=_panel_bg(roles, pygments_style_name),
        user_bg=_user_bg(roles, pygments_style_name),
        user_fg=_role_fn(roles, "panel.border.user", INFO_FALLBACK),
        assistant_fg=_role_fn(roles, "panel.border.assistant", SUCCESS_FALLBACK),
        system_fg=_role_fn(roles, "warning", WARNING_FALLBACK),
        spinner_fg=_role_fn(roles, "spinner", WARNING_FALLBACK),
        status_fg=_role_fn(roles, "status", WARNING_FALLBACK),
        markdown=_markdown_theme(roles, pygments_style_name),
    )

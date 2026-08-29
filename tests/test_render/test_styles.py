"""Tests for Rich-free role style parsing."""

from sqlsaber.theme.styles import compile_style, hex_to_rgb, parse_role_style


def test_parse_hex_and_bold() -> None:
    spec = parse_role_style("bold #88c0d0")
    assert spec.bold is True
    assert spec.rgb == (0x88, 0xC0, 0xD0)


def test_parse_drops_background() -> None:
    spec = parse_role_style("white on blue")
    assert spec.rgb == (229, 229, 229)


def test_hex_to_rgb_short() -> None:
    assert hex_to_rgb("#abc") == (0xAA, 0xBB, 0xCC)


def test_compile_style_wraps_text() -> None:
    spec = parse_role_style("#ff0000")
    styled = compile_style(spec)("hi")
    assert "hi" in styled
    assert "\x1b[" in styled

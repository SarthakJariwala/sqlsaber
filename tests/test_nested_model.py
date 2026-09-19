"""NestedModel parse, pin, and inherit."""

from __future__ import annotations

import pytest

from sqlsaber.config.providers import all_keys
from sqlsaber.nested_model import (
    INHERIT,
    ModelId,
    Pinned,
    parse_model_id,
    parse_nested_model,
    pin,
)


def test_parse_model_id_canonicalizes_google_alias() -> None:
    parsed = parse_model_id("google-gla:gemini-2.5-pro")
    assert parsed == ModelId(provider="google", model="gemini-2.5-pro")
    assert str(parsed) == "google:gemini-2.5-pro"


def test_parse_model_id_rejects_bare_id() -> None:
    with pytest.raises(ValueError, match="PROVIDER:MODEL") as exc:
        parse_model_id("gpt-5-mini")
    for key in all_keys():
        assert key in str(exc.value)


def test_parse_nested_model_blank_is_inherit() -> None:
    assert parse_nested_model(None) is INHERIT
    assert parse_nested_model("") is INHERIT
    assert parse_nested_model("  ") is INHERIT
    assert isinstance(parse_nested_model("openai:gpt-5-mini"), Pinned)


def test_pin_rejects_codex_api_key() -> None:
    with pytest.raises(ValueError, match="do not accept API keys"):
        pin("openai-codex:gpt-5.6-sol", api_key="sk-test")

"""Tests for tool override normalization."""

import pytest

from sqlsaber.nested_model import pin
from sqlsaber.overrides import ModelOverides, normalize_tool_overides


def test_normalize_tool_overides_accepts_model_overides_instance() -> None:
    normalized = normalize_tool_overides(
        {"viz": ModelOverides(model_name=" openai:gpt-5-mini ", api_key=" sk-test ")}
    )

    assert normalized == {"viz": pin("openai:gpt-5-mini", api_key="sk-test")}


def test_normalize_tool_overides_accepts_pin() -> None:
    override = pin("anthropic:claude-haiku-4-5")
    assert normalize_tool_overides({"analyze_data": override}) == {
        "analyze_data": override
    }


def test_normalize_tool_overides_accepts_model_id_string() -> None:
    assert normalize_tool_overides({"viz": "openai:gpt-5-mini"}) == {
        "viz": pin("openai:gpt-5-mini")
    }


def test_normalize_tool_overides_accepts_mapping_values() -> None:
    normalized = normalize_tool_overides(
        {"viz": {"model_name": "anthropic:claude-3-5-haiku", "api_key": None}}
    )

    assert normalized == {"viz": pin("anthropic:claude-3-5-haiku")}


def test_normalize_tool_overides_drops_empty_values() -> None:
    normalized = normalize_tool_overides(
        {"viz": {"model_name": "   ", "api_key": "   "}, "sandbox": None}
    )

    assert normalized == {}


def test_normalize_tool_overides_rejects_api_key_without_model_name() -> None:
    with pytest.raises(ValueError, match="api_key override requires model_name"):
        normalize_tool_overides({"viz": {"api_key": "sk-test"}})


def test_normalize_tool_overides_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown override fields"):
        normalize_tool_overides(
            {"viz": {"model_name": "openai:gpt-5-mini", "provider": "openai"}}
        )

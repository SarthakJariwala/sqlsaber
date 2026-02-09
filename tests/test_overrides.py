"""Tests for runtime tool override normalization."""

from types import SimpleNamespace

import pytest

from sqlsaber.overrides import (
    ModelOverides,
    build_tool_run_deps,
    get_tool_model_overide_from_ctx,
    normalize_tool_overides,
)


def test_normalize_tool_overides_accepts_model_overides_instance() -> None:
    normalized = normalize_tool_overides(
        {"viz": ModelOverides(model_name=" openai:gpt-5-mini ", api_key=" sk-test ")}
    )

    assert normalized == {
        "viz": ModelOverides(model_name="openai:gpt-5-mini", api_key="sk-test")
    }


def test_normalize_tool_overides_accepts_mapping_values() -> None:
    normalized = normalize_tool_overides(
        {"viz": {"model_name": "anthropic:claude-3-5-haiku", "api_key": None}}
    )

    assert normalized == {
        "viz": ModelOverides(model_name="anthropic:claude-3-5-haiku", api_key=None)
    }


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


def test_build_tool_run_deps_immutable_mapping() -> None:
    deps = build_tool_run_deps({"viz": ModelOverides(model_name="openai:gpt-5-mini")})

    with pytest.raises(TypeError):
        deps.tool_overides["viz"] = ModelOverides(model_name="anthropic:claude-3")


def test_get_tool_model_overide_from_ctx_returns_override() -> None:
    ctx = SimpleNamespace(
        deps=build_tool_run_deps({"viz": ModelOverides(model_name="openai:gpt-5-mini")})
    )

    overide = get_tool_model_overide_from_ctx(ctx, "viz")
    assert overide is not None
    assert overide.model_name == "openai:gpt-5-mini"


def test_get_tool_model_overide_from_ctx_without_tool_deps() -> None:
    assert get_tool_model_overide_from_ctx(SimpleNamespace(), "viz") is None

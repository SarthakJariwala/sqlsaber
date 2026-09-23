"""Tests for visualization plugin settings."""

from types import SimpleNamespace

from sqlsaber_viz.settings import settings


def test_settings_declares_model_and_binds_it() -> None:
    assert settings.field("model").env == "SQLSABER_VIZ_MODEL"
    factory = settings.bind({"model": "anthropic:claude-sonnet-4-5"}, {})
    context = SimpleNamespace(
        query_result_store=object(),
        tool_overrides={},
    )

    capability = factory(context)

    assert capability.tool._model_name == "anthropic:claude-sonnet-4-5"


def test_unset_model_defers_to_session_resolution() -> None:
    factory = settings.bind({}, {})
    context = SimpleNamespace(
        query_result_store=object(),
        tool_overrides={},
    )

    capability = factory(context)

    assert capability.tool._model_name is None

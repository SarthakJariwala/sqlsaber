"""Bundled handoff uses the ordinary plugin and model-tool paths."""

import subprocess
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.bundled.handoff import capability
from sqlsaber.bundled.handoff.runtime import Handoff
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.config.settings import Config


def options(**kwargs):
    return SQLSaberOptions(
        database="sqlite:///:memory:",
        settings=Config.in_memory(
            model_name="openai:main",
            api_keys={"openai": "configured-key"},
        ),
        **kwargs,
    )


@pytest.mark.parametrize("request_limit", [1, 3])
async def test_model_invokes_handoff_with_run_history_and_shared_usage(
    monkeypatch, request_limit
):
    child_prompts = []

    def child(messages, info):
        assert info.function_tools == []  # No recursion into handoff or SQL tools.
        child_prompts.append(messages[-1].parts[0].content)
        return ModelResponse(
            parts=[TextPart("Continue by investigating returns.")],
            usage=RequestUsage(input_tokens=17, output_tokens=9),
        )

    def parent(messages, info):
        tools = {tool.name: tool for tool in info.function_tools}
        assert "draft_handoff" in tools
        assert set(tools["draft_handoff"].parameters_json_schema["properties"]) == {
            "goal"
        }
        last = messages[-1].parts[0]
        if isinstance(last, ToolReturnPart):
            assert last.content == "Continue by investigating returns."
            return ModelResponse(parts=[TextPart("Here is your draft.")])
        return ModelResponse(
            parts=[ToolCallPart("draft_handoff", {"goal": "Investigate returns"})]
        )

    async with SQLSaber(options=options(capabilities=(capability,))) as saber:
        monkeypatch.setattr(
            PluginContext,
            "resolve_subagent_model",
            lambda *args, **kwargs: ("child", FunctionModel(child), "test"),
        )
        history = [
            ModelRequest(parts=[UserPromptPart("Revenue was 420; refunds were 37.")])
        ]
        saber._message_history = history.copy()
        thread_id = saber.info.thread_id
        with saber.agent.agent.override(model=FunctionModel(parent)):
            if request_limit == 1:
                with pytest.raises(UsageLimitExceeded):
                    await saber.query(
                        "Draft a handoff", usage_limits=UsageLimits(request_limit=1)
                    )
                assert child_prompts == []
                assert saber._message_history == history
            else:
                result = await saber.query(
                    "Draft a handoff", usage_limits=UsageLimits(request_limit=3)
                )
                assert result.usage.requests == 3
                assert result.usage.input_tokens >= 17
                assert len(child_prompts) == 1
                assert "Revenue was 420; refunds were 37." in child_prompts[0]
                assert (
                    "<handoff_goal>\nInvestigate returns\n</handoff_goal>"
                    in child_prompts[0]
                )
                assert saber._message_history[0] == history[0]
        assert saber.info.thread_id == thread_id


@pytest.mark.parametrize(
    "plugin_model,expected_model,expected_key",
    [
        (None, "openai:session", "session-key"),
        ("anthropic:plugin", "anthropic:plugin", None),
    ],
)
async def test_model_precedence_and_session_auth(
    monkeypatch, plugin_model, expected_model, expected_key
):
    factory = partial(capability, model_name=plugin_model)
    async with SQLSaber(
        options=options(
            capabilities=(factory,), model_name="openai:session", api_key="session-key"
        )
    ) as saber:
        plugin = next(c for c in saber.agent.capabilities if isinstance(c, Handoff))
        resolve = Mock(
            return_value=SimpleNamespace(
                model=FunctionModel(
                    lambda messages, info: ModelResponse(parts=[TextPart("draft")])
                ),
                provider="test",
            )
        )
        monkeypatch.setattr("sqlsaber.agents.model_factory.resolve_model", resolve)
        assert await saber.draft_handoff("continue") == "draft"
        resolve.assert_called_once_with(
            plugin.context.config.auth, expected_model, api_key_override=expected_key
        )


async def test_rebuild_refreshes_same_plugin_and_factory_remains_callable():
    async with SQLSaber(options=options(capabilities=(capability,))) as saber:
        plugin = next(c for c in saber.agent.capabilities if isinstance(c, Handoff))
        original_context = plugin.context
        plugin.context.config.model.name = "openai:updated"
        saber.reload_model_settings()
        assert plugin in saber.agent.capabilities
        assert plugin.context is not original_context
        assert plugin.context.main_model_name == "openai:updated"
        from sqlsaber.bundled.handoff import capability as reimported

        assert callable(reimported)


def test_installed_cli_settings_bind_and_disable(monkeypatch, tmp_path):
    from sqlsaber.cli.session import configured_capabilities
    from sqlsaber.config import plugins

    monkeypatch.setattr("platformdirs.user_config_dir", lambda *args: str(tmp_path))
    monkeypatch.delenv("SQLSABER_HANDOFF_MODEL", raising=False)
    entry = plugins.installed_plugins()["handoff"]
    monkeypatch.setattr(plugins, "installed_plugins", lambda: {"handoff": entry})
    declaration = plugins.load_plugin_settings("handoff")
    assert declaration is not None
    assert declaration.fields[0].help == "Unset: use the active session model."
    assert [item.name for item in configured_capabilities()] == ["handoff"]
    store = plugins.PluginConfigStore()
    store.save("handoff", plugins.SavedPlugin(settings={"model": "openai:saved"}))
    factory = configured_capabilities()[0]
    context = Mock(spec=PluginContext)
    plugin = factory(context)
    assert isinstance(plugin, Handoff)
    assert plugin._model_name_override == "openai:saved"
    context.resolve_subagent_model.assert_not_called()
    monkeypatch.setenv("SQLSABER_HANDOFF_MODEL", "openai:environment")
    assert (
        configured_capabilities()[0](context)._model_name_override
        == "openai:environment"
    )
    store.save(
        "handoff",
        plugins.SavedPlugin(enabled=False, settings={"model": "openai:saved"}),
    )
    assert configured_capabilities() == ()


def test_settings_and_factory_import_without_loading_model_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from importlib.metadata import entry_points; import sys; "
                "eps = entry_points(); "
                "[ep.load() for group in ('sqlsaber.capabilities', 'sqlsaber.plugin_settings') "
                "for ep in eps.select(group=group, name='handoff')]; "
                "assert 'pydantic_ai' not in sys.modules; "
                "assert 'openai' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

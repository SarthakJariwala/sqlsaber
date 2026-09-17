"""The same declaration drives headless edits, prompts, and CLI binding."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sqlsaber.cli import plugins as cli
from sqlsaber.cli.session import cli_sqlsaber_options, configured_capabilities
from sqlsaber.config import plugins as config
from sqlsaber.plugin_settings import PluginSettings, Setting


@pytest.fixture
def declaration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    monkeypatch.delenv("EXAMPLE_PROVIDER", raising=False)
    monkeypatch.delenv("EXAMPLE_TOKEN", raising=False)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")

    def validate(values):
        if values["memory"] <= 0:
            raise ValueError("memory must be positive")

    factory = Mock()
    declaration = PluginSettings(
        fields=(
            Setting(
                "provider",
                "Where?",
                choices=("local", "cloud"),
                default="local",
                env="EXAMPLE_PROVIDER",
            ),
            Setting("memory", "Memory", kind="integer", default=512, advanced=True),
            Setting(
                "image",
                "Image",
                default="local-image",
                when=lambda values: values["provider"] == "local",
            ),
            Setting(
                "template",
                "Template",
                when=lambda values: values["provider"] == "cloud",
            ),
            Setting(
                "token",
                "API token",
                kind="secret",
                env="EXAMPLE_TOKEN",
                when=lambda values: values["provider"] == "cloud",
            ),
        ),
        validate=validate,
        bind=Mock(return_value=factory),
        notice=lambda values: (
            "Uploads data to cloud; charges may apply."
            if values["provider"] == "cloud"
            else None
        ),
    )
    entries = {
        "example": SimpleNamespace(
            load=Mock(side_effect=AssertionError("must bind declaration"))
        )
    }
    monkeypatch.setattr(cli, "installed_plugins", lambda: entries)
    monkeypatch.setattr(config, "installed_plugins", lambda: entries)
    monkeypatch.setattr(cli, "load_plugin_settings", lambda name: declaration)
    monkeypatch.setattr(config, "load_plugin_settings", lambda name: declaration)
    return declaration


def test_setup_persists_typed_values_and_bind_reads_them(declaration, capsys):
    cli.setup("example", set_values=["memory=3072", "image=custom-image"])
    assert config.PluginConfigStore().get("example").settings == {
        "memory": 3072,
        "image": "custom-image",
    }
    cli.show("example")
    output = capsys.readouterr().out
    assert "3072" in output and "custom-image" in output and "saved" in output
    factories = configured_capabilities()
    assert len(factories) == 1
    values, secrets = declaration.bind.call_args.args
    assert values == {"provider": "local", "memory": 3072, "image": "custom-image"}
    assert secrets == {}


def test_precedence_and_inactive_values(declaration, monkeypatch):
    saved = {"provider": "local", "memory": 1536, "image": "private-local-image"}
    monkeypatch.setenv("EXAMPLE_PROVIDER", "cloud")
    resolved = config.resolve_settings("example", declaration, saved)
    assert resolved.values == {"provider": "cloud", "memory": 1536, "template": None}
    assert resolved.sources["provider"] == "environment (EXAMPLE_PROVIDER)"
    explicit = config.resolve_settings(
        "example", declaration, saved, overrides={"provider": "local", "memory": 2048}
    )
    assert explicit.values == {
        "provider": "local",
        "memory": 2048,
        "image": "private-local-image",
    }
    assert explicit.sources["provider"] == "option"


@pytest.mark.parametrize(
    "assignment",
    [
        "memory=0",
        "memory=1.5",
        "provider=typo",
        "unknown=value",
        "template=wrong-provider",
        "token=do-not-print",
    ],
)
def test_invalid_edits_never_write_or_echo_secret(declaration, assignment, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.setup("example", set_values=[assignment])
    assert exc.value.code == 2
    assert not config.PluginConfigStore().path.exists()
    output = capsys.readouterr()
    assert "do-not-print" not in output.out + output.err


def test_cloud_requires_confirmation_headlessly(declaration, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StringIO())
    with pytest.raises(SystemExit) as exc:
        cli.setup("example", set_values=["provider=cloud"])
    assert exc.value.code == 2
    assert not config.PluginConfigStore().path.exists()
    assert "--yes" in capsys.readouterr().err
    cli.setup("example", set_values=["provider=cloud"], yes=True)
    assert config.PluginConfigStore().get("example").settings == {"provider": "cloud"}


def test_secret_stdin_keyring_only_and_redaction(declaration, monkeypatch, capsys):
    import keyring

    stored = {}
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda service, key, value: stored.update({(service, key): value}),
    )
    monkeypatch.setattr(
        keyring, "get_password", lambda service, key: stored.get((service, key))
    )
    monkeypatch.setattr("sys.stdin", StringIO("sensitive-test-value\n"))
    cli.setup("example", set_values=["provider=cloud"], secret_stdin="token", yes=True)
    assert stored[("sqlsaber-plugins", "example.token")] == "sensitive-test-value"
    cli.show("example")
    output = capsys.readouterr()
    assert "sensitive-test-value" not in output.out + output.err
    assert "sensitive-test-value" not in config.PluginConfigStore().path.read_text()
    configured_capabilities()
    assert declaration.bind.call_args.args[1] == {"token": "sensitive-test-value"}
    monkeypatch.setenv("EXAMPLE_TOKEN", "environment-secret")
    configured_capabilities()
    assert declaration.bind.call_args.args[1] == {"token": "environment-secret"}


def test_keyring_failure_does_not_claim_saved(declaration, monkeypatch, capsys):
    import keyring

    monkeypatch.setattr(keyring, "set_password", lambda *args: None)
    monkeypatch.setattr(keyring, "get_password", lambda *args: None)
    monkeypatch.setattr("sys.stdin", StringIO("secret-to-discard\n"))
    with pytest.raises(SystemExit):
        cli.setup(
            "example", set_values=["provider=cloud"], secret_stdin="token", yes=True
        )
    assert not config.PluginConfigStore().path.exists()
    output = capsys.readouterr()
    assert "Could not save" in output.err
    assert "secret-to-discard" not in output.out + output.err


def test_disable_skips_loading_and_set_preserves_disable(declaration, monkeypatch):
    cli.disable("example")
    cli.set_setting("example", "memory", "4096")
    assert not config.PluginConfigStore().get("example").enabled
    monkeypatch.setattr(
        config,
        "load_plugin_settings",
        Mock(side_effect=AssertionError("disabled plugins must not load")),
    )
    assert configured_capabilities() == ()
    declaration.bind.assert_not_called()


def test_unset_restores_default_and_explicit_capabilities_bypass_cli(declaration):
    cli.setup("example", set_values=["memory=2048"])
    cli.unset("example", "memory")
    configured_capabilities()
    assert declaration.bind.call_args.args[0]["memory"] == 512
    declaration.bind.reset_mock()
    assert cli_sqlsaber_options(capabilities=[]).capabilities == []
    declaration.bind.assert_not_called()


def test_no_declaration_keeps_legacy_factory(declaration, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(config, "load_plugin_settings", lambda name: None)
    monkeypatch.setattr(
        config,
        "installed_plugins",
        lambda: {"legacy": SimpleNamespace(load=lambda: factory)},
    )
    factories = configured_capabilities()
    assert len(factories) == 1
    assert factories[0].create is factory


def test_corrupt_file_does_not_fall_back(declaration):
    path = config.PluginConfigStore().path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": 99, "plugins": {}}')
    with pytest.raises(ValueError, match="Repair the file"):
        configured_capabilities()
    declaration.bind.assert_not_called()


@pytest.mark.asyncio
async def test_wizard_asks_only_applicable_fields_and_cancellation_does_not_write(
    declaration, monkeypatch
):
    prompter = SimpleNamespace(
        select=AsyncMock(return_value="cloud"),
        text=AsyncMock(return_value="template-id"),
        secret=AsyncMock(return_value="masked-secret"),
        confirm=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(cli, "AsyncPrompter", lambda: prompter)
    values = {}
    secrets = await cli._wizard("example", declaration, values, False)
    assert values == {"provider": "cloud", "template": "template-id"}
    assert secrets == {"token": "masked-secret"}
    assert prompter.text.call_args.args[0] == "Template"
    assert prompter.confirm.call_args.args[0] == "Configure advanced settings?"
    prompter.select.return_value = None
    assert await cli._wizard("example", declaration, {}, False) is None
    assert not config.PluginConfigStore().path.exists()


@pytest.mark.parametrize(
    "kind,value",
    [
        ("integer", True),
        ("integer", "nan"),
        ("number", "inf"),
        ("number", False),
        ("boolean", "maybe"),
    ],
)
def test_strict_field_parsing(kind, value):
    with pytest.raises(ValueError):
        Setting("value", "Value", kind=kind).parse(value)


@pytest.mark.parametrize(
    "command", ["list", "show", "setup", "set", "unset", "enable", "disable"]
)
def test_command_help_has_examples(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.plugins_app([command, "--help"])
    assert exc.value.code == 0
    assert "Example" in capsys.readouterr().out

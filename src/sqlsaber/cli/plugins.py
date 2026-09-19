"""Generic CLI editor for plugin-owned settings declarations."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated

import cyclopts

from sqlsaber.cli.output import confirm_sync, fail_usage, out
from sqlsaber.cli.prompts import AsyncPrompter
from sqlsaber.config.plugins import (
    PluginConfigStore,
    PluginSetupRequired,
    SavedPlugin,
    installed_plugins,
    load_plugin_settings,
    migrate_legacy_plugin_models,
    resolve_settings,
    save_secret,
)
from sqlsaber.plugin_settings import PluginSettings, SettingValue
from sqlsaber.render import blocks as b

plugins_app = cyclopts.App(
    name="plugins",
    help="Configure installed plugins for future CLI sessions",
    help_epilogue="Examples:\n\nsaber plugins list\n\nsaber plugins setup notebook",
)


def _declaration(name: str) -> PluginSettings:
    if name not in installed_plugins():
        raise ValueError(f"Plugin '{name}' is not installed. Run: saber plugins list")
    migrate_legacy_plugin_models()
    declaration = load_plugin_settings(name)
    if declaration is None:
        raise ValueError(
            f"Plugin '{name}' exposes no settings. It can be enabled or disabled."
        )
    return declaration


@plugins_app.command(name="list")
def list_plugins() -> None:
    """List installed plugins without constructing capabilities.

    Examples:
        saber plugins list
    """
    rows: list[dict[str, b.Cell]] = []
    try:
        migrate_legacy_plugin_models()
        store = PluginConfigStore()
        for name in sorted(installed_plugins()):
            saved = store.get(name)
            state = "disabled" if not saved.enabled else "defaults"
            if saved.enabled:
                try:
                    declaration = load_plugin_settings(name)
                    if declaration is not None:
                        resolve_settings(
                            name, declaration, saved.settings, load_secrets=False
                        )
                    if saved.settings:
                        state = "configured"
                except PluginSetupRequired:
                    state = f"setup required: saber plugins setup {name}"
                except ValueError:
                    state = f"invalid configuration: saber plugins show {name}"
            rows.append({"plugin": name, "state": state})
        out(
            b.table(
                rows,
                columns=[
                    b.Column("plugin", "Plugin"),
                    b.Column("state", "Configuration"),
                ],
            )
        )
    except (ValueError, OSError) as exc:
        fail_usage(str(exc))


@plugins_app.command
def show(name: str, field: str | None = None) -> None:
    """Show exposed fields, effective values, and their sources. Secrets are redacted.

    Examples:
        saber plugins show notebook
        saber plugins show notebook backend
    """
    try:
        declaration = _declaration(name)
        saved = PluginConfigStore().get(name)
        resolved = resolve_settings(name, declaration, saved.settings, validate=False)
        fields = (declaration.field(field),) if field else declaration.fields
        rows: list[dict[str, b.Cell]] = []
        for setting in fields:
            active = setting.active(resolved.values)
            secret = setting.kind == "secret"
            value = resolved.values.get(setting.name, None)
            if value is None:
                value = (
                    "unset (uses main model)" if setting.kind == "model" else "unset"
                )
            if secret:
                value = (
                    "configured"
                    if setting.name in resolved.secrets
                    else "not configured"
                )
            rows.append(
                {
                    "setting": setting.name,
                    "value": value if active else "not applicable",
                    "source": resolved.sources.get(setting.name, "default")
                    if active
                    else "inactive",
                    "saved": "hidden"
                    if secret
                    else saved.settings.get(setting.name, "not saved"),
                }
            )
        out(
            b.md(
                f"## {name}\n\nAuto-loading: {'enabled' if saved.enabled else 'disabled'}"
            )
        )
        out(
            b.table(
                rows,
                columns=[
                    b.Column("setting", "Setting"),
                    b.Column("value", "Effective value"),
                    b.Column("source", "Source"),
                    b.Column("saved", "Saved value"),
                ],
                max_rows=len(rows),
            )
        )
        if field:
            setting = declaration.field(field)
            if setting.choices:
                out(b.md(f"Choices: {', '.join(setting.choices)}"))
            if setting.help:
                out(b.md(setting.help))
            if setting.env:
                out(b.md(f"Environment override: `{setting.env}`"))
        else:
            out(b.md(f"Field help: `saber plugins show {name} FIELD`", role="muted"))
        notice = declaration.notice(resolved.values)
        if notice:
            out(b.warn(notice))
        try:
            resolve_settings(name, declaration, saved.settings, load_secrets=False)
        except ValueError as exc:
            fail_usage(str(exc))
    except (ValueError, OSError) as exc:
        fail_usage(str(exc))


async def _wizard(
    name: str,
    declaration: PluginSettings,
    values: dict[str, SettingValue],
    advanced: bool,
) -> dict[str, str] | None:
    prompter = AsyncPrompter()
    secrets: dict[str, str] = {}
    for advanced_pass in (False, True):
        if advanced_pass and not advanced:
            choice = await prompter.confirm(
                "Configure advanced settings?", default=False
            )
            if choice is None:
                return None
            if not choice:
                break
        for setting in declaration.fields:
            if setting.advanced != advanced_pass:
                continue
            resolved = resolve_settings(
                name,
                declaration,
                values,
                use_environment=False,
                validate=False,
                load_secrets=False,
            )
            if not setting.active(resolved.values):
                continue
            current = resolved.values.get(setting.name)
            if setting.help:
                out(b.md(setting.help, role="muted"))
            if setting.kind == "secret":
                answer = await prompter.secret(
                    f"{setting.label} (Enter to keep existing credentials)"
                )
                if answer is None:
                    return None
                if answer.strip():
                    secrets[setting.name] = answer.strip()
                continue
            if setting.choices:
                answer = await prompter.select(
                    setting.label, choices=setting.choices, default=current
                )
            elif setting.kind == "boolean":
                answer = await prompter.confirm(setting.label, default=bool(current))
            else:

                def validate(text: str) -> bool | str:
                    try:
                        setting.parse(text if text else None)
                        return True
                    except ValueError as exc:
                        return str(exc)

                answer = await prompter.text(
                    setting.label,
                    default=str(current) if current is not None else "",
                    validate=validate,
                )
            if answer is None:
                return None
            values[setting.name] = setting.parse(answer if answer != "" else None)
    return secrets


def _save(
    name: str,
    declaration: PluginSettings,
    values: dict[str, SettingValue],
    secrets: dict[str, str],
    *,
    enabled: bool,
    yes: bool,
) -> None:
    resolved = resolve_settings(
        name,
        declaration,
        values,
        use_environment=False,
        load_secrets=False,
    )
    notice = declaration.notice(resolved.values)
    if notice:
        out(b.warn(notice))
        if not confirm_sync(
            yes=yes,
            prompt="Save these plugin settings?",
            hint=f"saber plugins setup {name} --set FIELD=VALUE --yes",
        ):
            out(b.warn("Setup cancelled. No settings saved."))
            return
    for key, value in secrets.items():
        setting = declaration.field(key)
        if not setting.active(resolved.values):
            raise ValueError(f"{key} does not apply to the selected configuration")
        save_secret(name, setting, value)
    PluginConfigStore().save(name, SavedPlugin(enabled, values))
    out(b.success(f"Saved {name} settings. Changes apply to new CLI sessions."))
    aliases = [
        setting.env
        for setting in declaration.fields
        if setting.env and setting.active(resolved.values)
    ]
    import os

    overridden = [alias for alias in aliases if alias in os.environ]
    if overridden:
        out(
            b.warn(
                f"Environment overrides remain active: {', '.join(overridden)}. Run: saber plugins show {name}"
            )
        )


@plugins_app.command
def setup(
    name: str,
    set_values: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            name="--set", help="FIELD=VALUE; repeatable. Never pass secrets here."
        ),
    ] = None,
    secret_stdin: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Read one secret FIELD from stdin and store it securely"
        ),
    ] = None,
    advanced: bool = False,
    yes: bool = False,
) -> None:
    """Configure and enable a plugin. With --set or --secret-stdin, never prompt for fields.

    Examples:
        saber plugins setup notebook
        saber plugins setup notebook --set backend=docker --set memory_mb=4096
        saber plugins setup viz --set model=openai:gpt-5-mini --yes
        saber plugins setup sandbox --set provider=e2b --secret-stdin e2b_api_key --yes
    """
    try:
        declaration = _declaration(name)
        values = dict(PluginConfigStore().get(name).settings)
        changed: list[str] = []
        for assignment in set_values or []:
            key, separator, value = assignment.partition("=")
            if not separator:
                raise ValueError("Expected --set FIELD=VALUE")
            setting = declaration.field(key)
            if setting.kind == "secret":
                raise ValueError(
                    f"Use --secret-stdin {key} instead of --set for secrets"
                )
            values[key] = setting.parse(value)
            changed.append(key)
        secrets: dict[str, str] = {}
        if secret_stdin is not None:
            setting = declaration.field(secret_stdin)
            if setting.kind != "secret":
                raise ValueError(f"{secret_stdin} is not a secret setting")
            value = sys.stdin.read().strip()
            if not value:
                raise ValueError("Secret stdin was empty")
            secrets[secret_stdin] = value
        if set_values is None and secret_stdin is None:
            if not sys.stdin.isatty():
                raise ValueError(
                    f"Setup requires a terminal or --set. Example: saber plugins setup {name} --set FIELD=VALUE. Use 'saber plugins show {name}' for fields."
                )
            answers = asyncio.run(_wizard(name, declaration, values, advanced))
            if answers is None:
                out(b.warn("Setup cancelled. No settings saved."))
                return
            secrets = answers
        resolved = resolve_settings(
            name, declaration, values, use_environment=False, load_secrets=False
        )
        for key in changed:
            if not declaration.field(key).active(resolved.values):
                raise ValueError(f"{key} does not apply to the selected configuration")
        _save(name, declaration, values, secrets, enabled=True, yes=yes)
    except (ValueError, OSError) as exc:
        fail_usage(str(exc))


@plugins_app.command(name="set")
def set_setting(name: str, field: str, value: str, *, yes: bool = False) -> None:
    """Change one saved non-secret setting without enabling a disabled plugin.

    Examples:
        saber plugins set notebook memory_mb 16384
    """
    try:
        declaration = _declaration(name)
        setting = declaration.field(field)
        if setting.kind == "secret":
            raise ValueError(f"Use: saber plugins setup {name} --secret-stdin {field}")
        saved = PluginConfigStore().get(name)
        values = {**saved.settings, field: setting.parse(value)}
        resolved = resolve_settings(
            name, declaration, values, use_environment=False, load_secrets=False
        )
        if not setting.active(resolved.values):
            raise ValueError(f"{field} does not apply to the selected configuration")
        _save(name, declaration, values, {}, enabled=saved.enabled, yes=yes)
    except (ValueError, OSError) as exc:
        fail_usage(str(exc))


@plugins_app.command
def unset(name: str, field: str, *, yes: bool = False) -> None:
    """Remove a saved non-secret override, restoring the plugin default.

    Examples:
        saber plugins unset notebook memory_mb
    """
    try:
        declaration = _declaration(name)
        if declaration.field(field).kind == "secret":
            raise ValueError("Remove credentials through your OS credential manager")
        saved = PluginConfigStore().get(name)
        values = dict(saved.settings)
        values.pop(field, None)
        _save(name, declaration, values, {}, enabled=saved.enabled, yes=yes)
    except (ValueError, OSError) as exc:
        fail_usage(str(exc))


def _enable(name: str, enabled: bool, yes: bool = False) -> None:
    try:
        if name not in installed_plugins():
            raise ValueError(f"Plugin '{name}' is not installed")
        store = PluginConfigStore()
        saved = store.get(name)
        declaration = load_plugin_settings(name) if enabled else None
        if declaration is not None:
            _save(name, declaration, saved.settings, {}, enabled=True, yes=yes)
            return
        store.save(name, SavedPlugin(enabled, saved.settings))
        out(
            b.success(
                f"{name} {'enabled' if enabled else 'disabled'} for new CLI sessions."
            )
        )
    except (ValueError, OSError) as exc:
        fail_usage(str(exc))


@plugins_app.command
def enable(name: str, *, yes: bool = False) -> None:
    """Enable auto-loading of an installed plugin.

    Examples:
        saber plugins enable notebook
    """
    _enable(name, True, yes)


@plugins_app.command
def disable(name: str) -> None:
    """Disable auto-loading without deleting settings or credentials.

    Examples:
        saber plugins disable notebook
    """
    _enable(name, False)

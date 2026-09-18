# SQLsaber Plugins

SQLsaber plugins are pydantic-ai capabilities distributed through entry points.

## Create a plugin

1. Create a package under `plugins/<name>/` with its own `pyproject.toml`.
2. Expose a capability factory:

```toml
[project.entry-points."sqlsaber.capabilities"]
my_plugin = "my_plugin:capability"
```

```python
from collections.abc import Mapping
from typing import Any

from pydantic_ai.toolsets import FunctionToolset
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.tools.base import Tool


class MyCapability(SqlSaberCapability):
    id = "my-plugin"
    description = "Use my specialist tool."

    def __init__(self, context: PluginContext):
        self.tool = MyTool(context.registry)
        self.toolset = FunctionToolset[Any](id=self.id)
        self.toolset.add_function(self.tool.execute, name=self.tool.name)

    def get_toolset(self):
        return self.toolset

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        return {self.tool.name: self.tool}


def capability(context: PluginContext):
    return MyCapability(context)
```

Factories receive the active database registry, knowledge manager, dangerous-mode flag, and normalized tool overrides. They may return one capability, a sequence, or an empty sequence when conditionally disabled.

## Configure installed plugins

Save CLI choices once, then start `saber` normally:

```bash
saber plugins list
saber plugins setup notebook
saber plugins show notebook
saber plugins set notebook memory_mb 16384
saber plugins unset notebook memory_mb
saber plugins disable notebook
saber plugins enable notebook
```

The interactive CLI exposes the same commands through `/plugins`. Changes apply
to new sessions. Disabling a plugin preserves its settings and credentials.

For scripts, supply repeatable assignments instead of opening the wizard:

```bash
saber plugins setup notebook --set backend=docker --set memory_mb=4096
```

Remote-provider setup prints the data-transfer notice and requires confirmation.
Use `--yes` to confirm in a script. Pass secrets with `--secret-stdin FIELD`, not
`--set`, or enter them in the masked setup prompt. Secrets go to the OS credential
store, never `plugin_config.json`. A failed credential-store write is an error;
SQLsaber does not fall back to plaintext storage.

Settings live in `plugin_config.json` in SQLsaber's platform-specific user config
directory, beside `model_config.json`. Precedence is explicit run options, declared
environment variables, saved values, then plugin defaults. `show` reports sources
and saved values without revealing secrets. Inactive provider-specific settings
remain saved for switching back, but are not passed to a different provider.

## Expose settings from your plugin

Register a companion entry point with the same name as the capability:

```toml
[project.entry-points."sqlsaber.plugin_settings"]
my_plugin = "my_plugin.settings:settings"
```

Export a `sqlsaber.plugin_settings.PluginSettings` value. Its `fields` tuple is the
allowlist used by setup, editing, and inspection. Each `Setting` declares a name,
label, kind (`text`, `integer`, `number`, `boolean`, `secret`, or `model`), and optional
default, choices, environment alias, help, and advanced-field marker.

Use `when(values)` to make a field conditional on an unconditional selector such
as `provider`. Keep all fields in the declaration. SQLsaber excludes inactive
fields from prompts and runtime values. A secret can declare a shared `credential`
identity so two plugins use the same provider account without duplicating keys.
Declare at most one `kind="model"` field. Unset inherits the session main model.
Use `model_setting()` and `configured_model(values)` to round-trip that field
into the plugin's runtime config.

Supply these functions on `PluginSettings`:

- `validate(values)` validates active non-secret values through your existing
  runtime configuration type. Raise `ValueError` with the field and constraint.
- `bind(values, secrets)` constructs a configured factory accepting `PluginContext`.
  The `secrets` mapping contains active credentials keyed by setting name.
- `notice(values)` optionally returns a remote-upload, billing, or login notice
  displayed before saving.

Keep declarations and validation free of prompts, database connections, and
network calls. Import provider SDKs only when the backend is used. Pass credentials
to provider clients without modifying `os.environ`.

See the [notebook declaration](notebook/src/sqlsaber_notebook/settings.py) and
[sandbox declaration](sandbox/src/sqlsaber_sandbox/settings.py) for implementations.
Core does not maintain a provider list or import either plugin by name. Plugins
without this entry point continue to load normally and can be enabled or disabled.
SDK capability factories do not read saved CLI settings.

## Porting a legacy tool plugin

The old `sqlsaber.tools` group and global `ToolRegistry` are removed. Keep your existing `Tool.execute` and rendering methods, put the instance in a `FunctionToolset`, expose it through `display_specs`, and change the entry point to `sqlsaber.capabilities`. Model overrides should be read from `context.tool_overrides` during construction rather than `ctx.deps`.

## Install a plugin locally

```bash
uv pip install -e plugins/<name>
```

## Run plugin tests

```bash
uv run pytest plugins/<name>/tests -q
```

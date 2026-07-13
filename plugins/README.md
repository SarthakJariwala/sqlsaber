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

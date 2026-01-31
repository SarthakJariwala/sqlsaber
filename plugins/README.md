# SQLsaber Plugins

SQLsaber supports optional tools distributed as plugins via entry points.

## Create a plugin

1. Create a package under `plugins/<name>/` with its own `pyproject.toml`.
2. Expose tools via entry points under `sqlsaber.tools`:

```toml
[project.entry-points."sqlsaber.tools"]
my_tool = "my_plugin.module:MyToolClass"
```

You can also expose a factory if tool registration is conditional:

```toml
[project.entry-points."sqlsaber.tools"]
my_plugin = "my_plugin:register_tools"
```

## Install a plugin locally

```bash
uv pip install -e plugins/<name>
```

## Run plugin tests

```bash
uv run pytest plugins/<name>/tests -q
```

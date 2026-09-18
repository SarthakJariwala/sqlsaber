# Synthesized design: plugin nested-model ownership

Base: opus. Cross-judge ([Arena cross-judge nested models](bc-1f69d470-4443-51b7-a6b3-b2d943d152d0)) scored opus 23, fable 21, grok 18, sol 17. Parent score matched opus 23 / fable 21.

Fable and opus converged on one shape. A two-branch sum lives on the capability config. CLI writes it through `plugins setup`. SDK writes it by constructing that config. `tool_overrides` still wins.

## Grafts

- Keep `resolve_model(auth, full_model_str: str, …)` and `ResolvedModel` fields unchanged.
- Put `migrate_legacy_plugin_models` in `sqlsaber.config.plugins`.
- No env alias on plugin `model`. Do not reuse `SQLSABER_NOTEBOOK_MODEL`.
- Export tool-name constants (`ANALYZE_DATA`, `ANALYZE_IN_SANDBOX`, `VIZ`). No `NestedTool` brand.
- `PluginSettings.__post_init__` rejects more than one `kind="model"` field.
- A pin never inherits the main API key.
- `saber models current` lists handoff only, plus a pointer to `saber plugins list`.
- `INHERIT` reuses the parent's `ResolvedModel` handle. Rebuild from `main.model_name` and `main.api_key` if a provider model carries per-agent state.

## Rejects

- Sol's `SQLSaberOptions.plugin_models` map.
- Grok's CLI translation into `tool_overrides`.
- Fable's cross-store `models current` table.
- Opus's `resolve_model(ModelId)` ripple and same-provider key inheritance.
- Handoff as an internal plugin.

## Public usage

```bash
saber plugins setup viz --set model=openai:gpt-5-mini --yes
saber plugins unset notebook model --yes
saber models set openai:gpt-5-mini --agent handoff
```

```python
partial(notebook, config=NotebookConfig(model=pin("openai:gpt-5-mini")))
```

`tool_overrides={"analyze_data": pin("anthropic:claude-haiku-4-5")}` still wins for that tool.

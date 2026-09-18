# Plugin nested-model ownership (grounding)

Measured from the checkout at `7344a75` on `cursor/plugin-model-config-31cb`.

## What exists

Core treats nested LLM workers as named "agents" in one global list.

- `SUBAGENT_KEYS = ("handoff", "viz", "notebook")` in `src/sqlsaber/config/settings.py:17`.
- CLI `saber models set|current|reset --agent` accepts `main` plus those keys (`src/sqlsaber/cli/models.py:37`, `378+`).
- Overrides persist in `model_config.json` under `subagents` (`ModelConfigManager.set_subagent_model`).
- `Config.in_memory(subagent_models=...)` keeps only keys in `SUBAGENT_KEYS` (`settings.py:279-282`).
- `PluginContext.resolve_subagent_model(name, tool_name=)` precedence (`capabilities/plugins.py:44-72`):
  1. `tool_overrides[tool_name]` (`ModelOverides`)
  2. `config.model.get_subagent_model(name)`
  3. main model
- API key follows the same split. A subagent override does not reuse the main key.

## Who consumes it

- Handoff (`agents/handoff_agent.py:42-45`): constructor override, else `get_subagent_model("handoff")`, else main. Not a capability. Invoked from `sdk/client.py` `draft_handoff` and CLI `/handoff`.
- Viz (`plugins/viz/src/sqlsaber_viz/spec_agent.py:33-37`): constructor override from `tool.model_overide`, else `get_subagent_model("viz")`, else main. Does not call `resolve_subagent_model`. Capability copies `context.tool_overrides` onto the tool (`capability.py:22`).
- Notebook (`plugins/notebook/.../capability.py:180-183`): `resolve_subagent_model("notebook", tool_name=self.name)` with tool name `analyze_data`.
- Sandbox (`plugins/sandbox/.../tools.py:165-167`): `resolve_subagent_model("sandbox", tool_name=self.name)`. `"sandbox"` is not in `SUBAGENT_KEYS`, so CLI `--agent` and `Config.in_memory(subagent_models=)` cannot set it. Only `tool_overrides` can.

## Plugin settings (CLI, not SDK)

PR #285 (`11b35fe`, merged 2026-09-17) added plugin-owned CLI settings. PR body quote:

> "This adds saved configuration without teaching core about individual plugins or providers."

- Entry group `sqlsaber.plugin_settings`. Fields are an allowlist (`plugin_settings.py`).
- Persist in `plugin_config.json`. Secrets in keyring.
- `saber plugins setup|show|set|unset|enable|disable` (`cli/plugins.py`).
- CLI session bind: `configured_capabilities()` resolves settings and calls `declaration.bind(...)` (`cli/session.py:28-65`).
- Docstring on `plugin_settings.py`: SDK factories do not read this store.
- Notebook and sandbox declare settings. Neither declares a model field (`plugins/notebook/src/sqlsaber_notebook/settings.py`, `plugins/sandbox/src/sqlsaber_sandbox/settings.py`).
- Viz has no `plugin_settings` entry point (`plugins/viz/pyproject.toml`). `plugins setup viz` fails with "exposes no settings" (verify skill `features/plugins.md`).
- `NotebookConfig` / `SandboxConfig` have no model field. Nested model is resolved from `PluginContext`, not from those configs.

## SDK override today

Documented: `SQLSaberOptions.tool_overrides` mapping tool name to `ModelOverides` (`docs/src/content/docs/sdk/advanced.mdx`). Example uses `"viz"`.

Also: `SQLSaberOptions.capabilities` with a bound factory, e.g. `partial(capability, config=NotebookConfig(...))`. That config cannot currently name a model.

`Config.in_memory(subagent_models=)` exists and is tested (`tests/test_config/test_settings.py:227-238`) but is not in the SDK credentials/models docs.

CLI auto-loads installed plugins. An embedded `SQLSaber` session does not (`docs/.../guides/plugins.mdx`).

## History (source control)

From [Why nested models are core agents](bc-a1396a8c-312d-53a6-91a9-e51852ad93e1). Dates and quotes are from git/`gh`, not inference.

- 2026-01-28 #105 (`592117c`): `/handoff` as a dedicated no-tool core agent with constructor `model_name` override.
- 2026-02-04 #109: viz plugin.
- 2026-02-05 #117: viz API `viz_model_name` for a cheaper/faster spec agent. Same day #118 (`ac31f8d`): `SUBAGENT_KEYS = ("handoff", "viz")` and `saber models set --agent`. PR body empty. Docs: override subagents like handoff and viz without changing the main model.
- 2026-02-09 #122: `tool_overrides` replaced viz-specific API kwargs.
- 2026-03-11 #139: `Config.in_memory` filters constructor keys to `SUBAGENT_KEYS`.
- 2026-07-14 #188: capabilities. `PluginContext` had no model-store helper yet.
- 2026-07-21 #199 (`c82590f`): notebook appended to `SUBAGENT_KEYS`. Introduced `resolve_subagent_model`. README told users `saber models set --agent notebook`.
- 2026-09-16 #282: sandbox calls `resolve_subagent_model("sandbox")`. Never added to `SUBAGENT_KEYS`.
- 2026-09-17 #285 (`11b35fe`): plugin settings "without teaching core about individual plugins or providers." Did not move model overrides. Settings live beside `model_config.json`.

No GitHub issue drove this. All listed PRs have empty `closingIssuesReferences`.

Unsearched: Amp threads linked from #285 and #282, Claude Code session on #117.

## Handoff vs a plugin

Handoff is a no-tool summarizer for `/handoff`, always present in vanilla sqlsaber, not discovered via entry points, not enable/disable, not provider-configured. It shares the global subagent table with plugins by history, not by shape.

Making it an internal plugin is a different ownership change. Leave it on `saber models set --agent handoff`.

## Premise that is already failing

"Nested models are a closed set of core agents." Sandbox already has a nested model outside that set. Plugin settings already exist as the CLI configuration path that must not teach core plugin names. `--agent viz|notebook` teaches core those names.

## Constraints a redesign must honor

1. CLI users configure plugins through `saber plugins setup`, including nested model.
2. SDK still overrides a plugin model without the CLI store. Prefer capability configuration (`NotebookConfig` / `SandboxConfig` / viz equivalent) because that is how plugins are registered.
3. `tool_overrides` is the documented session-level SDK override. Keep or explicitly migrate.
4. Core must not grow a plugin-name list (#285).
5. Handoff stays a core agent for this change.
6. Saved `model_config.json` `subagents.viz` / `subagents.notebook` exist in the wild.
7. `plugins setup` already has `--set FIELD=VALUE` and `--yes` (cli-for-agents). New model field must work non-interactively.
8. CLI startup must stay fast. Do not import pydantic_ai at plugin_settings declaration time.

## Explorer-confirmed details

Four how explorers agreed. No factual contradictions.

- Tool-override keys are tool names (`viz`, `analyze_data`, `analyze_in_sandbox`), not `--agent` names. `tool_overrides={"notebook": ...}` does not hit notebook.
- `PluginSettings.bind` runs in `configured_capabilities()` before `PluginContext` exists. Bind can close over a config object. It cannot call `resolve_subagent_model`.
- Viz and handoff construct a fresh file-backed `Config()`. They ignore `PluginContext.config` and session `SQLSaberOptions.model_name` / `settings` unless those values are also on disk.
- `discover_capabilities` is test-only. Production CLI uses `configured_capabilities()`. SDK uses `options.capabilities`.
- `/plugins` does not reload the live session. `/models set|reset` does call `reload_model_settings()`.
- Standalone `sqlsaber-notebook` uses `--model` / `SQLSABER_NOTEBOOK_MODEL`, not plugin settings.
- `NotebookConfig` docstring says it is budgets, not model-preview budgets.
- Handoff is a session verb (`/handoff` → `draft_handoff` → new thread). It is not a main-agent tool. A capability plugin cannot replace that without new slash-command extension points.

## Verify surfaces

- `.agents/skills/verify-sqlsaber/features/plugins.md`
- `.agents/skills/verify-sqlsaber/features/model-configuration.md` (`models-set-agent` currently includes viz and notebook)
- `scripts/verify_plugin_settings.py`

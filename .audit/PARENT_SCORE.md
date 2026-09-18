# Parent score (before cross-judge)

Scored 0-3 against TASK.md rubric. Parent model is grok-4.6.

## Scores

| criterion | fable | sol | grok | opus |
| --- | --- | --- | --- | --- |
| 1 CLI plugins setup only | 3 | 3 | 3 | 3 |
| 2 SDK is capability construction | 3 | 1 | 2 | 3 |
| 3 no core plugin-name list | 3 | 3 | 3 | 3 |
| 4 handoff stays --agent | 3 | 3 | 3 | 3 |
| 5 one resolve path, viz uses it | 2 | 2 | 2 | 3 |
| 6 inherit vs pin as a sum | 3 | 3 | 2 | 3 |
| 7 caller does not coordinate two stores | 2 | 1 | 2 | 3 |
| 8 design red flags | 2 | 2 | 2 | 2 |
| total | 21 | 18 | 19 | 23 |

## Criterion notes

1. All four expose `plugins setup|set|unset|show --set model= --yes`. Tie.

2. Fable and opus put the pin on the capability config the factory already takes. Sol adds `SQLSaberOptions.plugin_models` and keeps budget configs model-free, so the SDK caller names the plugin twice. Grok documents `tool_overrides` as the primary SDK path and folds capability config into that map, so capability construction exists but is not the one path.

3. All four shrink `SUBAGENT_KEYS` to handoff and discover plugin names from entry points. Tie.

4. All four leave handoff on `models --agent handoff` and refuse to make it a plugin. Tie.

5. Opus is the only package that puts viz AND handoff on `resolve_nested_model`. The others leave handoff on a second lookup, and fable leaves `HandoffAgent`'s fresh `Config()` alone.

6. Fable, sol, and opus use an inherit/pin sum. Grok uses presence vs absence of `ModelOverride`, which is a sum in spirit but not a type a plugin can hold.

7. Opus is the only one whose CLI user never joins stores and whose SDK user never joins two option fields. Fable's `models current` reads plugin settings from a models command. Grok's SDK can set both `tool_overrides` and `config.nested_model`. Sol requires `capabilities=` plus `plugin_models=`.

8. All four trip at least one red flag. Opus's new `plugin_model_migration.py` is temporal decomposition. Fable's models-current table leaks two stores into one command. Grok's `NestedTool` plus `with_nested_model` plus CLI translation is extra ceremony. Sol's host map is a second SDK knob.

## Hard-constraint misses

Sol misses "SDK via capability configuration" as stated in the user request and rubric 2. Grok keeps that path as a fold, which still works. None add sandbox to `SUBAGENT_KEYS`. None make handoff a plugin. None read `plugin_config.json` from SDK factories.

## Tentative base

Opus. Same whole shape as fable (capability config owns the nested model), with a tighter sum (`ModelId` + `Pinned`), a required `model=` on the resolver so plugins cannot forget to thread config, and handoff on the same resolve function without becoming a plugin.

Fable is the smaller sibling of that shape. If the judge prefers fable for API size, that is still the same pick family.

Sol is rejected as a base because the SDK caller coordinates two fields for one plugin.

Grok is rejected as a base because it makes `tool_overrides` the only runtime map, which forces CLI translation and a plugin-to-tool binding table (`NestedModelBinding`) that every plugin must declare.

## Tentative grafts (pending judge)

From fable into opus: keep `adopt_legacy_subagent_models` in `config/plugins.py`. Do not add a migration module. Do not make `saber models current` a cross-store dashboard.

From grok: do not alias plugin `model` to `SQLSABER_NOTEBOOK_MODEL`. Warn when a `tool_overrides` key matches no loaded tool. Do not take `NestedTool` or CLI translation into `tool_overrides`.

From sol: nothing on the host map. Keep budget-field docs honest that `model` is a nested-agent choice sitting beside budgets because `bind` already builds that object.

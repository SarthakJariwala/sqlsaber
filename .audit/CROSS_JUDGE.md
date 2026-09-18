# Cross-judge: plugin nested-model ownership

Judge read all four packages end to end, `TASK.md`, `design-red-flags.md`, `grounding.md`, and `why-synthesis.md`. Every factual claim below marked *measured* was checked against the checkout at `7344a75` (`src/sqlsaber/**`, `plugins/**`). Claims marked *inferred* follow from the design text. Nothing was implemented.

## Score table

| Criterion | fable | sol | grok | opus |
| --- | --- | --- | --- | --- |
| 1. CLI path is only `plugins setup/set/unset/show` with `--set` / `--yes` | 3 | 3 | 3 | 3 |
| 2. SDK path is capability construction; no `plugin_config.json` | 3 | 1 | 3 | 3 |
| 3. Core has no viz/notebook/sandbox name list | 3 | 3 | 3 | 3 |
| 4. Handoff stays `models --agent handoff` | 3 | 3 | 3 | 3 |
| 5. One resolve path; viz uses it | 2 | 2 | 2 | 3 |
| 6. Inherit vs override is a sum type, not a bag of optionals | 2 | 3 | 2 | 3 |
| 7. Interface depth; caller does not coordinate two stores | 3 | 1 | 1 | 3 |
| 8. Red-flag screen | 2 | 1 | 1 | 2 |
| **Total** | **21** | **17** | **18** | **23** |

## Evidence by criterion

**1. CLI path.**
- fable 3. `plugins setup NAME --set model=… --yes`, `set`, `unset`, `show` only; adds an env alias per plugin, which is the existing flag > env > saved > default precedence every plugin field already has (*measured*, `config/plugins.py` `resolve_settings`).
- sol 3. Same four verbs; uses `--yes` on `set`/`unset`, which those commands accept today (*measured*, `cli/plugins.py` `set_setting`/`unset` have `yes: bool`).
- grok 3. Same four verbs; declines any env alias, so the only write path is `--set`.
- opus 3. Same four verbs; `--agent viz` rejection looks the plugin up from entry points, not a list.

**2. SDK path.**
- fable 3. `partial(notebook, config=NotebookConfig(model=PinnedModel(…)))`; nothing reads `plugin_config.json`.
- sol 1. The SDK surface is a new `SQLSaberOptions.plugin_models` map keyed by plugin name; `NotebookConfig`/`SandboxConfig` stay model-free by design, so the override is not capability configuration and no id is baked into config (TASK.md hard constraint reads "via capability configuration" and "bake an id into config").
- grok 3. `NotebookConfig(nested_model=ModelOverride(…))` is supported, though the package leads with `tool_overrides={ANALYZE_DATA: …}` and folds config into that map second.
- opus 3. `NotebookConfig(model=pin(…))`; `PluginContext` loses `config: Config`, so a capability cannot reach either store (*measured*: no plugin reads `context.config` today except viz's own fresh `Config()`, so the removal breaks nothing in-repo).

**3. No plugin-name list in core.**
- fable 3. `SUBAGENT_KEYS = ("handoff",)`; redirect text, `models current` rows, and adoption all iterate entry points at runtime.
- sol 3. `HANDOFF_AGENT_NAME` only; the plugin passes its own `PLUGIN_NAME` to the resolver.
- grok 3. `SUBAGENT_KEYS = ("handoff",)`; legacy `--agent` names live on the plugin's `NestedModelBinding`.
- opus 3. `CoreAgent` StrEnum with one member; migration candidates come from `sqlsaber.capabilities` entry points.

**4. Handoff.**
- fable 3. `HandoffAgent` untouched; `AGENT_CHOICES` shrinks to `("main", "handoff")`.
- sol 3. `get_handoff_model`/`set_handoff_model` over `subagents.handoff`.
- grok 3. Unchanged verb; `HandoffAgent` now takes the session `Config` instead of `Config()`.
- opus 3. Unchanged verb; `SQLSaberAgent.resolve_core_agent_model(CoreAgent.HANDOFF)`.

**5. One resolve path.**
- fable 2. Notebook, sandbox, and viz all call `resolve_subagent_model(tool_name, *, model)`, but handoff keeps its own chain and its own fresh `Config()` (fable defers this explicitly).
- sol 2. Same three plugins on one resolver; handoff "continues to read `model_config.json`" on a separate path.
- grok 2. Viz reads `resolve_tool_model`, but precedence is split between the `with_nested_model` fold (session wins) and the read, and handoff keeps a second chain inside `HandoffAgent`.
- opus 3. `resolve_nested_model(choice, main=, auth=)` serves viz, notebook, sandbox, and handoff; precedence is the pure `most_specific` fold.

**6. Sum type.**
- fable 2. `SubagentModel = InheritMainModel | PinnedModel` is a proper sum, but `PluginContext.tool_overrides` stays `Mapping[str, ModelOverides]` where `model_name: str | None`, and the resolver's `match` relies on a docstring ("normalized; every entry has model_name") to exclude the `None` arm (*measured*: `overrides.py` `ModelOverides` is the optional bag today).
- sol 3. `PluginModelSelection = InheritMainModel | ExplicitModel`; `normalize_tool_overides` returns `Mapping[str, ExplicitModel]`, so the session tier is also required-model.
- grok 2. `ModelOverride.model_name` is required everywhere, so no illegal state exists, but inherit is spelled `None`, an anonymous branch a reader must know by convention; the `NestedTool` brand is erased to `str` inside.
- opus 3. `NestedModel = Inherit | Pinned`, `Pinned.id: ModelId` with canonical provider, `tool_overrides: Mapping[str, Pinned]` internally, codex-key rejection at construction.

**7. Interface depth.**
- fable 3. Plugin author holds one config field and makes one call, `resolve_subagent_model(self.name, model=self.config.model)`; only `models current` and adoption read both stores, and they are read-only or one-shot.
- sol 1. The SDK caller supplies `capabilities=[partial(notebook, config=…)]` *and* `plugin_models={"notebook": …}` and must keep the string key aligned with the plugin's hardcoded `PLUGIN_NAME`; a host constructing `Notebook(context, config=)` directly has no way in except through the context map.
- grok 1. Plugin author exports a `NestedTool`, declares a `NestedModelBinding`, bakes `nested_model`, folds with `with_nested_model` in the factory *and again* in `update_context`, then resolves; grok's own text warns that the second fold is not "a dead assignment", which is the footgun.
- opus 3. One field, one call, `resolve_subagent_model(self._config.model, tool=self.name)`; the plugin store is unreachable from `PluginContext`.

**8. Red flags.**
- fable 2. `_nested_model_rows()` in `cli/models.py` pulls `resolve_settings` and an entry-point scan into the models command, so two modules now depend on the plugin-settings layout (information leakage); the 3-tuple return and the optional-bag `ModelOverides` are kept by choice.
- sol 1. `ResolvedSettings` splits into `runtime_values` plus `model` plus a derived `values` property (one representation, two views); `ConfiguredCapabilities` and `exposed_fields` are thin pairings; `compose_plugin_models` puts a legacy overlay inside the agent; `set_handoff_model` is a pass-through specialization of `set_subagent_model`.
- grok 1. The CLI *translates* the saved id into `tool_overrides` while `bind` also bakes it into `config.nested_model`, so in the CLI path the same decision enters the one channel twice (leakage); `NestedModelBinding` exists only to serve that translator (shallow); `with_nested_model` yields a per-capability `PluginContext` variant (hidden state); `assemble_cli_plugins` is adopt → bind → translate (temporal).
- opus 2. `plugin_model_migration.py` is a one-function module fable and grok fold into `config/plugins.py`; `resolve_model(auth, id: ModelId)` and `ResolvedModel.id` change the main-agent path for a benefit only the key-inheritance change needs (*measured*: four production callers of `resolve_model` plus tests); `model_setting()`/`configured_model()` are two-line helpers.

## Hard-constraint violations (TASK.md)

- **sol** violates "SDK override without the CLI store, via capability configuration" and "Bake an id into config". Its SDK path is `SQLSaberOptions.plugin_models`; budget configs carry no model by design. The package argues this is a feature. The task says otherwise.
- **fable, grok, opus**: none against TASK.md.

Soft deviations worth naming:

- **opus** changes the API-key branch (a pin on the parent's provider inherits the parent's resolved key). `why-synthesis.md` lists "precedence and API-key branch" under *Preserve*. Opus names this in tradeoffs.
- **fable** aliases the notebook field to `SQLSABER_NOTEBOOK_MODEL` and calls it "already the standalone alias". *Measured*: that variable is the standalone `sqlsaber-notebook` process's main model (`plugins/notebook/.../cli.py:73`). The two meanings nearly coincide, and the standalone also requires `provider:model` (`_provider_from_model`, `cli.py:155`), so opus's bare-id fear does not materialize, but a value set for standalone would silently pin the nested model in every `saber` session. Grok's refusal is the safer default.
- **grok** makes `ModelOverides = ModelOverride` with `model_name` required, so `ModelOverides()` and `ModelOverides(api_key=…)` now raise at construction instead of at `normalize`. Small documented-type break, not silent.
- **opus** claims a `ty`-clean prototype of `nested_model.py`. No prototype is in its output directory. Unverified.

## Per-candidate critique

**fable.** Strengths: the smallest diff that fully solves the problem; the shape (sum type on capability config, resolver takes the config value plus tool name, `kind="model"`, adopt-then-delete, `SUBAGENT_KEYS` → handoff only, `VizConfig`, `SpecAgent(model)`) is the same shape opus reached independently; one `validate_model_id` shared with `saber models set`; adoption lives beside `PluginConfigStore`; `Config.in_memory(subagent_models={"notebook"})` raises with the replacement. Failures: leaves handoff as a standing exception to "one resolve path" with a fresh `Config()`; keeps the `ModelOverides` optional bag in the session tier and papers over it with a docstring, which is the exact "comment explaining when the combination is valid" smell; keeps `config: Config` on `PluginContext`, so the invariant "capabilities do not read the model table" is by discipline, not by type. Red flags: `models current` becomes a cross-store view in `cli/models.py`; the env alias reuse is justified loosely.

**sol.** Strengths: the cleanest internal type story of the four (`ExplicitModel` everywhere, string shorthand only at the boundary); a structural at-most-one-model check in `PluginSettings.__post_init__`; a correct write-first idempotent adoption; a clear "resolver has no legacy branch" stance. Failures: the SDK path is a host-owned `plugin_models` map, which violates the capability-configuration hard constraint and makes the SDK caller align two options fields by a string that the plugin also hardcodes; a host that constructs `Notebook(context, config=)` directly has no override path; `Config.in_memory` compatibility routing must also stop `InMemoryModelConfigManager` filtering on `SUBAGENT_KEYS` (*measured*, `settings.py:279-282`), which the package does not mention. Red flags: `ResolvedSettings` dual view, `ConfiguredCapabilities` pairing, `compose_plugin_models` overlay inside the agent, handoff-specific getters over the generic API.

**grok.** Strengths: the sharpest diagnosis of the tool-name versus agent-name trap and the only candidate that confronts `tool_overrides={"notebook": …}` at runtime; plugin-exported tool-name constants as a single source for `Tool.name`; refuses the env-alias collision with a reason; fixes `HandoffAgent`'s fresh `Config()`; keeps adoption in `config/plugins.py`. Failures: the assigned premise (one runtime channel = `tool_overrides`) forces a CLI translation layer, and the package then *also* bakes the id into `config.nested_model` and folds it back into the same map, so the CLI path carries one decision in two representations; the plugin author must re-fold in `update_context` or lose the config-level pin on every rebuild (*measured*: `_build_agent` calls `update_context` with a fresh context on each rebuild, `pydantic_ai_agent.py:206-209`); `reject_unknown_tool_overrides` as a hard error breaks any SDK caller sharing one overrides dict across a plugin subset, which grok itself lists as open. Red flags: `NestedModelBinding` is shallow, `with_nested_model` creates per-capability context variants, `assemble_cli_plugins` is temporal, `configured_capabilities()` becomes a one-line pass-through.

**opus.** Strengths: `NestedModel = Inherit | Pinned` with a branded `ModelId` makes both illegal states (key without model, unvalidated provider) unrepresentable; `PluginContext` drops `config`, making the model table structurally unreachable from a capability; handoff joins the same `resolve_nested_model`, so "one resolve path" is literally true; `CoreAgent` is the single source for `AGENT_CHOICES`, the in-memory filter, and the migration's exclusion; `most_specific` makes precedence a pure fold; `INHERIT` reuses the parent's built model so inheriting children stop re-running auth per tool call; migration candidates come from entry points and uninstalled plugins' keys are left alone. Failures: scope grows past ownership. `resolve_model` takes `ModelId` and `ResolvedModel` changes shape, rippling into the main agent, handoff, and tests; the same-provider key inheritance changes the #122/#199 credential contract; sharing one `Model` instance across parent and children is flagged but unverified for codex. Red flags: a separate one-function migration module; two two-line helpers in `plugin_settings.py`; the `ty` prototype claim has no artifact.

## Recommended base: opus

Fable and opus converged on the same shape. Opus is the base because it has the cleaner boundary, and the user's tie-break names that first. Where fable relies on discipline, opus relies on types:

- A capability cannot read the core model table because `PluginContext` has no `config`. Fable keeps `config` with a comment.
- A future maintainer cannot add a third nested-model state or a key without a model. `Pinned` requires `id: ModelId`, and `ModelId` validates the provider at construction. Fable's session tier still admits `ModelOverides(model_name=None)`.
- Adding a second core agent is one `CoreAgent` member; `AGENT_CHOICES`, the in-memory filter, and the migration exclusion all derive from it. Fable keeps a tuple that invites an append.
- Adding a precedence layer is one more argument to `most_specific`, not a new `or` term.
- Adding a fourth plugin with a nested model touches only that plugin: `model_setting()` in its settings, `model: NestedModel = INHERIT` on its config, one `resolve_subagent_model(self._config.model, tool=self.name)` call. Core is untouched and the migration discovers it through entry points.
- Handoff is not a standing exception. With fable, "one resolver, except handoff" is the precedent the next core agent copies.

Opus's totals lead by two points, and the two points are on the criteria (5 and 6) that protect invariants rather than on surface. The price is diff size, which the grafts below cut back to roughly fable's footprint without giving up the boundary.

## Graft from losers

**From fable.**
- Keep `resolve_model(auth, full_model_str: str, …)` and `ResolvedModel` unchanged. `resolve_nested_model` calls `resolve_model(auth, str(choice.id), api_key_override=choice.api_key)`. This removes the ripple into `pydantic_ai_agent.py`, `handoff_agent.py`, and `tests/test_agents/test_model_factory.py`, and drops the only reason for the contract change below. Fable's stated restraint ("the tuple is not what is broken") applies to `ResolvedModel` too.
- Put `migrate_legacy_plugin_models` in `config/plugins.py` beside `PluginConfigStore` instead of a new `plugin_model_migration.py`. Same decision ownership, one fewer file.

**From grok.**
- No env alias on the `model` field in this change. Do not reuse `SQLSABER_NOTEBOOK_MODEL`. If an alias is wanted later, use a distinct name.
- Export the tool-name constant from each plugin package (`ANALYZE_DATA = "analyze_data"`, `ANALYZE_IN_SANDBOX`, `VIZ`) and have `Tool.name` return it, so SDK users writing `tool_overrides` have one documented source. Take the constants, not the `NestedTool` brand or the runtime rejection.

**From sol.**
- `PluginSettings.__post_init__` rejects more than one `kind="model"` field. This gives opus's `model_setting()` convention a structural guarantee the migration can rely on, without sol's separate `model` slot or the `resolve_settings` rewrite that slot requires.

## Reject from losers

- **sol's `SQLSaberOptions.plugin_models`.** Violates the capability-configuration hard constraint; makes the SDK caller coordinate two options fields by a string key; forces the plugin to hardcode its entry-point name a second time; leaves direct `Notebook(context, config=)` construction with no override path.
- **sol's `ResolvedSettings` split and `PluginSettings.model` slot.** One value in two views, and every site in `cli/plugins.py` that iterates `declaration.fields` (`show`, `_wizard`, `_save`; *measured*) must switch to `exposed_fields`.
- **sol's `compose_plugin_models` in-agent legacy overlay.** A compatibility branch inside the agent for `Config.in_memory(subagent_models=)`. Opus's raising alias is louder and shorter.
- **grok's CLI translation into `tool_overrides` and `with_nested_model` fold.** The same id flows into one map two ways in the CLI path; per-capability context variants add hidden state; the `update_context` re-fold is a footgun grok documents rather than removes. Opus's resolver reads the config value directly and needs no fold.
- **grok's `NestedModelBinding`.** Its only consumer is the translator being rejected.
- **grok's `reject_unknown_tool_overrides` as a hard error.** Breaks SDK callers registering a plugin subset. Keep opus's warning.
- **fable's `_nested_model_rows()` cross-store table in `models current`.** Couples `cli/models.py` to plugin-settings resolution and entry-point loading. Keep opus's one-line pointer to `saber plugins list`.
- **fable's retained `ModelOverides` in `PluginContext.tool_overrides`.** The docstring-enforced "every entry has model_name" is the smell criterion 6 exists to catch. Parse to `Pinned` at the boundary as opus does.
- **opus's same-provider API-key inheritance.** Changes a contract `why-synthesis.md` lists under *Preserve*. Keep "a pin never inherits the main key" from fable, grok, and sol. Revisit in its own PR if wanted.
- **opus's `resolve_model(ModelId)` signature change.** Scope creep whose only justification is the rejected inheritance change. See graft from fable.
- **opus's shared `Model` instance between parent and inheriting children.** Not a reject, a gate. Verify pydantic-ai `Model` objects (including `OpenAICodexModel` with a credential source) carry no per-agent state before shipping; if any do, `INHERIT` rebuilds from `main.model_name` and `main.api_key`.

## Convergence

Two of four converged. Fable and opus, from different models and different assigned directions, produced the same shape: a two-branch sum type held by the capability's config object, one `PluginContext` resolver that takes the config value and the tool name, `Setting(kind="model")` validated at `--set` time with scalar JSON on disk, adopt-then-delete migration, `SUBAGENT_KEYS` reduced to handoff, a new `VizConfig` and viz settings entry point, `SpecAgent(model)`. That agreement is the high-signal result of this arena.

Grok and sol each diverged along their assigned direction, and each package's red flags concentrate exactly where it diverged. Grok's one-channel premise produced the translation layer and the double fold. Sol's model-free-budgets premise produced the parallel map and the hard-constraint violation. Both packages are internally coherent and both name their tradeoffs honestly. Neither shape survives the rubric.

Handoff was the one place all four agreed on the *verb* and split on the *mechanism*. Fable and sol leave it alone, grok gives it the session `Config`, opus routes it through the shared resolver. Opus's answer is the only one that leaves no standing exception for the next core agent to copy.

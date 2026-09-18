# Why nested models sit in core

Synthesized by bc-211de350-d76e-5e07-a360-bcd76eab856e. Confidence language is the synthesizer's; do not flatten.

## The Question

Why does SQLsaber configure nested plugin models (viz, notebook, and later sandbox) through a core `SUBAGENT_KEYS` list and `saber models set --agent`, instead of through plugin/capability configuration? Why is handoff in that same list? Was the global subagent table a day-one design, or did plugin settings ([#285](https://github.com/SarthakJariwala/sqlsaber/pull/285)) arrive later and leave model overrides behind?

The user's candidate hypothesis: plugin settings are the right owner now; the global table is leftover; handoff being in the list does not mean handoff should become a plugin.

## The Code in Question

- `src/sqlsaber/config/settings.py`: `SUBAGENT_KEYS: tuple[str, ...] = ("handoff", "viz", "notebook")` at line 17; `ModelConfigManager.get_subagent_model` / `set_subagent_model` / `get_subagent_models` persist a `subagents` dict inside `model_config.json`. The file-backed setter accepts any string key. `InMemoryModelConfigManager.__init__` drops any key not in `SUBAGENT_KEYS`.
- `src/sqlsaber/cli/models.py`: `AGENT_CHOICES = ("main", *SUBAGENT_KEYS)`; `_normalize_agent` rejects anything else; `saber models set|current|reset --agent` read and write the table; `saber models current` renders a "Subagent overrides" table iterating `SUBAGENT_KEYS`.
- `src/sqlsaber/capabilities/plugins.py`: `PluginContext.resolve_subagent_model(name, *, tool_name)` resolves `tool_overrides[tool_name]` → `config.model.get_subagent_model(name)` → `main_model_name`.
- `src/sqlsaber/plugin_settings.py`: SDK factories do not read this store; the CLI binds resolved settings before constructing SQLSaberOptions.
- `src/sqlsaber/cli/session.py`: `configured_capabilities()` loads plugin settings, calls `declaration.bind(...)`, never touches `set_subagent_model`.
- `src/sqlsaber/agents/handoff_agent.py`: core no-tool `HandoffAgent`.
- Call sites: notebook `resolve_subagent_model("notebook", tool_name=self.name)`; sandbox `resolve_subagent_model("sandbox", tool_name=self.name)`.

## What We Found

**Direct — the table's origin and initial contents.** `SUBAGENT_KEYS`, `get/set_subagent_model`, and `--agent` were introduced together in `ac31f8d`, [#118](https://github.com/SarthakJariwala/sqlsaber/pull/118) "feat: allow overriding subagent model configuration via cli", merged 2026-02-05T23:43Z. The initial value was `("handoff", "viz")`. The PR body is empty. The same commit added to `docs/guides/models.mdx`: "You can override the model used by subagents like handoff and viz without changing the main model."

**Direct — handoff predates the table and is not a plugin.** [#105](https://github.com/SarthakJariwala/sqlsaber/pull/105) / `592117c` added `HandoffAgent` as a dedicated agent with no tools. It lives in `src/sqlsaber/agents/`, has no entry point, and constructs `Config()` directly.

**Direct — the viz model override started as an API concern with a stated motive.** [#117](https://github.com/SarthakJariwala/sqlsaber/pull/117) added `viz_model_name` "allowing users to specify a different (potentially cheaper or faster) model for chart generation while using a different model for the main SQL agent." #118 then gave the CLI a persisted equivalent for both `viz` and `handoff`.

**Direct — the override mechanisms were generalized before capabilities existed.** [#122](https://github.com/SarthakJariwala/sqlsaber/pull/122) replaced viz-specific kwargs with `tool_overrides`. [#139](https://github.com/SarthakJariwala/sqlsaber/pull/139) added `Config.in_memory(subagent_models=...)` filtering to `SUBAGENT_KEYS`.

**Direct — capabilities arrived without a model helper; notebook added one and joined the table.** [#188](https://github.com/SarthakJariwala/sqlsaber/pull/188) `PluginContext` had four fields and no model resolver. [#199](https://github.com/SarthakJariwala/sqlsaber/pull/199) changed the tuple to `("handoff", "viz", "notebook")`, added `resolve_subagent_model`, and documented `saber models set --agent notebook`.

**Direct — sandbox uses the resolver but never joined the table.** [#282](https://github.com/SarthakJariwala/sqlsaber/pull/282) calls `resolve_subagent_model("sandbox", ...)`. No commit has ever added `sandbox` to `SUBAGENT_KEYS`.

**Direct — plugin settings arrived ~7 months after the table and did not include a model field.** [#285](https://github.com/SarthakJariwala/sqlsaber/pull/285) "This adds saved configuration without teaching core about individual plugins or providers." File list does not include `settings.py` SUBAGENT_KEYS or `cli/models.py`. Settings modules have no model field. README: settings live "beside `model_config.json`."

**Direct — current docs present both mechanisms side by side.** Models guide: override handoff, viz, or notebook. Plugins guide tells users `saber models set --agent notebook` for the model and env for the backend.

**Direct — no issue drove any of this.** Seven GitHub issues total; none about this. Single author on every cited PR: SarthakJariwala.

## What We Can Reasonably Infer

**The table predates any alternative owner; this is a chronology fact, not an inference.** On 2026-02-05 there was no `PluginContext`, no `plugin_settings`, and the viz plugin was one day old. The evidence points toward #118 putting subagent overrides in `model_config.json` because it was the only persisted model configuration in existence, not as a rejection of a plugin-owned alternative. The empty PR body means the author did not write this down.

**"Subagent" in this codebase is a role, not a plugin category.** The list has always mixed core and plugin agents. Handoff's membership does not imply handoff is or should be a plugin.

**#199 extended the existing table rather than inventing a plugin-owned model surface, and #285 left it there.** The #285 body's stated target is settings that currently depend on environment variables and runtime defaults. The evidence suggests model selection was outside #285's stated scope rather than a considered-and-rejected candidate, but the record does not say which.

**Sandbox's omission from the table is more likely accretion than design, though this is thinly supported.** Nothing written distinguishes oversight during a large PR from deferral knowing #285 was landing the next day.

**#285's principle and `SUBAGENT_KEYS` are in tension, but #285 did not claim to resolve it.** Whether the author considers this a violation to fix or an acceptable exception for the `saber models` surface is not stated anywhere.

## Competing Hypotheses

**H1 (user's candidate): plugin settings are the right owner now; the table is leftover.** For: chronology; #285 principle; three coexisting override mechanisms; sandbox never joining. Against: `resolve_subagent_model` was purpose-built in #199 to read the global table; the same author reworked both plugins' settings in #285 without moving models. Handoff cannot live in plugin settings, so some core table survives under H1.

**H2: model choice is a core `saber models` concern, deliberately separate from plugin runtime settings.** For: `saber models current` table; #285 scoped itself to env-var/runtime settings; SDK must not read the plugin store. Against: no author statement asserts this; `SUBAGENT_KEYS` hardcoding plugin names sits awkwardly with #285; sandbox's gap is unexplained.

**H3: accretion without a design decision.** For: empty bodies on #118/#122; #199 appended; #282 called the resolver without joining. Against: #199 listed "model configuration" as a deliberate integration item; three-tier precedence is a coherent contract.

## What We Don't Know

- Why #118 bundled handoff and viz under one flag.
- Whether #285 considered and rejected a `model` setting. Amp threads were not searched.
- Why sandbox was never added to `SUBAGENT_KEYS`.
- Whether #122 hid a design intent for `tool_overrides` versus the subagent table.
- The Claude Code session linked from #117 was not fetched.

## Sources Consulted

- **Source control history**: commits `592117c`, `ac31f8d`, `355f1a4`, `ed758f0`, `c82590f`, `11b35fe`; PRs #105, #109, #117, #118, #122, #139, #188, #199, #282, #285.
- **Issue / ticket tracker**: GitHub Issues via `gh` (7 issues, none relevant). Linear/Jira: no MCP.
- **Long-form documents**: in-repo docs covered under source control. Notion/Confluence: no MCP.
- **Real-time team chat**: not searched. No Slack MCP. Amp threads on #285/#282 and Claude Code session on #117 unsearched.
- **Infrastructure observability**: not applicable; no MCP.
- **Error / exception tracking**: not applicable; no MCP.
- **Product analytics warehouse**: not applicable; no MCP.

## Confidence Summary

The table's origin, initial contents, and later chronology are Direct. "Plugin settings arrived later" is a chronology fact. "Left model overrides behind" is accurate as a description of what #285 did not touch, but whether that was oversight, deferral, or design is unknown. "Handoff in the list does not mean handoff should become a plugin" is supported: the list has never been a plugin list. "Plugin settings are the right owner now" is a design judgment the record neither confirms nor refutes; H2 is a live alternative.

## Constraints for planning

**Preserve**

- `saber models set|current|reset --agent handoff` must keep working.
- `resolve_subagent_model` precedence (tool override → subagent → main) and API-key branch are the SDK-facing contract since #122/#199.
- SDK callers do not read the CLI plugin store. Programmatic override must continue through `tool_overrides` and capability construction without `plugin_config.json`.
- Existing `model_config.json` may contain `subagents.viz` and `subagents.notebook`. Read, migrate, or document the break.
- Docs that teach `--agent notebook` / `--agent viz` must update in the same change.

**Change**

- If plugin model overrides move to plugin settings: add a `model` Setting per plugin; bind supplies it to the capability config; `resolve_subagent_model` consults that source. Decide how it interacts with `tool_overrides`.
- If honoring #285's principle, `SUBAGENT_KEYS` shrinks to core agents (`handoff`) and CLI/in-memory filters follow.
- Sandbox currently has no CLI model path. Give it one in the same change.
- `saber models current` must either keep showing plugin models from the new source or stop claiming to overview all nested overrides.

**Avoid**

- Do not turn handoff into a plugin or a capability.
- Do not make SDK factories read `plugin_config.json`.
- Do not remove file-backed any-key `set_subagent_model` without a migration story.
- Do not present #285's "without teaching core about individual plugins" as a written mandate covering models; the body does not say so.

**Risk**

- Author intent for #118, #122, and #285's model omission is unwritten.
- Sandbox's gap may be a pending one-line tuple append or a plugin-settings field.
- Changing `SUBAGENT_KEYS` alters `Config.in_memory(subagent_models={"sandbox": ...})` which currently discards sandbox.
- Changing where the middle tier comes from changes when `main_api_key` is reused.

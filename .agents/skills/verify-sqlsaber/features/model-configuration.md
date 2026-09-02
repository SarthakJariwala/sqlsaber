# Model configuration

Model commands let a user inspect and select the main model, control its thinking level, set per-agent overrides, list models from the remote catalog, and reset saved choices.

## Sub-features

- `models-current` shows the effective main model, thinking state, and subagent overrides.
- `models-set-main` saves a provider-prefixed main model and optional thinking level.
- `models-set-agent` saves an override for `handoff`, `viz`, or `notebook`.
- `models-list` fetches the current supported catalog from models.dev.
- `models-reset` writes the command's built-in main-model reset target or clears one subagent override.

## How to get to it (user POV)

- Run `saber models current [--agent AGENT]`.
- Run `saber models set PROVIDER:MODEL [--thinking-level LEVEL] [--agent AGENT]`.
- Run `saber models list`.
- Run `saber models reset [--agent AGENT]`, with `--yes` for automation.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- Model configuration does not require a provider key. A later query does.

- **Initial state.** Capture `saber models current`. It prints the current model, thinking state, and rows for `handoff`, `viz`, and `notebook`.
- **Main selection.** Set `openai:gpt-5 --thinking-level off`, capture `models current --agent main`, and require the model plus `Thinking: disabled`.
- **Subagent override.** Set `openai:gpt-5-mini --agent handoff`, then capture `models current --agent handoff`. It shows the override and main model. Reset the override with `--yes` and confirm it uses main again.
- **Main reset.** Run `models reset --yes`, then capture current state. It shows `openai:gpt-5.6-sol`, the reset target printed by the command.
- **Persisted proof.** Copy the file from `path model-config` before reset and after reset. The JSON matches each `models current` read-back.
- **Remote catalog.** When network access is available, run `models list` with a short explicit timeout. Require `Available Models` and provider-prefixed IDs. Require a current marker only when the configured model appears in the returned catalog. A network failure makes only this entry unreachable.

## Gotchas

- `--thinking-level` applies only to the main model.
- Model IDs must use a supported `PROVIDER:MODEL` prefix, but saving one does not prove the provider accepts it.
- `models list` calls `https://models.dev/api.json`; current and set are local.
- `main`, `handoff`, `viz`, and `notebook` are the accepted agent names.
- Fresh config and `models reset` both use `openai:gpt-5.6-sol`. Assert that identifier in `models current` and in the file from `path model-config`.

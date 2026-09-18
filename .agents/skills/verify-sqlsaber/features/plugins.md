# Plugins

Plugin commands let a user list installed capabilities, inspect and save non-secret settings, and enable or disable auto-loading for later CLI sessions.

## Sub-features

- `plugins-list` shows each installed plugin and whether it is using defaults, saved settings, disabled, or still needs setup.
- `plugins-show` prints effective values and sources for one plugin. Secrets are redacted.
- `plugins-setup-set` configures and enables a plugin with `--set FIELD=VALUE` (repeatable) and `--yes`, without a prompt.
- `plugins-set-unset` changes or removes one saved non-secret field without enabling a disabled plugin.
- `plugins-enable-disable` toggles auto-loading. Disable keeps saved settings.
- `plugins-setup-interactive` opens a terminal wizard when `--set` is omitted. A non-TTY call without `--set` is a usage error.

## How to get to it (user POV)

- Run `saber plugins list`.
- Run `saber plugins show NAME` or `saber plugins show NAME FIELD`.
- Run `saber plugins setup NAME --set FIELD=VALUE --yes` for scripted setup.
- Run `saber plugins setup NAME` in a terminal for the wizard.
- Run `saber plugins set NAME FIELD VALUE` or `saber plugins unset NAME FIELD`.
- Run `saber plugins enable NAME` or `saber plugins disable NAME`.
- In the interactive session, `/plugins …` is the first management family in the command palette.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- This checkout's `uv sync --locked` installs the in-repo plugins (`notebook`, `sandbox`, `viz`) through the default dependency group.
- Never pass a secret through `--set`. `--secret-stdin` is `verified-unreachable` with the null keyring.

- **List.** Capture `saber plugins list`. A fresh home shows `notebook` and `viz` as `defaults`, and `sandbox` as `setup required: saber plugins setup sandbox` because `provider` is required.
- **Show.** Capture `saber plugins show notebook`. Require `Auto-loading: enabled`, `backend` effective `docker` from `default`, `model` effective `unset (uses main model)`, and secret fields as `hidden` / `not applicable` when inactive.
- **Non-interactive setup.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence plugins/setup-notebook.txt -- uv run saber plugins setup notebook --set backend=docker --yes`. Then `saber plugins setup sandbox --set provider=docker --yes`. Then `saber plugins setup viz --set model=openai:gpt-5-mini --yes`. Output includes `Saved … settings. Changes apply to new CLI sessions.`
- **Set and unset.** Run `saber plugins set notebook memory_mb 2048 --yes`, capture `plugins show notebook` (`memory_mb` `2048` / `saved`), then `saber plugins unset notebook memory_mb --yes`.
- **Disable and enable.** Capture `saber plugins disable viz` (`viz disabled for new CLI sessions.`), list (`viz` `disabled`), then `saber plugins enable viz` and list (`viz` `configured` because the saved model remains).
- **Non-TTY wizard.** Use `verify-sqlsaber run` for `uv run saber plugins setup notebook` with no `--set`. Exit `2`. Stderr contains `Setup requires a terminal or --set`.
- **Persisted proof.** Copy the file from `path plugin-config`. Version `1` records enabled flags and saved non-secret settings that match the last list/show.
- **Interactive wizard.** In a fresh TTY drive of `saber plugins setup notebook` with no `--set`, cancel with Esc. Require `Execution backend`, visible choices including `docker`, and `Setup cancelled. No settings saved.` Do not substitute `--set` when claiming wizard behavior.

## Gotchas

- Settings apply to **new** CLI sessions, not the process that wrote them.
- `viz` declares a `model` field. Unset inherits the main model. `show` prints `unset (uses main model)`.
- An installed plugin with a required unset field (sandbox `provider`) warns on query and TUI bind: `sandbox.provider is required. Run: saber plugins setup sandbox`. That warning is not a failed database lookup.
- `--set` rejects secrets. Typed secret storage needs a disposable keyring; this harness uses a null backend.
- `SQLSABER_NOTEBOOK_BACKEND` and similar env vars override saved values. Doctor does not unset them; a present override changes `show` source text.

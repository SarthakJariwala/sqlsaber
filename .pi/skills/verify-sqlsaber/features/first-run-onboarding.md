# First-run onboarding

When a user starts `saber` without `-d` and has no saved database, SQLsaber opens a guided terminal setup for a database, authentication, and model choice before entering chat.

## Sub-features

- `onboarding-trigger` appears only when no saved database and no command-line selector exist.
- `onboarding-database` collects, tests, and saves a named connection.
- `onboarding-auth-model` offers provider authentication and model selection after the database succeeds.
- `onboarding-cancel` exits without pretending setup completed and prints manual setup guidance.

## How to get to it (user POV)

- Run `saber` in a fresh home.
- Complete the database prompts, then the authentication and model prompts.
- Press Ctrl+C to cancel.
- Pass `-d FILE` or configure a saved database to skip onboarding.

## Driving it with verify-sqlsaber

Preconditions:

- Launch has completed and doctor reports `HEALTHY`.
- No database command has run in this isolated home.

- **Trigger and cancel.** Run `saber` through `drive` with Ctrl+C as deterministic input after the welcome form appears. Require `Welcome to SQLsaber`, the first database prompt, `Database setup is required to continue`, the `-d` guidance, a setup-incomplete exit, and no saved database config.
- **Bypass with a file.** In a fresh run, start `saber -d "$FIXTURE"` with a harmless placeholder credential and send Ctrl+D on the empty editor. It reaches the interactive session without the onboarding welcome because a selector was supplied.
- **Completed database step.** Build JSON input events with `uv run python -c 'import json,sys; print(json.dumps([[2,"onboarded\\r"],[4,"\\x1b[B\\x1b[B\\r"],[6,sys.argv[1]+"\\r"]]))' "$FIXTURE"`, then pass that value to `drive --input-sequence` for `uv run saber`. The events name the connection, choose SQLite, and enter only the disposable fixture path. Require a successful connection and a `db list` read-back. If prompt timing differs, retain the transcript and report the harness gap rather than sending more input blindly.
- **Authentication and model step.** This needs an environment credential matching the selected provider and network access to the model catalog. After the guided database succeeds, select that provider. Require `Step 2 of 2: Authentication`, credential detection through the named environment variable, model fetch or documented fallback, `Model set to:`, the all-set summary, and the interactive-session start. `auth_config.json` must record `api_key`; derive the provider from the environment notice and selected model in the transcript. `model_config.json` must match the selected model. Only a real query proves provider acceptance. If either prerequisite is absent, retain the attempted route and mark only `onboarding-auth-model` unreachable.
- **Keyring boundary.** Full credential storage is unreachable by design because the harness uses a null keyring. Do not enter a real key to bypass this boundary; use an inherited environment credential for the completed step.

## Gotchas

- `launch` seeds a fixture but does not register it, so the first run remains a real onboarding state.
- Onboarding is bypassed by any explicit `-d` selector.
- At least a database must succeed for onboarding to return success.
- Prompt timing is deterministic only for setup controls. Never schedule input around a network-backed model response.

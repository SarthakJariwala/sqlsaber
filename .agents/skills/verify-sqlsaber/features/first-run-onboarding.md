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

- **Trigger and cancel.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 30 --input $'\003' --input-delay 4 --evidence first-run-onboarding/cancel.txt -- uv run saber`. Require `Welcome to SQLsaber`, `What would you like to name this connection?`, `Database setup is required to continue.`, `saber -d <connection-string>`, `Setup incomplete. Please configure your database and try again.`, exit code `1`, and no `database_config.json`.
- **Bypass with a file.** In a fresh run, start `saber -d "$FIXTURE"` with a placeholder `OPENAI_API_KEY` and send Ctrl+D after about 4 seconds. It reaches the interactive session without `Welcome to SQLsaber`. The fresh default model is `openai:gpt-5.6-sol`.
- **Completed database step.** Build JSON input events with `uv run python -c 'import json,sys; print(json.dumps([[4,"odb\\r"],[7,"\\x1b[B\\x1b[B\\r"],[10,sys.argv[1]+"\\r"]]))' "$FIXTURE"`, then pass that value to `drive --input-sequence`. The name field is prefilled with `mydb` and the cursor is at the start of the field, so `odb` saves as `odbmydb`. Two down arrows choose SQLite. Require `Connection to 'odbmydb' successful` and a `db list` read-back. If prompt timing differs, retain the transcript and report the harness gap rather than sending more input blindly.
- **Authentication and model step.** This needs an environment credential matching the selected provider and network access to the model catalog. After the database succeeds, press Enter to select `openai` when `OPENAI_API_KEY` is present, then Enter again (~22 seconds) for `GPT-5.6 Sol (Recommended)`. Require `Step 2 of 2: Authentication`, `Existing authentication found for openai: OPENAI_API_KEY`, `Model set to: openai:gpt-5.6-sol`, `You're all set!`, and `Starting interactive session...`. `auth_config.json` records `{"auth_method": "api_key"}`. `model_config.json` records `openai:gpt-5.6-sol`. Esc on Step 2 skips auth (`You can set it up later using 'saber auth setup'`) and still prints `You're all set!` after a saved database; that is not the completed auth-model entry. If no matching env credential or catalog fetch exists, retain the attempted route and mark only `onboarding-auth-model` unreachable.
- **Keyring boundary.** Full credential storage is unreachable by design because the harness uses a null keyring. Do not enter a real key to bypass this boundary; use an inherited environment credential for the completed step.

## Gotchas

- `launch` seeds a fixture but does not register it, so the first run remains a real onboarding state.
- Onboarding is bypassed by any explicit `-d` selector.
- At least a database must succeed for onboarding to return success. Cancel on the name prompt is exit `1` with the `-d` guidance, not `Onboarding cancelled.`
- The connection-name prompt starts as `mydb`. Typed text inserts at the start unless you clear the field.
- Prompt timing is deterministic only for setup controls. Never schedule input around a network-backed model response.
- The default Step 2 provider is OpenAI. Enter accepts that highlight. An `OPENAI_API_KEY` does not change the cursor.

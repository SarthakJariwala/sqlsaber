# Authentication

Authentication commands let a user configure provider API keys, inspect which providers are available through environment variables or stored credentials, and remove a stored key.

## Sub-features

- `auth-setup` selects a provider. An existing environment variable completes setup without storing a key. A typed key would go to the OS keyring when both env and keyring are empty.
- `auth-status` reports whether API-key authentication is configured and where each available provider key comes from.
- `auth-reset` removes one stored provider key after confirmation and leaves environment variables unchanged.

## How to get to it (user POV)

- Run `saber auth setup` in a terminal.
- Run `saber auth status`.
- Run `saber auth reset [PROVIDER]`, with `--yes` for deliberate automation.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY` and `keyring backend: keyring.backends.null.Keyring`.
- Never type a real API key into a transcript.

- **Fresh status.** Capture `saber auth status`. A fresh run prints `Authentication Status`, says no method is configured, and points to `saber auth setup`.
- **Environment status.** If `OPENAI_API_KEY` is present, run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 30 --input-sequence '[[3, "\u001b[B\r"]]' --evidence authentication/setup-openai.txt -- uv run saber auth setup`. One down arrow selects `openai`. Output includes `Existing authentication found for openai: OPENAI_API_KEY` and `Openai API key configured successfully!`. Then capture `saber auth status`. It prints `API Key authentication configured` and `configured via OPENAI_API_KEY` without the secret. `path auth-config` is `{"auth_method": "api_key"}`.
- **Reset without stored credentials.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence authentication/reset-empty.txt -- uv run saber auth reset openai --yes`. With the null keyring it exits `0` and says no stored credentials were found. This proves the no-op route, not credential deletion.
- **Typed-key setup.** Cancel at the provider or key prompt and retain the transcript. Key storage and the corresponding destructive reset are `verified-unreachable` in this harness because the null backend blocks any persistent keyring. Their concrete prerequisite is a disposable credential store with a seeded test key; never use the operator's OS keyring.

## Gotchas

- Environment variables take precedence over keyring values.
- Status reports provider availability only after an authentication method has been configured.
- `auth reset` never changes environment variables.
- Without an explicit provider, reset opens an interactive selector and fails usage when stdin is not a terminal.
- The verification harness uses a null keyring to protect the operator's credentials. Do not weaken that isolation to make setup pass.

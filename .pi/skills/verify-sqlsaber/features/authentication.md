# Authentication

Authentication commands let a user configure provider API keys, inspect which providers are available through environment variables or stored credentials, and remove a stored key.

## Sub-features

- `auth-setup` selects a provider and stores an entered API key in the operating-system keyring.
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
- **Environment status.** If a provider credential is already present in the coordinator environment, create only the isolated non-secret auth method marker through the normal setup path, then capture status. It names the corresponding environment variable without printing its value.
- **Reset without stored credentials.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence authentication/reset-empty.txt -- uv run saber auth reset openai --yes`. With the null keyring it exits `0` and says no stored credentials were found. This proves the no-op route, not credential deletion.
- **Setup route.** Start `saber auth setup` through `drive`, cancel at the provider or key prompt, and retain the transcript. Key storage and the corresponding destructive reset are `verified-unreachable` in this harness because the null backend deliberately blocks any persistent keyring. Their concrete prerequisite is a disposable credential store with a seeded test key; never use the operator's OS keyring.

## Gotchas

- Environment variables take precedence over keyring values.
- Status reports provider availability only after an authentication method has been configured.
- `auth reset` never changes environment variables.
- Without an explicit provider, reset opens an interactive selector and fails usage when stdin is not a terminal.
- The verification harness uses a null keyring to protect the operator's credentials. Do not weaken that isolation to make setup pass.

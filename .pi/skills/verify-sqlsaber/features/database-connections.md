# Database connections

Database connection commands let a user save named databases, inspect and test them, choose a default, change schema exclusions, and remove connections.

## Sub-features

- `db-add-interactive` collects connection details in terminal prompts.
- `db-add-noninteractive` saves a fully specified SQLite, DuckDB, PostgreSQL, or MySQL connection, including an optional description.
- `db-list-test` lists saved details and tests either a named connection or the default with a health query.
- `db-exclude` replaces, adds, removes, clears, or interactively edits excluded schemas.
- `db-default` chooses the connection used when `-d` is omitted.
- `db-remove` confirms deletion or accepts deliberate `--yes` automation.

## How to get to it (user POV)

- Run `saber db add NAME` for prompted setup.
- Run `saber db add NAME --type sqlite --database FILE --description TEXT --no-interactive` for scripted local setup.
- Run `saber db list` or `saber db test [NAME]` to inspect and test connections.
- Run `saber db exclude NAME --set LIST`, `--add LIST`, `--remove LIST`, or `--clear`. With no action flag, SQLsaber opens an editor and saves the response.
- Run `saber db set-default NAME` or `saber db remove NAME` to change saved state.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)` points to the seeded SQLite file.
- The isolated run has no saved connection named `verification` or `secondary`.

- **Non-interactive add.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence database-connections/add.txt -- uv run saber db add verification --type sqlite --database "$FIXTURE" --description "verification fixture" --no-interactive`. Output confirms the add and says `verification` became the default.
- **List and default test.** Capture `saber db list`, then `saber db test` without a name. The table contains `verification`, `sqlite`, the fixture path, and a default marker. The test says `Connection to 'verification' successful`.
- **Exclusions.** Set `temp,audit`, add `archive`, remove `temp`, and capture `saber db list`; it shows `audit, archive`. Clear exclusions and confirm the list is blank. This covers `--set`, `--add`, `--remove`, and `--clear` without an interactive editor.
- **Default selection.** Add `secondary` against the same fixture, set it as default, and capture a list where only `secondary` is marked. Remove `secondary --yes`; a final list contains `verification` and marks it as default.
- **Interactive add.** Start a fresh run. Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 30 --input-sequence '[[2, "\u001b[B\u001b[B\r"], [4, "FIXTURE_PATH\r"]]' --evidence database-connections/interactive-add.txt -- uv run saber db add verification`, replacing `FIXTURE_PATH` before passing the JSON. The two down-arrow keys choose SQLite. Require the add confirmation and a list read-back. If the prompt does not consume the path, retain the transcript and report this entry as unreachable with the attempted input.
- **Persisted proof.** Copy the file from `path database-config` after the final list. Its default, connection names, paths, and exclusions match the list. The saved description matches the original add argument; `db list` does not display descriptions.

## Gotchas

- Non-interactive SQLite adds save the path as supplied. Interactive SQLite and DuckDB setup resolves it to an absolute path.
- Database passwords use the operating-system keyring in normal use. Verification selects a null backend, so never use it to prove server-password persistence.
- The first connection becomes default automatically. Removing the default promotes the first remaining connection.
- `db test` opens the real database. Keep remote checks read-only.
- `db exclude NAME` with no action flag is a mutation, not an inspection command.
- `db remove --yes` skips confirmation but still deletes saved configuration and any stored password.

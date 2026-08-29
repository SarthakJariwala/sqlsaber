# Database connections

Database connection commands let a user save named databases, inspect and test them, choose a default, change schema exclusions, and remove connections.

## Sub-features

- `db-add-interactive` collects connection details in a terminal prompt.
- `db-add-noninteractive` saves a fully specified SQLite, DuckDB, PostgreSQL, or MySQL connection.
- `db-list-test` lists saved details and opens the selected database for a health query.
- `db-exclude` reads or changes excluded schemas.
- `db-default` chooses the connection used when `-d` is omitted.
- `db-remove` confirms deletion or accepts deliberate `--yes` automation.

## How to get to it (user POV)

- Run `saber db add NAME` for prompted setup.
- Run `saber db add NAME --type sqlite --database FILE --no-interactive` for scripted local setup.
- Run `saber db list`, `saber db test NAME`, or `saber db exclude NAME` to inspect a connection.
- Run `saber db set-default NAME` or `saber db remove NAME` to change saved state.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)` points to the seeded SQLite file.
- The isolated run has no saved connection named `verification` or `secondary`.

- **Non-interactive add.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence database-connections/add.txt -- uv run saber db add verification --type sqlite --database "$FIXTURE" --no-interactive`. Output confirms the add and says `verification` became the default.
- **List read-back.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence database-connections/list.txt -- uv run saber db list`. The `Database Connections` table contains `verification`, `sqlite`, the fixture path, and a default check mark.
- **Connection test.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence database-connections/test.txt -- uv run saber db test verification`. Output says `Connection to 'verification' successful` and exits `0`.
- **Exclusions.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence database-connections/exclude.txt -- uv run saber db exclude verification --set temp,audit`. A second `saber db list` transcript shows `temp, audit`. Run `saber db exclude verification --clear` and confirm the list is blank.
- **Default selection.** Add `secondary` against the same disposable fixture, then run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence database-connections/default.txt -- uv run saber db set-default secondary`. A new list transcript marks `secondary` as default and no longer marks `verification`.
- **Removal.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence database-connections/remove.txt -- uv run saber db remove secondary --yes`. Output confirms removal. A final list contains `verification`, omits `secondary`, and marks `verification` as default.
- **Interactive add.** Start a fresh run. Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 30 --input $'\033[B\033[B\r'"$FIXTURE"$'\r' --input-delay 1.5 --evidence database-connections/interactive-add.txt -- uv run saber db add verification`. The two down-arrow keys choose `sqlite`, Enter confirms it, and the next line supplies the fixture path. Require the same add confirmation and list read-back. If questionary does not consume the buffered path, retain the transcript and report `db-add-interactive` as unreachable with this helper. Do not substitute the non-interactive path.
- **Persisted proof.** Run `DB_CONFIG=$("$VERIFY_SQLSABER" path "$RUN_ID" database-config)`, copy it to the feature evidence directory, and parse it read-only. Its `default` and `connections` values must match the final `saber db list` transcript.

## Gotchas

- A SQLite add saves the path as supplied, while DuckDB normalizes it to an absolute path. Compare against the displayed value for that database type.
- Database passwords go to the operating-system keyring. Never test server passwords in a shared keyring; use SQLite or an isolated verification keyring backend.
- The first connection becomes default automatically. Removing the default promotes the first remaining connection.
- `db test` opens the real database. Keep remote connection checks read-only and record the endpoint used.
- `db remove --yes` skips only confirmation. It still deletes the saved config and any stored password for that connection.

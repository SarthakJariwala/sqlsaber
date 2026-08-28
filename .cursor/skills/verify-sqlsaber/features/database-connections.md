# Database connections

Database connections let a user save a named SQLite, DuckDB, PostgreSQL, or MySQL target, see it in the connection list, test that it answers, choose a default, and remove it without touching other connections.

## Sub-features

- `db-add-sqlite` saves a named SQLite connection from flags, without prompts.
- `db-list` shows saved names, types, and the default marker.
- `db-test` runs a live `SELECT 1` against the named connection.
- `db-set-default` changes which connection is used when `-d` is omitted.
- `db-remove` deletes a named connection after `--yes`.

## How to get to it (user POV)

- Run `saber db add <name> --type sqlite --database <abs-path> --no-interactive`.
- Run `saber db list`.
- Run `saber db test <name>` or `saber db test` for the default.
- Run `saber db set-default <name>`.
- Run `saber db remove <name> --yes`.

## Driving it with control-sqlsaber

Preconditions:

- `control-sqlsaber doctor` is clean.
- Baseline connection `verify-sqlite` exists and is the default.
- No connection is named `verify-alt`.
- Seed sqlite path is `control-sqlsaber path SEEDED_DB`.

- **Add second connection.** Save another SQLite name against the seed file. Run `control-sqlsaber cli --out artifacts/database-connections/add.txt -- db add verify-alt --type sqlite --database "$(control-sqlsaber path SEEDED_DB)" --no-interactive --description "verification alternate"`. Exit code `0`. Stdout contains `Successfully added database connection 'verify-alt'` and does not contain `Set 'verify-alt' as default database`.
- **List both.** Run `control-sqlsaber cli --out artifacts/database-connections/list-both.txt -- db list`. Exit code `0`. Stdout contains `Database Connections`, `verify-sqlite`, `verify-alt`, and `sqlite`.
- **Test new connection.** Run `control-sqlsaber cli --out artifacts/database-connections/test-alt.txt -- db test verify-alt`. Exit code `0`. Stdout contains `Testing connection to 'verify-alt'...` and `Connection to 'verify-alt' successful`.
- **Change default.** Run `control-sqlsaber cli --out artifacts/database-connections/set-default-alt.txt -- db set-default verify-alt`. Exit code `0`. Stdout contains `Successfully set 'verify-alt' as default database`.
- **Confirm default.** Run `control-sqlsaber cli --out artifacts/database-connections/list-default-alt.txt -- db list`. The `verify-alt` row is marked default (`✓` in the Default column).
- **Restore default.** Run `control-sqlsaber cli -- db set-default verify-sqlite`. Stdout contains `Successfully set 'verify-sqlite' as default database`.
- **Remove extra connection.** Run `control-sqlsaber cli --out artifacts/database-connections/remove.txt -- db remove verify-alt --yes`. Exit code `0`. Stdout contains `Successfully removed database connection 'verify-alt'`.
- **Confirm removal.** Run `control-sqlsaber cli --out artifacts/database-connections/list-final.txt -- db list`. Stdout contains `verify-sqlite` and does not contain `verify-alt`.
- **Confirm persistence.** Copy `$(control-sqlsaber path CONFIG_FILE)` to `artifacts/database-connections/database_config.json`. The JSON `connections` object has `verify-sqlite` and does not have `verify-alt`.
- **Proof.** Keep the add, list-both, test, remove, list-final, and config JSON artifacts together. They show the extra connection appearing, answering, and disappearing while `verify-sqlite` remains.

## Gotchas

- `saber db add` without `--no-interactive` opens questionary prompts. That path is not this recipe.
- SQLite `--database` must be an absolute path. A relative path is stored as given and later `db test` depends on cwd.
- The first saved connection is auto-defaulted. `verify-sqlite` already is, so `verify-alt` must not print the default line.
- Rich table cells truncate when `COLUMNS` is narrow. Assert on the connection name strings, not on every wrapped cell. The helper exports `COLUMNS=200`.
- `db remove` without `--yes` exits non-zero when stdin is not a TTY. Use `--yes`.
- Do not `db remove verify-sqlite` in this recipe. Later features need the baseline connection.
- Passwords for PostgreSQL/MySQL go to the OS keyring. This run uses a null keyring; do not treat a skipped server-password save as a SQLite bug.

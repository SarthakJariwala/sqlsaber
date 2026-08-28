# Knowledge base

Knowledge base lets a user save named business context for the current database, list and show those entries, search them by keyword, and remove them.

## Sub-features

- `knowledge-add` stores a name, description, optional SQL, and optional source on `verify-sqlite`.
- `knowledge-list` shows IDs and names for that database.
- `knowledge-show` prints the full entry by ID.
- `knowledge-search` ranks entries by full-text match.
- `knowledge-search-empty` reports a complete miss.
- `knowledge-remove` deletes one entry after `--yes`.

## How to get to it (user POV)

- Run `saber knowledge add "Name" "Description"` with optional `--sql`, `--source`, and `-d`.
- Run `saber knowledge list`.
- Run `saber knowledge show ENTRY_ID`.
- Run `saber knowledge search "query"`.
- Run `saber knowledge remove ENTRY_ID --yes`.

## Driving it with control-sqlsaber

Preconditions:

- `control-sqlsaber doctor` is clean.
- Default database is `verify-sqlite`.
- No knowledge entry is named `Revenue KPI`.

- **Add entry.** Run `control-sqlsaber cli --out artifacts/knowledge-base/add.txt -- knowledge add "Revenue KPI" "Recognized revenue from shipped orders only" --sql "SELECT SUM(amount_cents) FROM orders WHERE status = 'shipped'" --source "finance-wiki"`. Exit code `0`. Stdout contains `Knowledge entry added for database 'verify-sqlite'`, `Name: Revenue KPI`, and an `ID:` UUID. Record that UUID as `ENTRY_ID`.
- **List.** Run `control-sqlsaber cli --out artifacts/knowledge-base/list.txt -- knowledge list`. Exit code `0`. Stdout contains `Knowledge Entries for Database: verify-sqlite`, `Revenue KPI`, and `Total entries: 1`.
- **Show.** Run `control-sqlsaber cli --out artifacts/knowledge-base/show.txt -- knowledge show ENTRY_ID`. Exit code `0`. Stdout contains `Name: Revenue KPI`, `Recognized revenue from shipped orders only`, `SELECT SUM(amount_cents) FROM orders WHERE status = 'shipped'`, and `Source: finance-wiki`.
- **Search hit.** Run `control-sqlsaber cli --out artifacts/knowledge-base/search-hit.txt -- knowledge search "shipped revenue"`. Exit code `0`. Stdout contains `Knowledge Search Results` and `Revenue KPI`.
- **Search miss.** Run `control-sqlsaber cli --out artifacts/knowledge-base/search-miss.txt -- knowledge search "volcano"`. Exit code `0`. Stdout contains `No knowledge entries matched 'volcano' for database 'verify-sqlite'`.
- **Remove.** Run `control-sqlsaber cli --out artifacts/knowledge-base/remove.txt -- knowledge remove ENTRY_ID --yes`. Exit code `0`. Stdout contains `Knowledge entry removed from database 'verify-sqlite'`.
- **Confirm removal.** Run `control-sqlsaber cli --out artifacts/knowledge-base/list-empty.txt -- knowledge list`. Stdout contains `No knowledge entries found for database 'verify-sqlite'`.
- **Proof.** Keep add, show, search-hit, search-miss, and list-empty. The ID from add must be the ID shown and removed.

## Gotchas

- Knowledge is scoped to a saved connection name. Adding with `-d` of an unknown name exits `1` with `database connection 'NAME' not found`.
- Without any saved connection, `knowledge list` exits `1` with `no database connections configured`. This recipe assumes launch already added `verify-sqlite`.
- `knowledge remove` and `knowledge clear` require `--yes` when stdin is not a TTY.
- Search ranking is full-text, not substring-on-list. Assert on the printed name, not on result order beyond the expected hit being present.
- Do not leave `Revenue KPI` behind. Other recipes assume the empty knowledge list unless they add their own entries.

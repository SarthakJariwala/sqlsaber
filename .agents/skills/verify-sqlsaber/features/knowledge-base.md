# Knowledge base

The knowledge base lets a user store database-scoped definitions and SQL patterns, find and inspect them later, and make that context available to natural-language queries.

## Sub-features

- `knowledge-add` stores a name and description with optional SQL and source fields.
- `knowledge-list-show` lists scoped entries and displays one full record by ID.
- `knowledge-search` returns full-text matches ranked within one database.
- `knowledge-remove` deletes one confirmed entry.
- `knowledge-clear` deletes all entries for the selected database.
- `knowledge-query-use` lets the query agent search knowledge for its current database.

## How to get to it (user POV)

- Run `saber knowledge add NAME DESCRIPTION` with optional `--sql`, `--source`, and `--database`.
- Run `saber knowledge list`, `show ID`, or `search QUERY` to read entries.
- Run `saber knowledge remove ID` or `clear` to delete entries, with `--yes` for deliberate automation.
- Ask a natural-language question whose answer depends on a saved definition or SQL pattern.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- A saved SQLite connection named `verification` exists and is the default.
- No knowledge entry is named `Paid order definition` in this isolated run.

- **Add and identify.** Add `Paid order definition` with description `Orders count as paid when status equals paid`, SQL `SELECT COUNT(*) FROM orders WHERE status = 'paid'`, and source `verification-fixture`. Output prints the exact name and a full UUID. Extract the UUID with `grep -Eo '[0-9a-f]{8}-[0-9a-f-]{27}'`, not an anchored `ID:` line.
- **Read paths.** Capture `knowledge list`, `show "$ENTRY_ID"`, a search for `paid orders`, and a search for `volcano`. The list and show match the added record. The first search finds it; the second reports no match and exits `0`.
- **Persisted row proof.** Before deletion, copy the file from `path knowledge-db` to `knowledge/before-delete.db`. Query the copy read-only and require the UUID, name, database `verification`, SQL, and source.
- **Remove and clear.** Remove the entry with `--yes` and confirm an empty list. Add two entries, clear database `verification --yes`, and confirm the final list is empty.
- **Persisted empty proof.** Copy the final knowledge database to `knowledge/after-clear.db` and query it read-only. It has no rows for `verification`.
- **Agent retrieval.** When a matching provider credential works, add a deliberately unique rule, ask a saved-selector query that needs it, and require the visible answer plus a `search_knowledge` tool result naming the entry. If no credential works, record that prerequisite and keep the CRUD proof.

## Gotchas

- Knowledge commands require a saved database name. Passing an ad hoc file to a query does not create a knowledge scope.
- Search uses FTS5 token matching with OR semantics for plain terms. Assert the expected name is present, not a fixed result order. A TTY `drive` table can wrap `Paid order definition` across two rows; grep a unique token such as `Paid order`, or use `run` for a single-line name.
- Search indexes name, description, and SQL, not source.
- IDs are UUIDs. Capture the full value from add or list.
- `remove` and `clear` require confirmation in a terminal. Use `--yes` only against the isolated database.
- Copy row-level proof before clear. The final database can prove emptiness, not the deleted row's fields.

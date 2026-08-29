# Knowledge base

The knowledge base lets a user store database-scoped definitions and SQL patterns, find them later, inspect full entries, and remove stale context.

## Sub-features

- `knowledge-add` stores a name and description with optional SQL and source fields.
- `knowledge-list-show` lists scoped entries and displays one full record by ID.
- `knowledge-search` returns full-text matches ranked within one database.
- `knowledge-remove` deletes one confirmed entry.
- `knowledge-clear` deletes all entries for the selected database.

## How to get to it (user POV)

- Run `saber knowledge add NAME DESCRIPTION` with optional `--sql`, `--source`, and `--database`.
- Run `saber knowledge list`, `show ID`, or `search QUERY` to read entries.
- Run `saber knowledge remove ID` or `clear` to delete entries, with `--yes` for deliberate automation.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- A saved SQLite connection named `verification` exists and is the default.
- No knowledge entry is named `Paid order definition` in this isolated run.

- **Add entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence knowledge/add.txt -- uv run saber knowledge add "Paid order definition" "Orders count as paid when status equals paid" --sql "SELECT COUNT(*) FROM orders WHERE status = 'paid'" --source "verification-fixture"`. Output prints a full entry ID, the exact name, and exit code `0`.
- **List read-back.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence knowledge/list.txt -- uv run saber knowledge list`. The table title names database `verification`, contains the new ID and name, and reports `Total entries: 1`.
- **Show entry.** Run `EVIDENCE=$("$VERIFY_SQLSABER" path "$RUN_ID" evidence)` and `ENTRY_ID=$(awk '/^ID:/{print $2; exit}' "$EVIDENCE/knowledge/add.txt")`. Require a 36-character ID, then run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence knowledge/show.txt -- uv run saber knowledge show "$ENTRY_ID"`. The output includes the full description, SQL, source, and database.
- **Search match.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence knowledge/search-match.txt -- uv run saber knowledge search "paid orders" --limit 5`. Results contain `Paid order definition`. Run a second search for `volcano`; output says no entries matched and still exits `0`.
- **Remove one.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence knowledge/remove.txt -- uv run saber knowledge remove "$ENTRY_ID" --yes`. A list read-back reports no knowledge entries.
- **Clear all.** Add two new entries, run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence knowledge/clear.txt -- uv run saber knowledge clear --database verification --yes`, then list again. Output reports two cleared entries and the final list is empty.
- **Persisted proof.** Run `KNOWLEDGE_DB=$("$VERIFY_SQLSABER" path "$RUN_ID" knowledge-db)`, copy it into the feature evidence directory, and query the copy with Python's `sqlite3` in read-only mode. The `knowledge` rows for `verification` must match the final CLI list. Record row IDs, names, and database names, not only a count.

## Gotchas

- Knowledge commands require a saved database name. Passing an ad hoc file to a query does not create a knowledge scope.
- Search uses FTS5 token matching. Use plain terms such as `paid orders`, not punctuation-heavy SQL fragments.
- IDs are UUIDs. Capture the full value from `add` or `list`; do not use a shortened prefix.
- `remove` and `clear` require confirmation in a terminal. Use `--yes` only against this run's isolated database.
- A successful delete message is not enough. Confirm through `list` and the copied `knowledge.db`.

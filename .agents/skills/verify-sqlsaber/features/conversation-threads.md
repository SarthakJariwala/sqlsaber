# Conversation threads

Conversation threads let a user find saved query sessions, read complete transcripts, inspect durable artifacts, resume context, export HTML, and preview or perform retention cleanup.

## Sub-features

- `threads-list-empty` reports an empty store before any query runs.
- `threads-list-show` lists retained sessions and renders one transcript by full ID.
- `threads-artifacts` lists durable outputs referenced by a thread.
- `threads-resume` reopens a thread in the terminal UI or continues it with root `--thread`.
- `threads-export` writes an HTML transcript whose core text is embedded in the file.
- `threads-prune-preview` reports eligible old threads without deleting them.
- `threads-prune-delete` deletes eligible old threads after confirmation; completed sessions also remove threads older than 30 days.

## How to get to it (user POV)

- Run `saber threads list` before or after a single-shot query, then `saber threads show ID`.
- Run `saber threads artifacts ID` to inspect retained output references.
- Run `saber threads resume ID` or `saber --thread ID "FOLLOW-UP"`.
- Run `saber threads export ID --output FILE`.
- Run `saber threads prune --days N --dry-run` before deliberate deletion with `--yes`.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- Populated-thread entries require a completed real model query against a saved connection named `verification`.
- `THREAD_ID` holds the full UUID from that query.

- **Empty list.** Immediately after launch, capture `saber threads list`. It exits `0` and prints `No threads found.`. This remains valid when model credentials are absent.
- **List and show.** After the query, list with `--database verification --limit 10`, then show the full ID. Require database, title, model, original question, answer, and SQL or tool details.
- **Artifacts.** Run `threads artifacts "$THREAD_ID"`. A thread without publications prints `No artifacts found.`. Published output includes publication ID and kind, artifact kind, name, size, URI, and availability.
- **Resume.** Start `threads resume "$THREAD_ID"` through `drive`, send Ctrl+D on the empty editor, and require prior context plus `Goodbye!`. Continue once with root `--thread "$THREAD_ID" "Repeat the prior answer as an integer only."`; then show the thread again and require both turns. Root `--thread` without a question is an expected usage error.
- **HTML export.** Export to `$("$VERIFY_SQLSABER" path "$RUN_ID" evidence)/threads/thread.html`. Require the thread ID and visible transcript text in the saved file.
- **Prune preview.** This needs at least one thread older than the selected age. Copy the thread database and record full IDs plus the eligible count, then run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence threads/prune-dry-run.txt -- uv run saber threads prune --days 1 --dry-run`. Require its count to match the eligible rows. Capture another list and database copy; every ID and count remains unchanged. A store with no eligible row proves only that the command starts, not dry-run safety.
- **Prune deletion.** In a separate isolated run with a genuinely old thread, preview first, then run the same command with `--yes`. Require the eligible ID to disappear from both `threads list` and a copied database while newer IDs remain. Do not edit timestamps to manufacture eligibility. If no old thread exists, record the age prerequisite as unreachable.
- **Automatic retention.** With a thread older than 30 days, complete any query session and require that old ID to disappear while recent IDs remain. This entry needs both an old thread and a working provider credential.
- **Persisted proof.** Copy `THREADS_DB=$("$VERIFY_SQLSABER" path "$RUN_ID" threads-db)` into evidence before cleanup, then run `uv run python - "$EVIDENCE/threads/threads.db"` with `sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)`. Record full IDs, database names, creation times, and message presence; they must match list and show.

## Gotchas

- Thread IDs must be complete UUIDs.
- Automatic resume trusts saved database selectors. Ad hoc paths need an explicit repeated `-d` override.
- `threads show` can mark expired query results or artifacts unavailable. Record that state.
- `--days` must be at least 1.
- `threads prune --yes` also cleans durable results and artifacts. Never test deletion against the user's store.
- Export defaults to the current directory. Always pass an evidence path.
- The HTML embeds transcript data and layout, but fonts, Markdown rendering, and syntax highlighting load from external CDNs. Do not claim the enhanced rendering is fully offline.

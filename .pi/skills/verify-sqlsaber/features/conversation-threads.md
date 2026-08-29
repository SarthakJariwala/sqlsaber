# Conversation threads

Conversation threads let a user find saved query sessions, read complete transcripts, inspect durable artifacts, resume context, export HTML, and prune old history.

## Sub-features

- `threads-list-show` lists retained sessions and renders one transcript by full ID.
- `threads-artifacts` lists durable outputs referenced by a thread.
- `threads-resume` reopens a thread in the terminal UI or continues it with root `--thread`.
- `threads-export` writes a standalone HTML transcript.
- `threads-prune` previews or deletes threads older than a chosen age.

## How to get to it (user POV)

- Finish a single-shot query, then run `saber threads list` and `saber threads show ID`.
- Run `saber threads artifacts ID` to inspect retained output references.
- Run `saber threads resume ID` for the terminal UI or `saber --thread ID "FOLLOW-UP"` for one automated follow-up.
- Run `saber threads export ID --output FILE` to create HTML.
- Run `saber threads prune --days N --dry-run` before deliberate deletion with `--yes`.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY` and a real model-backed query has completed in this run.
- The query transcript contains a full continuation thread ID for database `verification`.
- `THREAD_ID` holds that full ID, and the saved database remains configured.

- **List entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence threads/list.txt -- uv run saber threads list --database verification --limit 10`. The `Threads` table contains `THREAD_ID`, database `verification`, a title, activity time, and model.
- **Show entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence threads/show.txt -- uv run saber threads show "$THREAD_ID"`. The output includes thread metadata, the original user question, assistant answer, and SQL/tool details.
- **Artifacts entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence threads/artifacts.txt -- uv run saber threads artifacts "$THREAD_ID"`. A thread without publications prints `No artifacts found.`. If the query published an artifact, require publication ID, kind, name, size, URI, and no unexpected `unavailable` marker.
- **Interactive resume.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 30 --input $'/exit\r' --input-delay 2 --evidence threads/resume-exit.txt -- uv run saber threads resume "$THREAD_ID"`. It renders prior context, enters the terminal UI, accepts `/exit`, and prints `Goodbye!`.
- **Non-interactive follow-up.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 180 --evidence threads/follow-up.txt -- uv run saber --thread "$THREAD_ID" "Repeat the prior answer as an integer only."`. The command uses saved history, returns a database-backed answer, and keeps the same retained context.
- **HTML export.** Set `EXPORT=$("$VERIFY_SQLSABER" path "$RUN_ID" evidence)/threads/thread.html`, then run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence threads/export.txt -- uv run saber threads export "$THREAD_ID" --output "$EXPORT"`. The HTML remains under evidence, contains `THREAD_ID` and visible transcript text, and opens without external app assets.
- **Prune preview.** Copy the thread database, count `threads` rows, run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence threads/prune-dry-run.txt -- uv run saber threads prune --days 1 --dry-run`, then copy and count again. Counts and thread IDs must match. This proves the preview did not delete state.
- **Persisted proof.** Run `THREADS_DB=$("$VERIFY_SQLSABER" path "$RUN_ID" threads-db)`, copy it into the feature evidence directory, and query the copied `threads` table in read-only mode. Its full IDs, database names, and message presence must match `list` and `show`.

## Gotchas

- Thread IDs must be complete UUIDs. Terminal width should not be used as a reason to copy a truncated value.
- Automatic resume requires saved database selectors. Threads created from an ad hoc path need an explicit repeated `-d` override.
- `threads show` may mark expired query results or artifacts unavailable. Record that state instead of silently omitting it.
- `--dry-run` is proven only when the copied thread database has identical rows before and after.
- `threads prune --yes` also runs durable result and artifact cleanup. Never test deletion against the user's thread store.
- Export output defaults to the current directory. Always pass an evidence path so cleanup cannot remove it.

# Conversation threads

Conversation threads let a user list saved chats, read a transcript, continue a thread in the TUI or with one non-interactive follow-up, and preview prune without deleting.

## Sub-features

- `threads-list-empty` reports no threads on a fresh launch.
- `threads-list` shows IDs after a query has been saved.
- `threads-show` prints metadata and the transcript for one ID.
- `threads-followup` continues a thread with `saber --thread ID "question"`.
- `threads-prune-dry-run` reports what would be deleted without removing threads.

## How to get to it (user POV)

- Run `saber threads list`.
- Run `saber threads show THREAD_ID`.
- Run `saber threads resume THREAD_ID` for an interactive continuation.
- Run `saber --thread THREAD_ID "follow-up question"` for one non-interactive continuation.
- Run `saber threads prune --days 30 --dry-run`.

## Driving it with control-sqlsaber

Preconditions:

- `control-sqlsaber doctor` is clean.
- For `threads-list-empty` only: no query has been run in this isolated HOME yet (true immediately after launch).
- For list/show/follow-up: a thread exists. Create one only via the user path in [one-shot query](./one-shot-query.md) (`cli -- -d verify-sqlite "How many customers are in CA?"`). That requires a provider key. If the key is absent, skip the populated-thread entry points and still run the empty list plus dry-run on empty.

- **Empty list.** Right after launch, run `control-sqlsaber cli --out artifacts/conversation-threads/list-empty.txt -- threads list`. Exit code `0`. Stdout is `No threads found.`
- **Populate (optional).** Drive one-shot query against `verify-sqlite`. Note the printed `saber threads resume` / `--thread` ID if present.
- **List populated.** Run `control-sqlsaber cli --out artifacts/conversation-threads/list.txt -- threads list`. Stdout contains table title `Threads` and the thread ID. Database column includes `verify-sqlite`.
- **Show transcript.** Run `control-sqlsaber cli --out artifacts/conversation-threads/show.txt -- threads show THREAD_ID`. Stdout contains `Thread: THREAD_ID`, `Database: verify-sqlite`, and the original user question.
- **Non-interactive follow-up.** Run `control-sqlsaber cli --out artifacts/conversation-threads/followup.txt -- --thread THREAD_ID "How many orders are pending?"`. Exit code `0`. Stdout contains `Connected to:` and an answer about the pending `Globex` order. A second `threads show THREAD_ID` includes both questions.
- **Dry-run prune.** Run `control-sqlsaber cli --out artifacts/conversation-threads/prune-dry-run.txt -- threads prune --days 0 --dry-run`. Then run `threads list` again to `artifacts/conversation-threads/list-after-dry-run.txt`. Thread IDs from before the dry-run are still present. If the dry-run said it would delete N threads, N must match the listed count, and the files on disk must still contain those IDs.
- **Proof.** Empty-list plus either (populated list, show, unchanged IDs after dry-run) or an explicit skip of populated paths for missing provider keys.

## Gotchas

- Threads are stored under the isolated HOME (`threads.db` from `control-sqlsaber paths`). Listing from the operator's real config is a failed isolation check.
- `saber threads resume ID` starts the TUI. For non-interactive proof use `saber --thread ID "question"`.
- `--thread` requires a query. `saber --thread ID` with no question exits `2`.
- Ad-hoc file queries may not auto-resume from saved connection names. This recipe uses `verify-sqlite`.
- `threads prune` without `--dry-run` deletes. Always pass `--dry-run` in verification unless a recipe is specifically testing deletion, and then only inside this isolated HOME.
- `--days 0` is aggressive; that is why the post-dry-run list is mandatory. Trusting the word `dry-run` without a second list is not proof.

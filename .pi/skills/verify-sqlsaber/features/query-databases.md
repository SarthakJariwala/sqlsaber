# Query databases

Querying lets a user ask a natural-language question against a file or saved connection, inspect generated SQL and results, continue in the terminal UI, and retain the conversation as a thread.

## Sub-features

- `query-single` answers one command-line question and exits.
- `query-stdin` reads a question from piped standard input.
- `query-file` resolves an ad hoc SQLite, DuckDB, or CSV path without saving it first.
- `query-multiple` connects repeated `-d` selectors in one session.
- `query-interactive` opens the terminal UI, accepts follow-up questions, and exits through `/exit` or `/quit`.
- `query-dangerous` keeps writes blocked unless the user passes `--allow-dangerous`.

## How to get to it (user POV)

- Run `saber -d FILE "QUESTION"` for a single answer.
- Pipe a question with `printf '%s\n' "QUESTION" | saber -d FILE`.
- Repeat `-d` to query multiple saved connections or files.
- Run `saber -d FILE` in a terminal to open the interactive UI.
- Add `--thinking` or `--allow-dangerous` when that behavior is deliberate.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY` and names a provider credential present in the environment.
- The run has a model that matches that credential. For `ANTHROPIC_API_KEY`, run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence query/model.txt -- uv run saber models set anthropic:claude-sonnet-4-5-20250929 --thinking-level off`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)` points to the seeded database with three employees and two paid orders.

- **Single-shot entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 180 --evidence query/single.txt -- uv run saber -d "$FIXTURE" "Count all employees. Use the database and report the integer only."`. The transcript shows the fixture connection, SQL execution or tool result, the value `3`, exit code `0`, and a continuation thread ID.
- **Stdin entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 180 --evidence query/stdin.txt -- bash -c 'printf "%s\n" "Count paid orders. Use the database and report the integer only." | uv run saber -d "$1"' _ "$FIXTURE"`. The answer is `2` and exit code is `0`.
- **Saved selector entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence query/register-verification.txt -- uv run saber db add verification --type sqlite --database "$FIXTURE" --no-interactive`, then run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 180 --evidence query/saved-selector.txt -- uv run saber -d verification "List active employee names in alphabetical order."`. The answer contains `Ada` before `Grace` and omits `Linus`.
- **Multiple database entry.** Register the same disposable fixture with `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence query/register-staff.txt -- uv run saber db add staff --type sqlite --database "$FIXTURE" --no-interactive` and `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence query/register-orders.txt -- uv run saber db add orders --type sqlite --database "$FIXTURE" --no-interactive`. Then run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 180 --evidence query/multiple.txt -- uv run saber -d staff -d orders "Report the configured database names available in this session."`. The visible answer names both selectors. This proves connection selection, not a cross-database join.
- **Interactive entry.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 30 --input $'/exit\r' --input-delay 2 --evidence query/interactive-exit.txt -- uv run saber -d "$FIXTURE"`. The terminal UI starts, handles `/exit`, prints `Goodbye!`, and exits `0`.
- **Thread read-back.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --evidence query/threads.txt -- uv run saber threads list`. The table contains the thread IDs printed by the completed questions. Set `THREAD_ID` to each full ID and run `saber threads show "$THREAD_ID"` through the helper for each claimed query and capture its question, SQL/tool result, and answer.
- **Safety entry.** Snapshot the fixture row counts, ask for a write without `--allow-dangerous`, and snapshot them again. The command must refuse or avoid the write, and the before/after counts must match. Test `--allow-dangerous` only in a separate run because it can mutate the fixture.

## Gotchas

- Model output varies. Assert database-backed values and tool results, not prose or token counts.
- A present environment variable may contain an expired credential. Record the provider error and mark model-backed entries unreachable if the first query fails authentication.
- The stdin recipe needs `bash -c` inside the PTY so `saber` sees a non-terminal stdin stream.
- Interactive TUI input uses carriage return. A newline may render in the editor instead of submitting.
- `--allow-dangerous` is real write access, not a preview mode. SQLsaber still blocks `DROP`, `TRUNCATE`, admin operations, and `UPDATE` or `DELETE` without `WHERE`.
- Repeated ad hoc CSV selectors merge views differently from repeated saved database selectors. Report which path the proof used.

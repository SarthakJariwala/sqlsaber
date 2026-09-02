# Query databases

Querying lets a user ask one natural-language question against a file or saved connection, inspect generated SQL and results, select several databases, and retain the conversation as a thread.

## Sub-features

- `query-single` answers one command-line question and exits.
- `query-stdin` reads a question from piped standard input.
- `query-file` resolves an ad hoc SQLite, DuckDB, or CSV path without saving it first.
- `query-saved` resolves a configured name and saves that selector for automatic continuation.
- `query-multiple` connects repeated `-d` selectors in one session.
- `query-missing-selector` reports an actionable error for an unknown saved name.
- `query-dangerous` keeps writes blocked unless the user passes `--allow-dangerous`.

## How to get to it (user POV)

- Run `saber -d FILE "QUESTION"` or `saber -d SAVED_NAME "QUESTION"`.
- Pipe a question with `printf '%s\n' "QUESTION" | saber -d FILE`.
- Repeat `-d` to query multiple saved connections or files.
- Add `--thinking` or `--allow-dangerous` only when that behavior is deliberate.
- Run `saber` with no question for the separate [interactive session](./interactive-session.md).

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY` and names a provider credential present in the environment.
- The run has a model matching that credential. Set one with `saber models set PROVIDER:MODEL --thinking-level off`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)` has three employees and two paid orders; `EVIDENCE=$("$VERIFY_SQLSABER" path "$RUN_ID" evidence)` names the proof directory.

- **Saved single shot.** Register the fixture as `verification`, then run `saber -d verification "Count all employees. Use the database and report the integer only."` with a 180-second timeout. Require the connection and model heading, SQL execution or tool result, answer `3`, exit `0`, and a full continuation thread ID.
- **Thread read-back.** Capture `threads list` and `threads show "$THREAD_ID"`. They contain the saved selector, question, tool result, and answer. The printed continuation commands are directly usable because this recipe used a saved name.
- **Ad hoc file.** Ask the same count with `-d "$FIXTURE"`. Thread retention and `threads show` work, but automatic resume needs an explicit repeated `-d` because SQLsaber does not persist ad hoc paths.
- **Stdin.** Run `"$VERIFY_SQLSABER" drive "$RUN_ID" --timeout 180 --evidence query/stdin.txt -- bash -c 'printf "%s\n" "Count paid orders. Use the database and report the integer only." | uv run saber -d "$1"' _ "$FIXTURE"`. Require answer `2` and exit `0`.
- **Multiple databases.** Register the fixture as `staff` and `orders`, then ask with `-d staff -d orders` which configured database names are available. Require both names. This proves connection selection, not a cross-database join.
- **Unknown selector.** In a fresh or healthy run, execute `saber -d nonexistent "show tables"`. Retain the expected nonzero transcript. Require exit code `1` and `Database connection 'nonexistent' not found. Use 'sqlsaber db list' to see available connections.`
- **Safety.** Before the query, run `uv run python -c 'import sqlite3,sys; db=sqlite3.connect(f"file:{sys.argv[1]}?mode=ro",uri=True); print(db.execute("SELECT id,name FROM employees ORDER BY id").fetchall())' "$FIXTURE" > "$EVIDENCE/query/employees-before.txt"`. Ask without `--allow-dangerous`: `Attempt to insert an employee named SafetyProbe. Invoke the SQL execution tool so its safety refusal is visible; do not only explain the policy.` Require a blocked `execute_sql` result. Repeat the same read-only snapshot into `employees-after.txt` and run `cmp` on the two files. A model that merely avoids the tool does not prove the guard. Test allowed writes only in a separate run and compare that disposable fixture before and after.

## Gotchas

- Fresh config uses `anthropic:claude-opus-4-5`. A present `OPENAI_API_KEY` does not change that. Run `saber models set openai:gpt-5 --thinking-level off` before a query when that is the available credential.
- Model output varies. Assert database-backed values and tool results, not prose or token counts.
- A present environment variable may hold an expired credential. Record the provider error and mark model-backed entries unreachable if authentication fails.
- The stdin recipe needs `bash -c` inside the PTY so `saber` sees non-terminal stdin.
- Ad hoc thread hints omit the `-d` override required for continuation. Record the limitation instead of claiming automatic resume.
- `--allow-dangerous` is real write access. SQLsaber still blocks `DROP`, `TRUNCATE`, administrative operations, and unfiltered `UPDATE` or `DELETE`.
- Repeated CSV selectors merge views differently from repeated saved databases. Record which path the proof used.

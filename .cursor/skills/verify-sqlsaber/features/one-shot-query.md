# One-shot query

One-shot query lets a user ask a natural-language question from the shell, get a streamed answer against the connected database, and exit without opening the TUI.

## Sub-features

- `query-saved` runs a question with `-d verify-sqlite`.
- `query-adhoc-file` runs a question with `-d` pointing at the seed sqlite path.
- `query-connected-banner` prints the connected database and model before the answer.
- `query-thread-hint` prints continue hints when a thread was saved.

## How to get to it (user POV)

- Run `saber -d verify-sqlite "How many customers are in CA?"`.
- Run `saber -d /abs/path/verify.sqlite "How many customers are in CA?"`.
- Pipe the question: `echo "How many customers are in CA?" | saber -d verify-sqlite`.

## Driving it with control-sqlsaber

Preconditions:

- `control-sqlsaber doctor` is clean.
- A provider API key is present in the environment (`ANTHROPIC_API_KEY` for the default model `anthropic:claude-opus-4-5`, or another configured provider). If doctor reports `auth  absent`, skip this feature and record that unmet precondition. Do not mark it verified from unit tests.
- Default database is `verify-sqlite`.

- **Saved connection.** Run `control-sqlsaber cli --out artifacts/one-shot-query/saved.txt -- -d verify-sqlite "How many customers are in CA?"`. Exit code `0`. Stdout contains `**Connected to**: verify-sqlite (SQLite)` and an answer that reports one California customer (`Acme`).
- **Ad-hoc file.** Run `control-sqlsaber cli --out artifacts/one-shot-query/adhoc.txt -- -d "$(control-sqlsaber path SEEDED_DB)" "How many customers live in New York?"`. Exit code `0`. Stdout contains `**Connected to**:` and an answer that reports one New York customer (`Globex`).
- **Unknown saved name.** Run `control-sqlsaber cli --out artifacts/one-shot-query/missing.txt -- -d nonexistent "show tables"`. Exit code `1`. Stderr contains `Database connection 'nonexistent' not found`.
- **Proof.** Keep saved and adhoc transcripts. Both must show the connected banner and a user-visible answer, not only tool-call traces. If a thread ID is printed, `saber threads list` in a follow-up `cli` call contains that ID.

## Gotchas

- This path calls a real model and executes read-only SQL against the seed sqlite. It is not dry-run.
- Without a provider key the process may prompt or fail on auth. That is an unmet precondition, not a query-engine bug.
- `-d NAME` looks up a saved connection. `-d /path/file.sqlite` (or `.csv`, `.duckdb`) is an ad-hoc file and does not need `db add`.
- Stdin with a TTY and no query starts the TUI instead. Always pass the question as an argument in this recipe.
- `--allow-dangerous` is out of scope here. Default is read-only.
- Do not assert on exact model wording. Assert on the banner, exit code, and the countable fact from `customers`.

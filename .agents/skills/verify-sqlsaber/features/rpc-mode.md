# RPC mode

`saber rpc` speaks JSONL on stdin/stdout so an IDE or other app can embed the SQL agent without the terminal UI.

## Sub-features

- `rpc-help` lists flags and repository-backed examples from `saber rpc --help`.
- `rpc-startup-no-database` writes one `startup` JSON error to stdout and exits `1` when no database is configured.
- `rpc-usage` reports `--thread` with `--no-thread` on stderr and exits `2` with empty stdout.
- `rpc-ready` emits a `ready` event for an ad hoc SQLite file, then answers `get_state`, `get_tables`, `get_messages`, and `shutdown` as JSONL on stdout.
- `rpc-keep-open` answers a command while stdin stays open. Closing stdin after every line is not enough; that path can hide a buffered-read deadlock.
- `rpc-abort` accepts a `prompt`, then a `shutdown` (or `abort`) before the model finishes, and emits `agent_end` with `status:"aborted"`.

## How to get to it (user POV)

- Run `saber rpc --help`.
- Run `saber rpc` with no saved database.
- Pipe JSONL: `printf '%s\n' '{"type":"get_state"}' '{"type":"shutdown"}' | saber rpc -d FILE --no-thread`.
- Keep stdin open and write one command at a time (the embedding path).

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- Use `verify-sqlsaber run`, not `drive`. RPC is redirected JSONL. `run` sets the child's stdin to `/dev/null`, so any session that needs commands must wrap a pipe or a keep-open writer in `bash -c` or `python3`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)`.

- **Help.** `run` `uv run saber rpc --help`. Exit `0`. Stdout contains `Examples` and `--no-thread`.
- **No database.** `run` `uv run saber rpc`. Exit `1`. Stdout is one JSON object with `command` `startup`, `success` false, and `sqlsaber db add`. Stderr is empty.
- **Exclusive flags.** `run` `uv run saber rpc --thread abc --no-thread`. Exit `2`. Stderr contains `mutually exclusive`. Stdout is empty.
- **Idle session.** Pipe `get_state`, `get_tables`, `get_messages`, `get_last_assistant_text`, and `shutdown` into `uv run saber rpc -d "$FIXTURE" --no-thread`. Exit `0`. First line is `ready` with `database.name` `verification` and `threadPersistence` false. `get_tables` lists `departments`, `employees`, and `orders`. Stderr is empty.
- **Keep stdin open.** Start `uv run saber rpc -d "$FIXTURE" --no-thread` with a Python `Popen` pipe. Read `ready`. Write one `get_state` line and flush without closing stdin. The `get_state` response must arrive while the pipe is still open. Then `shutdown`.
- **Abort.** Pipe a `prompt` and an immediate `shutdown`. Require `response` `prompt` success, `agent_start` with `promptId`, `agent_end` `aborted`, and `shutdown` success. Do not require a model completion.
- **Completed prompt.** Optional. Keep stdin open after `prompt` until `agent_end`. Needs a working provider credential. If the model never streams, retain the idle and abort transcripts and mark this entry unreachable.

## Gotchas

- `echo "show users" | saber` is a one-shot question, not RPC. RPC is the `rpc` subcommand.
- Construction can prompt for an API key via `getpass`. RPC disables that; a missing key is a `startup` JSON error.
- `verify-sqlsaber run` cannot feed stdin itself. A batch `printf | saber rpc` also sends EOF, so it does not prove the keep-open path.
- `BufferedReader.read(n)` on stdin waits for `n` bytes or EOF. The process must use a single raw read (`read1`) or an interactive client hangs after `ready`.
- Default thinking follows the saved model config. `--no-thinking` is the same flag pair as the root command.

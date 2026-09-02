# Terminal output

SQLsaber renders styled saber-tui blocks to a terminal and plain Markdown when stdout is redirected, while keeping diagnostics on stderr and avoiding ANSI in redirected output.

## Sub-features

- `output-tty` renders readable interactive terminal blocks through `drive`.
- `output-redirected-table` serializes tables as GitHub pipe tables.
- `output-redirected-values` serializes key-value blocks with Markdown labels.
- `output-streams` keeps normal output on stdout, diagnostics on stderr, and records the exit code.
- `output-no-ansi` emits no terminal escape sequences when redirected.

## How to get to it (user POV)

- Run any command in a terminal for styled output.
- Redirect or pipe stdout, for example `saber db list > databases.md`.
- Redirect stderr separately for errors.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- Register the fixture as `verification` so table and key-value commands have content.

- **Redirected table.** Use `verify-sqlsaber run`, not `drive`, for `uv run saber db list`. In the transcript's stdout section require a Markdown header row, separator row, and `verification`. Stderr is empty and exit code is `0`.
- **Redirected values.** Use `run` for `uv run saber models current --agent main`. Stdout contains Markdown labels such as `**Current model**` and `**Thinking**`.
- **Error stream.** Use `run` for an expected usage error such as `uv run saber theme set not-a-real-theme`. Retain the nonzero transcript. The actionable error is in stderr, stdout does not contain it, and the exit code is recorded.
- **No ANSI.** Search all three transcripts for byte `0x1b`; none is present in the command output sections.
- **TTY comparison.** Capture the same `db list` through `drive`. It has the same data, but do not require the Markdown pipe layout.

## Gotchas

- `drive` always allocates a PTY, even when its own output is redirected by the coordinator. Use `run` to test SQLsaber's non-TTY branch.
- Query streaming buffers differently without a TTY. Prove a model-backed redirected query only when a provider credential works.
- The evidence wrapper adds section headings. ANSI and stream assertions apply to command output inside those sections.
- Usage errors can exit nonzero. A retained transcript is valid proof even when the helper returns that code.

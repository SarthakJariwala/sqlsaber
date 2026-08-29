# SQLsaber verification map

This directory tracks SQLsaber behavior that a user can reach from the terminal. Read this index, choose the relevant feature IDs, and record which entry points the proof covers.

## Baseline preconditions

- Run from the repository root after `uv sync --locked` succeeds.
- Create a unique run with `verify-sqlsaber launch "$RUN_ID"`.
- Require `HEALTHY` from `verify-sqlsaber doctor "$RUN_ID"`.
- Use the fixture returned by `verify-sqlsaber path "$RUN_ID" fixture`.
- Keep app state under `.pi/verification/sqlsaber/.state/$RUN_ID/` and proof under `.pi/verification/sqlsaber/$RUN_ID/`.
- Use a different `RUN_ID` for every concurrent agent. Never point the helper at the user's normal config or data directories.
- A model-backed query also requires a provider credential in the environment and a matching model selected inside the isolated run.

## Driving conventions

- Run every `saber` command through `verify-sqlsaber drive` so it gets an isolated PTY and transcript.
- Start a stateful recipe from a new run unless the recipe says otherwise.
- Treat command names, option names, full IDs, headings, and prompt text as stable handles. Ignore colors and terminal coordinates.
- Use `--yes` only after the recipe has identified the exact isolated object to delete.
- Use a second CLI command to read back mutations. Then inspect a copy of the isolated JSON or SQLite file when persistence matters.
- An external database or model provider may introduce network side effects. Prefer the seeded SQLite fixture and the provider already configured for verification.

## Proof and skip reporting

- Capture the action, output, and exit code in the same PTY transcript.
- Capture the resulting state through a second user-facing command.
- For persisted changes, save a read-only JSON or SQLite observation beside the transcripts.
- Name the feature ID and entry point in the proof notes.
- Report skipped entry points with the unmet precondition. Do not claim them through another entry point.
- Query proof must show the visible answer and the SQL or tool result that supports it.
- Keep proof after cleanup and confirm both `launch.txt` and `cleanup.txt` remain.

## Feature entry contract

Each feature file has an H1 title and one user-focused summary. It then uses four H2 sections in this order: `Sub-features`, `How to get to it (user POV)`, `Driving it with verify-sqlsaber`, and `Gotchas`.

## Features

- [Command discovery](./command-discovery.md) covers version output, root help, and command-specific help.
- [Query databases](./query-databases.md) covers single-shot questions, stdin, file selectors, multiple databases, and the terminal UI.
- [Database connections](./database-connections.md) covers adding, listing, testing, editing, selecting, and removing saved connections.
- [Knowledge base](./knowledge-base.md) covers database-scoped add, list, show, search, remove, and clear behavior.
- [Conversation threads](./conversation-threads.md) covers thread listing, transcript display, artifacts, resume, export, and pruning.

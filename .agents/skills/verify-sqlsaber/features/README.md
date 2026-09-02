# SQLsaber verification map

This directory tracks SQLsaber behavior that a user can reach from the terminal. Read this index, choose the relevant feature IDs, and record which entry points the proof covers.

## Baseline preconditions

- Run from the repository root after `uv sync --locked` succeeds.
- Create a unique run with `verify-sqlsaber launch "$RUN_ID"`.
- Require `HEALTHY` from `verify-sqlsaber doctor "$RUN_ID"`.
- Use the fixture returned by `verify-sqlsaber path "$RUN_ID" fixture`.
- Keep app state under `.agents/skills/verify-sqlsaber/artifacts/.state/$RUN_ID/` and proof under `.agents/skills/verify-sqlsaber/artifacts/$RUN_ID/`.
- Use a different `RUN_ID` for every concurrent agent. Never point the helper at the user's normal config or data directories.
- The helper uses `keyring.backends.null.Keyring`, so verification cannot read or write the operator's OS keyring. Driven commands still inherit provider credentials from the coordinator environment.
- A model-backed query requires one of those environment credentials and a matching model selected inside the isolated run.

## Driving conventions

- Run terminal commands through `verify-sqlsaber drive`. Use `verify-sqlsaber run` only when the recipe calls for redirected, non-TTY output.
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
- [First-run onboarding](./first-run-onboarding.md) covers the guided setup route when no database is configured.
- [Database connections](./database-connections.md) covers adding, listing, testing, editing, selecting, and removing saved connections.
- [Authentication](./authentication.md) covers provider setup, status, and stored-key reset.
- [Model configuration](./model-configuration.md) covers model selection, thinking levels, per-agent overrides, current state, and reset.
- [Themes](./themes.md) covers named and interactive theme selection, persistence, and reset.
- [Query databases](./query-databases.md) covers single-shot questions, stdin, file selectors, multiple databases, and write safety.
- [Interactive session](./interactive-session.md) covers the terminal UI, palette, slash commands, and exits.
- [Knowledge base](./knowledge-base.md) covers database-scoped add, list, show, search, remove, clear, and agent retrieval.
- [Conversation threads](./conversation-threads.md) covers empty and populated listing, transcript display, artifacts, resume, export, and pruning.
- [Terminal output](./terminal-output.md) covers redirected Markdown, stream separation, and ANSI-free output.

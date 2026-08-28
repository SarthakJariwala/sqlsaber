# SQLsaber verification map

This directory is the maintained source for verifying user-facing SQLsaber behavior. Read the index before driving the app, then use the matching feature file as the recipe.

## Baseline preconditions

- Run `.cursor/skills/verify-sqlsaber/control-sqlsaber launch` then `doctor`.
- Isolated HOME is `$WORKDIR/home` from `control-sqlsaber paths`. Never use the operator's SQLsaber config dir.
- Seed sqlite is `control-sqlsaber path SEEDED_DB` with tables `customers` and `orders`.
- Saved connection `verify-sqlite` is the default.
- `PYTHON_KEYRING_BACKEND` is `keyring.backends.null.Keyring`.
- Provider API keys are optional. Features that talk to a model say so in their preconditions.
- Put the helper on your command path for the run, or invoke it by repo-relative path.
- Never drive a `saber` process or tmux session this run did not start.

## Driving conventions

- Start every recipe from the baseline state unless its preconditions say otherwise.
- Run non-interactive commands as `control-sqlsaber cli -- <args>`.
- Run interactive chat as `control-sqlsaber tui start` / `send` / `capture` / `stop`.
- Treat every command as literal. Keep connection name `verify-sqlite` and quoted knowledge names unchanged unless the recipe introduces another name.
- Restore baseline after a mutation that would break later recipes (`verify-sqlite` remains the default; extra connections and knowledge entries created by a recipe are removed in that recipe).
- Do not remove proof artifacts during cleanup.

## Proof and skip reporting

- Capture the user action and the resulting state, not only the last command.
- CLI proof includes the command, stdout, stderr, and exit code.
- Mutation proof includes a read-only second view (`db list`, `knowledge show`, config JSON, or sqlite).
- TUI proof includes a pane capture with `DB: verify-sqlite` visible.
- Record the feature ID and entry point used with every artifact.
- Report an unreachable path with the attempted command and the unmet precondition (especially missing provider keys).
- Do not report a skipped entry point as verified through a different path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with control-sqlsaber` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Database connections](./database-connections.md) covers adding, listing, testing, defaulting, and removing saved connections.
- [Knowledge base](./knowledge-base.md) covers adding, listing, showing, searching, and removing per-database knowledge.
- [One-shot query](./one-shot-query.md) covers non-interactive natural-language questions against a saved connection or an ad-hoc file.
- [Interactive session](./interactive-session.md) covers the TUI chat, slash commands, command palette, and exit.
- [Conversation threads](./conversation-threads.md) covers listing, showing, resuming, non-interactive follow-up, and dry-run prune.

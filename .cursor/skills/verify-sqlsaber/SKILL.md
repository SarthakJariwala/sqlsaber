---
name: verify-sqlsaber
description: Drive the SQLsaber CLI and interactive TUI the way a user does — isolated HOME, saber subcommands, tmux for chat. Use when proving database connections, knowledge, one-shot queries, threads, or the interactive session.
---

# Verify SQLsaber

SQLsaber is a terminal SQL assistant (`saber` / `sqlsaber`). The primary surface is the CLI: non-interactive subcommands plus an interactive saber-tui chat. Secondary surfaces (Python SDK, docs site, optional viz/notebook/sandbox plugins) are out of scope for this skill.

There is no long-lived server. Launch installs deps once, creates an isolated `HOME`, and seeds a SQLite file plus a saved connection named `verify-sqlite`. Each drive is a `saber` process (CLI) or a tmux session (TUI) against that HOME.

Never point this skill at the operator's real `~/Library/Application Support/sqlsaber` or `~/.config/sqlsaber`. Two verification HOMs can exist on disk, but this helper allows only one active run. Refuse to drive a saber process you did not start.

## Launch

From the repo root:

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber launch
.cursor/skills/verify-sqlsaber/control-sqlsaber doctor
```

Launch is ready when it prints `ready: saber db list shows verify-sqlite` and doctor prints only `ok` lines.

What launch does:

1. `uv sync` in the repo (dev command from `AGENTS.md`).
2. Creates `$TMPDIR/sqlsaber-verify-$RUN_ID/` with a disposable `HOME`.
3. Loads `.cursor/skills/verify-sqlsaber/seed.sql` into `$WORKDIR/verify.sqlite` (`customers`, `orders`).
4. Runs `saber db add verify-sqlite --type sqlite --database $SEEDED_DB --no-interactive --description "verification sqlite"`.
5. Sets `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` so API keys and DB passwords never touch the operator keychain.
6. Sets `SQLSABER_SKIP_VERSION_CHECK=1` and `SQLSABER_LOG_FILE=$WORKDIR/sqlsaber.log`.

Teardown:

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber cleanup
```

Cleanup deletes the tmux session this run started (if any) and the isolated workdir. It does not delete `.cursor/skills/verify-sqlsaber/artifacts/`.

## Doctor

Run before driving, and again whenever output looks like the operator's real config leaked in:

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber doctor
```

Doctor must report:

- `version` from `saber --version` (dotted `x.y.z`, currently the checkout's package version).
- `help` — `saber --help` contains `SQLsaber` and `SQL assistant`.
- `isolated` — `HOME_DIR` is under `sqlsaber-verify-` and is not the operator home.
- `config` — `database_config.json` exists under that HOME.
- `default_db` — `saber db list` contains `verify-sqlite`.
- `db_test` — `saber db test verify-sqlite` prints `Connection to 'verify-sqlite' successful`.
- `keyring` — null backend.
- `tui` — `stopped` or `running` for session `sqlsaber-verify`.
- `auth` — whether a provider env key is visible (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`). Query and TUI chat need one; db/knowledge/theme/models-current do not.

Any `FAIL` line means do not drive.

Inspect paths with:

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber paths
.cursor/skills/verify-sqlsaber/control-sqlsaber path SEEDED_DB
```

`paths` prints shell-quoted `KEY=value` lines (`eval "$(... paths)"` is safe). Prefer `path SEEDED_DB` when passing one file into `cli`. On macOS config is `$HOME_DIR/Library/Application Support/sqlsaber`. On Linux config is `$HOME_DIR/.config/sqlsaber` and knowledge may be under `$HOME_DIR/.local/share/sqlsaber`. Always take file locations from `path`.

## Drive

Non-interactive commands go through the helper so they inherit the isolated env:

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber cli -- db list
.cursor/skills/verify-sqlsaber/control-sqlsaber cli --out artifacts/example.txt -- db test verify-sqlite
```

`cli --` is followed by arguments to `saber` (no extra `saber` token). Relative `--out` paths resolve from `.cursor/skills/verify-sqlsaber/`. The helper also writes `$out.err` and `$out.exit`.

Stable handles — match these strings, not column layout (Rich wraps tables):

| Action | Command | Observable |
| --- | --- | --- |
| Help | `cli -- --help` | `SQLsaber` and subcommands `auth`, `db`, `knowledge`, `models`, `theme`, `threads` |
| Version | `cli -- --version` | dotted version on stdout, exit `0` |
| Add SQLite | `cli -- db add NAME --type sqlite --database ABS_PATH --no-interactive` | `Successfully added database connection 'NAME'` |
| First connection | (same) | also `Set 'NAME' as default database` |
| List | `cli -- db list` | table title `Database Connections` and the connection name |
| Test | `cli -- db test NAME` | `Testing connection to 'NAME'...` then `Connection to 'NAME' successful` |
| Default | `cli -- db set-default NAME` | `Successfully set 'NAME' as default database` |
| Remove | `cli -- db remove NAME --yes` | `Successfully removed database connection 'NAME'` |
| Knowledge add | `cli -- knowledge add "Name" "Description" --sql SQL --source SRC` | `Knowledge entry added for database 'verify-sqlite'` plus `**ID**:` UUID |
| Knowledge list | `cli -- knowledge list` | `Knowledge Entries for Database: verify-sqlite` or `No knowledge entries found for database 'verify-sqlite'` |
| Knowledge search | `cli -- knowledge search "query"` | `Knowledge Search Results` with the name, or `No knowledge entries matched 'query'` |
| Knowledge show | `cli -- knowledge show UUID` | `**ID**:`, `**Name**:`, `**Description:**` |
| Knowledge remove | `cli -- knowledge remove UUID --yes` | `Knowledge entry removed from database 'verify-sqlite'` |
| Auth status | `cli -- auth status` | `Authentication Status` |
| Models | `cli -- models current` | `Current model:` |
| Threads list | `cli -- threads list` | `No threads found.` or table title `Threads` |
| Threads show | `cli -- threads show THREAD_ID` | `**Thread**: THREAD_ID` |
| One-shot query | `cli -- -d verify-sqlite "natural language"` | `**Connected to**: verify-sqlite (sqlite)` then streamed answer. Needs a provider key. |
| Ad-hoc file | `cli -- -d ABS_SQLITE_OR_CSV "natural language"` | skips saved-connection lookup; still needs a provider key |

Destructive commands require `--yes` when stdin is not a TTY. Do not pass `--force` or `-y`.

Interactive chat is a TUI (`saber-tui`). Start it only through the helper:

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber tui start
.cursor/skills/verify-sqlsaber/control-sqlsaber tui send --literal '/clear'
.cursor/skills/verify-sqlsaber/control-sqlsaber tui send Enter
.cursor/skills/verify-sqlsaber/control-sqlsaber tui capture --path artifacts/interactive-session/pane.txt
.cursor/skills/verify-sqlsaber/control-sqlsaber tui stop
```

Ready when the pane contains `slash commands` and a footer `DB: verify-sqlite (SQLite)`. Typing `/` on an empty editor opens the command palette (`Thinking mode`, `Handoff thread`, `Clear conversation`, `Exit`). Typed slash commands still work: `/clear`, `/thinking`, `/thinking on`, `/thinking off`, `/exit`, `/quit`. Empty editor + `Ctrl+D` exits. `Ctrl+C` interrupts a running query.

A natural-language submit in the TUI calls the configured model. Without a provider key the editor prompt will stall on auth; do not treat that as a product failure of db/knowledge.

## Evidence

Proof artifacts go under `.cursor/skills/verify-sqlsaber/artifacts/<feature-id>/`. Cleanup must not delete that tree.

Standards:

- Drive `saber` through `control-sqlsaber`, not by importing `sqlsaber.cli` or writing `database_config.json` by hand.
- Capture the command output of the action and a second read-only view of the resulting state (`db list`, `knowledge show`, `threads show`, or the TUI pane after submit).
- For mutations, also capture the side-effect file: `database_config.json`, a `sqlite3` read of `knowledge.db`, or `threads.db` — paths from `control-sqlsaber paths`.
- CLI proof is stdout, stderr, and exit code (`$out`, `$out.err`, `$out.exit`).
- TUI proof is a full pane capture that shows the banner or footer identity (`SQLsaber` ASCII block or `DB: verify-sqlite`) plus the action result.
- One-shot query and TUI chat may call a real model provider. That is the user path. Do not stub pydantic-ai. If no provider key is in the environment, record the unmet precondition and skip those entry points; do not mark them verified via unit tests.
- `threads prune --dry-run` must be checked by listing threads before and after and confirming the same IDs remain.

## Cleanup

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber tui stop   # if a TUI was started
.cursor/skills/verify-sqlsaber/control-sqlsaber cleanup
```

Cleanup kills tmux session `sqlsaber-verify` (the session launch/tui start created) and `rm -rf` on the run's `sqlsaber-verify-*` workdir only. It never `pkill saber` and never deletes `artifacts/`.

## Helpers

`control-sqlsaber` is executable. Invoke it with a path from the repo root, or `cd` into the skill directory.

```bash
.cursor/skills/verify-sqlsaber/control-sqlsaber launch
.cursor/skills/verify-sqlsaber/control-sqlsaber doctor
.cursor/skills/verify-sqlsaber/control-sqlsaber cli --out artifacts/database-connections/list.txt -- db list
.cursor/skills/verify-sqlsaber/control-sqlsaber path SEEDED_DB
.cursor/skills/verify-sqlsaber/control-sqlsaber paths
.cursor/skills/verify-sqlsaber/control-sqlsaber tui start
.cursor/skills/verify-sqlsaber/control-sqlsaber cleanup
```

`seed.sql` is verification scaffolding. Launch applies it; cleanup removes the resulting sqlite file with the workdir.

Read `features/README.md` before choosing what to drive.

---
name: verify-sqlsaber
description: Launch and verify SQLsaber through its CLI and terminal UI. Use it to prove command startup, database setup, natural-language queries, knowledge entries, threads, and other user-visible terminal behavior.
---

# Verify SQLsaber

SQLsaber's primary interface is the `saber` CLI. Running `saber` without a question opens its terminal UI. The Python SDK is a secondary interface and is outside this skill's default scope.

Every run gets its own home, XDG directories, SQLite fixture, logs, and evidence directory. Two runs can operate at once when they use different IDs. Never run verification commands against the developer's normal home directory, and never reuse a `RUN_ID` that is still active.

Read [`features/README.md`](features/README.md) before choosing a recipe.

## Launch

There is no server to leave running. Launch installs the locked checkout once, creates an isolated run, seeds a three-employee SQLite database, and checks that the CLI version matches `pyproject.toml`.

From the repository root:

```bash
export RUN_ID="sqlsaber-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export VERIFY_SQLSABER=".pi/skills/verify-sqlsaber/bin/verify-sqlsaber"
"$VERIFY_SQLSABER" launch "$RUN_ID"
```

Readiness is the line `READY SQLsaber <version>`. The command also prints the fixture and evidence paths. Start each `saber` invocation with the `drive` helper below. It gives that process a fresh PTY inside the isolated run.

Teardown is:

```bash
"$VERIFY_SQLSABER" cleanup "$RUN_ID"
```

Run cleanup after failed attempts too.

## Doctor

Run this read-only check whenever a command behaves unexpectedly:

```bash
"$VERIFY_SQLSABER" doctor "$RUN_ID"
```

Require `HEALTHY`. Doctor checks the checkout revision, CLI version, seeded fixture, config/data/log paths, and whether another process is driving this run. It reports which model credential environment variables are present without printing their values. SQLsaber has no port or long-lived process to inspect.

`doctor.txt` is written to `.pi/verification/sqlsaber/$RUN_ID/`. A present credential is not proof that the remote provider accepts it. The first real query establishes that.

## Drive

Resolve scratch paths through the helper rather than guessing platform-specific `platformdirs` locations:

```bash
FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)
EVIDENCE=$("$VERIFY_SQLSABER" path "$RUN_ID" evidence)
DB_CONFIG=$("$VERIFY_SQLSABER" path "$RUN_ID" database-config)
```

Drive a command in its own 120 by 40 PTY and save a readable transcript:

```bash
"$VERIFY_SQLSABER" drive "$RUN_ID" \
  --evidence database-connections/add.txt \
  -- uv run saber db add verification \
  --type sqlite --database "$FIXTURE" --no-interactive

"$VERIFY_SQLSABER" drive "$RUN_ID" \
  --evidence database-connections/list.txt \
  -- uv run saber db list
```

The transcript records the command, terminal output, and exit code. Use `--timeout 180` before `--` for model-backed queries. For a known short interactive action, pass literal PTY input with `--input` and an optional `--input-delay`. Example:

```bash
"$VERIFY_SQLSABER" drive "$RUN_ID" \
  --evidence interactive/exit.txt \
  --timeout 30 --input $'/exit\r' --input-delay 2 \
  -- uv run saber -d "$FIXTURE"
```

Do not use fixed input delays for a model response. Use single-shot mode for model-backed proof. The helper's fixed input option is only for deterministic prompt or exit actions.

Use the exact commands and expected text in the selected feature file. Stable handles here are command names, flags, printed headings such as `Database Connections`, full IDs, and prompt strings. Do not assert terminal coordinates or color codes.

## Evidence

Proof belongs under `.pi/verification/sqlsaber/$RUN_ID/`. The helper creates that directory and never removes it. Put each feature in its own subdirectory.

A valid proof:

- drives `uv run saber` through the PTY helper, not a Python function or test fixture;
- captures the user command and its immediate output in one transcript;
- captures a second user-facing read after a mutation, such as `saber db list`, `saber knowledge show`, or `saber threads show`;
- copies or queries the isolated persisted file when the feature has a side effect;
- records the feature ID and exact entry point in `notes.txt` when only part of a feature map was exercised;
- retains stderr and the exit code for expected failures.

For database registration, copy the persisted config after the CLI read-back:

```bash
mkdir -p "$EVIDENCE/database-connections"
cp "$DB_CONFIG" "$EVIDENCE/database-connections/database_config.json"
```

For knowledge and thread state, use the paths returned by `path knowledge-db` and `path threads-db`. Query copied SQLite files in read-only mode. Do not edit them.

Natural-language proof needs a real configured provider and the seeded database. Capture the question, generated SQL/tool output, visible answer, exit code, and the saved thread from `saber threads list` or `show`. Model test doubles do not prove the CLI. Mocks are acceptable only at an existing production plugin or provider boundary, and the proof must name that boundary.

`--allow-dangerous` is not a dry-run. It permits writes within SQLsaber's safety rules. Use only the disposable fixture, then compare the fixture before and after. Thread pruning's `--dry-run` must be backed by a before/after count from the copied thread database, not by the label alone.

## Cleanup

Clean only the run you created:

```bash
"$VERIFY_SQLSABER" cleanup "$RUN_ID"
test ! -e ".pi/verification/sqlsaber/.state/$RUN_ID"
test -f ".pi/verification/sqlsaber/$RUN_ID/launch.txt"
test -f ".pi/verification/sqlsaber/$RUN_ID/cleanup.txt"
```

The helper records the exact active PID while a PTY is running. Cleanup will only signal that PID when its environment carries this run's marker. It never kills by process name. It removes the isolated home, fixture, app logs, and PID record. It preserves all proof artifacts.

Do not remove `.pi/verification/sqlsaber/$RUN_ID/` during cleanup. If you created exports as part of a feature, write them inside that evidence directory before teardown.

## Helpers

`.pi/skills/verify-sqlsaber/bin/verify-sqlsaber` is executable and has five commands:

```text
verify-sqlsaber launch RUN_ID
verify-sqlsaber doctor RUN_ID
verify-sqlsaber drive RUN_ID --evidence RELATIVE_PATH [--timeout SECONDS] [--input TEXT] -- COMMAND...
verify-sqlsaber path RUN_ID {state,evidence,fixture,database-config,knowledge-db,threads-db,log}
verify-sqlsaber cleanup RUN_ID
```

Run it only from this checkout. `launch` refuses an existing state or evidence directory. This prevents two agents from sharing a run and prevents old artifacts from mixing with new proof.

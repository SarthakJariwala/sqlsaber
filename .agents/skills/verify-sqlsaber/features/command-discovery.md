# Command discovery

Command discovery lets a user identify the installed SQLsaber version, see the root command families, and inspect the syntax for one command without configuring a database or model.

## Sub-features

- `discovery-version` prints the checkout's application version and exits successfully.
- `discovery-root-help` lists the `auth`, `db`, `knowledge`, `models`, `theme`, and `threads` commands.
- `discovery-command-help` shows parameters, options, and repository-backed examples for a selected command.
- `discovery-read-only` does not create database, model, knowledge, or thread files.

## How to get to it (user POV)

- Run `saber --version`.
- Run `saber --help` or `saber -h`.
- Run `saber <command> --help`, such as `saber db add --help`.

## Driving it with verify-sqlsaber

Preconditions:

- The run is launched and doctor reports `HEALTHY`.
- No SQLsaber feature command has run in this isolated home.

- **Version entry.** Run `.agents/skills/verify-sqlsaber/bin/verify-sqlsaber drive "$RUN_ID" --evidence command-discovery/version.txt -- uv run saber --version`. Exit code `0` and terminal output are the version in `pyproject.toml`.
- **Root help entry.** Run `.agents/skills/verify-sqlsaber/bin/verify-sqlsaber drive "$RUN_ID" --evidence command-discovery/root-help.txt -- uv run saber --help`. Exit code `0`, the heading contains `Commands`, and all six command families appear.
- **Short help entry.** Run `.agents/skills/verify-sqlsaber/bin/verify-sqlsaber drive "$RUN_ID" --evidence command-discovery/short-help.txt -- uv run saber -h`. It reaches the same root help and exits `0`.
- **Command help entry.** Run `.agents/skills/verify-sqlsaber/bin/verify-sqlsaber drive "$RUN_ID" --evidence command-discovery/db-add-help.txt -- uv run saber db add --help`. The output names `NAME`, `--type`, `--database`, `--no-interactive`, and the SQLite example.
- **Read-only proof.** Run `STATE=$(.agents/skills/verify-sqlsaber/bin/verify-sqlsaber path "$RUN_ID" state)` and `find "$STATE/home/config" "$STATE/home/data" -type f -print | sort > ".agents/skills/verify-sqlsaber/artifacts/$RUN_ID/command-discovery/app-state-files.txt"`. The file is empty. The app log may exist at the separate path returned by `path log`, and `uv run` may populate `home/cache`; neither is user configuration.

## Gotchas

- Help intentionally initializes config directories during imports. Assert that no config files exist, not that the home directory has no directories.
- Run through the checkout with `uv run saber`. A globally installed `saber` may have a different version and command set.
- Cyclopts sends usage errors to stderr, but normal help and version output use stdout.
- The root command uses the displayed application name `sqlsaber` even when the executable is `saber`.

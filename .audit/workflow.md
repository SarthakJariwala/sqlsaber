# Workflow: saber-tui-only rendering

Designed after Phase A. Each unit ends in a check. Do not start the next unit until the current one is verified.

## Unit 0. Harness (done, currently red)

`scripts/assert_no_legacy_renderers.py` lists Rich, Questionary, and `create_console` uses. Baseline: 134 hits, exit 1. Evidence: `.audit/baseline-legacy-imports.txt`.

## Unit 1. Architect (this phase)

Arena of four isolated sketches. Synthesize one design package into `.audit/sketch/`. Implementation fills that sketch. If fill-in shows repeated friction, scrap and re-arena.

## Unit 2. Theme + printer scaffold

Lift ANSI styles out of `tui_chat.py`. Add a printer that writes `Component.render(width)` to a stream, with a pipe mode that emits unstyled markdown. Tests capture StringIO. `ChatApp` and CLI both depend on this module. `uv run pytest tests/test_cli/test_printer.py` (new). `--help` still under 0.5s.

## Unit 3. Prompts (riskiest unknown)

Replace questionary with saber-tui `Input` / `SelectList`. Standalone short-lived TUI for CLI commands. Overlay on a running `ChatApp` when the REPL needs a confirm. Non-TTY still requires `--yes` and never opens a TUI. Prove with unit tests that drive `handle_input`, not a real tty if we can inject a Terminal fake. Keep `Prompter` as the application boundary.

## Unit 4. One-shot streaming without Rich Live

Port `StreamingQueryHandler` onto the printer + `Markdown` component. TTY may reprint the live markdown block. Pipes write source markdown. Existing streaming tests retargeted. No `rich.live`.

## Unit 5. DisplayManager and tool specs

`DisplayManager` takes the printer, not Console. `SpecRenderer` and SQL tools emit markdown once. Delete `render_result(console)` / `is_terminal` branches. Tools that need images use `ToolResultTUI`. HTML methods stay. `env -u FORCE_COLOR uv run pytest tests/test_tools/test_display.py tests/test_database/test_schema_display.py`.

## Unit 6. CLI command modules

`db`, `models`, `theme`, `threads`, `knowledge`, `auth`, `onboarding`, `slash_commands`, `update_check` print through the printer. Stable handles in `verify-sqlsaber` must still match. `env -u FORCE_COLOR uv run pytest tests/test_cli`.

## Unit 7. Plugins

Notebook drops Rich `Rule`/`Markdown`/`Console` and keeps `render_result_tui` as the only terminal path. Viz writes plotext ANSI through the printer's ANSI block, stripped when piped. Plugin tests green.

## Unit 8. Delete the old stack

Remove `create_console`, `ChatConsole`, `append_rich`, `RichCapture`, Rich theme, questionary, direct deps. Ruff banned-api for `rich` and `questionary`. Harness exits 0. `uv lock` no longer lists questionary as a sqlsaber requirement. Rich may remain transitive via cyclopts.

## Unit 9. Product proof

`control-sqlsaber launch` + doctor. Drive database-connections, knowledge-base, interactive-session. One-shot query if a provider key is present, otherwise record the skip. Capture artifacts under `.cursor/skills/verify-sqlsaber/artifacts/`.

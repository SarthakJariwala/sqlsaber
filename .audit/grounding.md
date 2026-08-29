# How rendering works today (grounding for the redesign)

## Overview

SQLsaber has two user surfaces that print:

1. Interactive chat. A persistent saber-tui `TUI` owned by `ChatApp` in `src/sqlsaber/cli/tui_chat.py`.
2. Everything else. A global Rich `Console` from `create_console()` in `src/sqlsaber/theme/manager.py`, used by cyclopts command modules, onboarding, one-shot streaming, and tool display.

Questionary sits behind `Prompter` in `src/sqlsaber/application/prompts.py` (async and sync wrappers) plus a few direct `questionary.select` / `.text` / `.confirm` calls.

The 2026 saber-tui migration (`ce1ebbc`) and native markdown streaming (`0c92d07`) already moved the REPL onto saber-tui components. They left a Rich bridge: `ChatConsole(Console)` and `ChatApp.append_rich`, so DisplayManager and tools still speak Rich.

## Key types

- `saber_tui.tui.Component`: `render(width: int) -> list[str]` (ANSI lines).
- `saber_tui.tui.TUI`: container plus input loop. `start()` / `stop()` own the terminal.
- `ChatApp`: chat shell. `append_markdown`, `append_panel`, `append_image`, `append_user_message`. Still has `append_rich`.
- `ChatConsole`: subclasses Rich Console; `print` forwards to `append_rich`.
- `RichCapture`: StringIO Console used to snapshot Rich output into `_AnsiBlock`.
- `_TUITheme`: ANSI style callables plus `MarkdownTheme` with pygments `highlight_code`. Lives inside `tui_chat.py`.
- `ThemeManager`: pygments style plus Rich `Theme` role strings like `"bold $primary"`. `create_console(**kwargs)` applies that theme.
- `DisplayManager`: all CLI formatting. Owns `LiveMarkdownRenderer` (Rich Live + Rich Markdown).
- `Tool` (`tools/base.py`): `render_result(console)`, `render_result_tui(tui)`, HTML. `ToolResultTUI` is the native surface (`append_panel` / `append_markdown` / `append_image`).
- `SpecRenderer` (`tools/display.py`): data-driven tool output. Dual path: Rich Table/Panel/Syntax when `console.is_terminal`, else markdown/tabulate.
- `Prompter`: `text`, `select`, `confirm`, `path`. Implemented with questionary.
- `StreamingQueryHandler`: one-shot path, Rich Live.
- `TUIStreamingQueryHandler`: interactive path, prefers `render_*_tui`, falls back to `append_rich(DisplayManager)`.

## Flow

Interactive:

`commands.query` with no query text → `InteractiveSession.run` → `build_chat_app` → `TUI.start` → `TUIStreamingQueryHandler`. Assistant text becomes `saber_tui.components.Markdown.set_text`. Tools that implement `render_result_tui` (notebook) write components. SQL results often still go through `append_rich` + DisplayManager.

One-shot:

`commands.query("...")` → `StreamingQueryHandler` → `DisplayManager.live` (Rich Live Markdown) → `DisplayManager.show_tool_*` → Rich or markdown depending on `console.is_terminal`. Session summary is Rich Text. Continue hints are Rich markup.

CLI subcommands (`db`, `models`, `theme`, `threads`, `knowledge`, `auth`):

Module-level `console = create_console()`. Lists are Rich Tables. Messages use Rich markup. Interactive forms use questionary. Piped stdin uses `--yes` / `--no-interactive` and markdown-ish prints.

Onboarding:

`welcome_screen` prints a Rich Panel ASCII banner, then `AsyncPrompter` for database and auth.

## saber-tui public API that matters

Installed 0.6.0 at `.venv/lib/python3.12/site-packages/saber_tui`.

Present: `TUI`, `Terminal`, `Markdown` (+ `MarkdownTheme`, pygments hook), `Box`, `Text`, `Image`, `Input`, `SelectList` + `SelectItem`, `Editor`, `Loader`, `CancellableLoader`, `SettingsList`, `Spacer`, `AutocompleteProvider`, `strip_ansi`, `visible_width`, `wrap_text_with_ansi`.

Absent: Console, print helper, Table, Panel, Live, Syntax, Questionary-style standalone `ask()`.

A component can be printed without a TUI by calling `component.render(width)` and writing the lines. That is the non-interactive path.

## Dual paths to delete

Every tool renderer currently branches on `console.is_terminal`. TTY gets Rich widgets. Pipes get markdown. Interactive TUI already wants markdown via `render_result_markdown` / `render_result_tui`. First-principles target: produce markdown (or native components) once. TTY runs it through themed `Markdown.render`. Pipes write the source markdown with no ANSI.

Viz is special: plotext already returns ANSI. TTY can wrap that in `_AnsiBlock` / `Text`. Pipes strip ANSI. No Rich `Text.from_ansi`.

## Prompt sites

- `application/prompts.py`: the ABC and both questionary backends.
- `cli/safety.py`: `questionary.confirm` for destructive commands.
- `cli/theme.py`: `questionary.select` when theme name omitted.
- `cli/database.py`: `questionary.text` for exclude-schemas when flags omitted.
- `cli/auth.py`: `questionary.select` for provider.
- `cli/models.py` / `application/model_selection.py`: `Choice` type from questionary.
- Onboarding / db_setup / auth_setup: `AsyncPrompter`.

Command palette in ChatApp already uses saber-tui `SelectList` as an overlay. That is the prompt pattern to reuse inside the TUI. Outside the TUI, a short-lived `TUI` with `Input` or `SelectList` is the replacement for questionary. Non-TTY must keep refusing interactive confirm and requiring `--yes`.

## Module map (production)

CLI print: `cli/commands.py`, `display.py`, `streaming.py`, `buffered_streaming.py`, `tui_streaming.py`, `tui_chat.py`, `interactive.py`, `threads.py`, `database.py`, `models.py`, `knowledge.py`, `auth.py`, `theme.py`, `onboarding.py`, `slash_commands.py`, `safety.py`, `update_check.py`, `html_export.py` (DisplayManager for HTML only).

Theme: `theme/manager.py`, `theme/__init__.py`.

Tools: `tools/base.py`, `tools/display.py`, `tools/sql_tools.py`.

App: `application/prompts.py`, `auth_setup.py`, `db_setup.py`, `model_selection.py`, `config/api_keys.py`.

Plugins: `plugins/notebook/src/sqlsaber_notebook/capability.py`, `plugins/viz/src/sqlsaber_viz/tools.py`.

## Constraints the design must honor

- Python 3.12+ modern type hints. Relative imports inside packages.
- CLI `--help` startup under 0.5s. Do not import pydantic_ai or saber-tui Markdown at `commands.py` import time if that pulls heavy stacks. Cyclopts help may still import Rich internally.
- Do not keep a Rich Console facade "temporarily".
- HTML export (`render_result_html`) is not terminal rendering. It can stay. It must not depend on Console.
- `FORCE_COLOR` currently breaks Rich `is_terminal` tests. After the migration those tests should not care about Rich.
- `verify-sqlsaber` stable handles (exact strings like `Database Connections`, `Successfully added database connection`) must remain.
- Theme roles (primary, accent, success, warning, error, info, muted, table.header, panel.border.*) already exist. Lift ANSI callables out of `tui_chat.py` so CLI and TUI share them.
- `tabulate` may remain as a markdown-table builder. It is not a printer.

## Cyclopts

`cyclopts` 4.11 depends on `rich` and `rich-rst`. `--help` will still use Rich unless we replace the CLI framework. That is out of scope. sqlsaber code must not import Rich. After drop, Rich stays only as a transitive cyclopts dependency.

## Extra constraints from the how-explorers

- saber-tui Markdown tables are box-drawing, not GitHub pipes. Piped agent-friendly output should keep GitHub GFM. TTY can render GFM through `Markdown`.
- `SelectItem.value` is `str`. Thinking-level `Choice` values today include `ThinkingLevel` enums. Stringify at the Prompter boundary.
- `SyncPrompter` is never constructed. Delete it with questionary.
- `BufferedStreamingHandler` has no callers. Delete it.
- `ChatConsole` only overrides `print` and `print_json`. Do not preserve that subclass.
- `RichCapture` always uses `force_terminal=True`, so TUI-captured tools take the Rich Table branch today. After the rewrite, tools must not have that branch.
- Core SQL tools have no `render_*_tui`. SQL results already have `render_result_markdown`. Viz has none. Notebook already has a native TUI path.
- Welcome banner and slash-command feedback still go through Rich capture inside the TUI.
- `threads resume` dumps a Rich transcript to stdout, then starts an empty TUI. Replay should land in `ChatApp` instead.
- Passwords use `getpass`, not questionary. Keep `getpass` until saber-tui grows a masked Input. It is not a printer.
- `confirm_action` is the only well-gated TTY check. Onboarding, `db add`, `models set` with no MODEL, `db exclude`, and `getpass` for API keys are unguarded.
- Command palette already uses `SettingsList`. That is the in-TUI prompt pattern. ChatApp does not use `TUI.show_overlay`.
- Update-check can `console.print` on stdout while the TUI owns the terminal. Route notices through `ChatApp` or drop them during TUI.
- HTML export builds a throwaway Console only to construct DisplayManager. After DisplayManager drops Console, HTML stays.
- Module-level `create_console()` runs at import in many CLI modules. That is a startup-cost leak. Printer construction must be lazy.

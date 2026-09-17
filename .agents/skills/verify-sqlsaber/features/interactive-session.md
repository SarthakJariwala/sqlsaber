# Interactive session

The interactive session is the full-screen chat started by `saber` with no question. It shows database and model identity, accepts questions and local slash commands, and exits back to the shell.

## Sub-features

- `interactive-open` shows `Welcome to SQLsaber!`, the chat editor, slash-command hint, and database footer.
- `interactive-palette` opens settings with `/` on an empty editor.
- `interactive-clear-thinking` clears history and changes session-only thinking state.
- `interactive-handoff` drafts a goal that can start a new thread with current context.
- `interactive-exit` accepts `/exit`, `/quit`, bare `exit` or `quit`, and Ctrl+D on an empty editor.
- `interactive-interrupt` uses Ctrl+C to cancel a running query.

## How to get to it (user POV)

- Run `saber -d FILE` or `saber -d SAVED_NAME` with no question.
- Type `/` on an empty prompt to open the palette. The first page lists session commands (`Thinking mode`, `Handoff thread`, `Clear conversation`, `Exit`, `Command help`) and then management commands starting with `/plugins list`.
- Type `/clear`, `/thinking`, `/handoff GOAL`, `/help`, `/exit`, or `/quit`. Management families from the CLI (`/plugins`, `/auth`, `/db`, `/knowledge`, `/models`, `/theme`, `/threads`) work as slash commands too.
- Press Ctrl+C during a query or Ctrl+D on an empty editor.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)` is available.
- TUI startup paints the editor before constructing the agent. A matching provider key is required only when the session binds (a submitted question, thinking change, or handoff). A harmless placeholder environment value is sufficient only for local controls that never bind.

- **Open, palette, clear, and exit.** Start `uv run saber -d "$FIXTURE"` through `drive` with a placeholder `OPENAI_API_KEY` when the configured model is OpenAI (the fresh default is `openai:gpt-5.6-sol`). Use `--timeout 40 --input-sequence '[[8, "/"], [11, "\u001b[B\u001b[B\r"], [15, "\u0004"]]'`. Two down arrows select `Clear conversation` (the third palette row). Require `Welcome to SQLsaber!`, `slash commands`, `table name completions`, `DB: verification` without a type suffix, palette labels including `Thinking mode`, `Command help`, and `/plugins list`, `Conversation history cleared.`, and `Goodbye!`. Do not require the old ASCII `SQLSABER` banner or `DB: verification (SQLite)` on this unbound path.
- **Thinking.** In a fresh drive, open the palette on `Thinking mode` (first row). Right-arrow cycles the value (`medium` → `high`), then Enter applies it. Applying thinking binds the session; the footer then includes `(SQLite)` and `Thinking: high`. Do not paste `/thinking off` as one line; a leading `/` still opens the palette and leftover text can submit as a query.
- **Exit aliases.** In separate fresh drives where needed, submit bare `exit` or `quit`, or send Ctrl+D on an empty editor after the editor is ready. Require `Goodbye!`.
- **Interrupt.** After the editor is ready, submit a short question and send Ctrl+C while `Crunching data...` is visible, then Ctrl+D. Require `Query interrupted` and `Goodbye!`. Do not wait for a model completion.
- **Handoff.** Palette row `Handoff thread` fills `/handoff `. Submitting a goal drafts a prompt (`Edit the handoff draft and press Enter to start a new thread`). Confirming that draft calls the model and is not a fixed-delay path. If the drive stops at the draft, record that limit; do not invent a new thread ID.

## Gotchas

- A credential for the wrong provider does not satisfy a bind. The default model is `openai:gpt-5.6-sol`. A present `ANTHROPIC_API_KEY` does not match that default.
- `/` opens the palette only when the editor is empty. Bytewise automated typing that starts with `/` also opens it. Use palette keys for deterministic harness proof.
- Delays of 2/4/6 seconds fire before `uv run saber` shows the editor. Use 8/11/15 seconds from process start.
- Fixed input delays are safe for local controls, not for model responses.
- `/handoff` invokes the model after the draft is confirmed. `/clear`, palette open, and exit do not bind. Changing thinking from the palette does bind.
- An ad-hoc `-d "$FIXTURE"` footer uses the file stem (`DB: verification`) until bind; after bind it includes the type (`DB: verification (SQLite)`). A saved name prints that name, and the type may appear in lowercase (`sqlite`) on the post-onboarding path.
- If the sandbox plugin is installed and `provider` is unset, bind and query warn `sandbox.provider is required. Run: saber plugins setup sandbox`. That is plugin setup, not a TUI failure.
- The helper strips ANSI. Assert visible labels and messages, not screen coordinates or colors.

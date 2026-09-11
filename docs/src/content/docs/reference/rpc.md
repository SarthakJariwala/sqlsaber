---
title: RPC mode
description: "Headless JSONL protocol for embedding SQLsaber in IDEs and other applications."
---

`saber rpc` is a headless JSONL protocol so IDEs and other applications can embed the SQL agent. It is the SQLsaber analogue of `pi --mode rpc`. One process owns one conversation. Commands arrive on stdin. Responses and events leave on stdout. Nothing else is written to stdout.

The default `saber` command still treats non-TTY stdin as the natural-language question (`echo "show users" | saber`). RPC is a subcommand so that path stays unchanged.

```bash
saber rpc                                   # default database, conversation saved as a thread
saber rpc -d analytics                      # a saved connection
saber rpc -d sales -d analytics             # several databases in one session
saber rpc -d ./orders.csv --no-thread       # ad-hoc file, nothing persisted
saber rpc --thread 4f1c…                    # resume a saved thread
saber rpc --allow-dangerous --thinking      # writes allowed, extended reasoning on
```

| Flag | Meaning |
| --- | --- |
| `-d, --database` | Saved connection name, file (CSV/Parquet/SQLite/DuckDB), or DSN. Repeatable. Default connection if omitted. |
| `--thread ID` | Resume a saved thread. History is loaded before `ready`. Mutually exclusive with `--no-thread`. |
| `--no-thread` | Do not persist the conversation. `threadId` stays `null`. |
| `--thinking` / `--no-thinking` | Initial reasoning setting. `set_thinking_level` changes it at runtime. |
| `--allow-dangerous` | Allow INSERT/UPDATE/DELETE and restricted DDL, same scope as the interactive CLI. |
| `--csv-tool-results` | Model-facing SQL results as CSV instead of JSON. |
| `--system-prompt TEXT_OR_PATH` | Replace the built-in prompt. |

Process rules:

- stdout carries protocol lines only. Diagnostics go to stderr. Logs go to the usual log file (`SQLSABER_LOG_FILE`).
- RPC never starts the onboarding TUI and never checks for updates. If no database is configured it emits one JSON line and exits 1.
- Exit codes: `0` after `shutdown` or stdin EOF; `1` startup failure (one `startup` response line on stdout); `2` usage error (stderr, from the CLI parser before the protocol starts).

## Framing

- One JSON object per line, `\n` terminated. A trailing `\r` is stripped. Blank lines are ignored.
- Split on `\n` only. `U+2028` / `U+2029` are legal inside JSON strings and are not record separators.
- UTF-8 in both directions. Input lines longer than 1 MiB are rejected with one `parse` error; the rest of that line is discarded.
- Every command may carry `"id"` (string or integer). The single response to that command echoes it. Events never carry `id`.
- A line that is not a JSON object with a string `type` yields `{"type":"response","command":"parse","success":false,"error":"…"}`.

## Lifecycle

The first stdout line is the `ready` event, or a startup failure after which the process exits 1:

```json
{"type":"ready","protocolVersion":1,"state":"idle","database":{"name":"analytics","type":"PostgreSQL","names":["analytics"]},"model":{"name":"claude-sonnet-4-5","id":"anthropic:claude-sonnet-4-5"},"thinkingLevel":"off","thinkingLevels":["off","minimal","low","medium","high","maximum"],"dangerousMode":false,"csvToolResults":false,"threadId":null,"threadPersistence":true,"messageCount":0}
```

```json
{"type":"response","command":"startup","success":false,"error":"No database connections configured. Use 'sqlsaber db add <name>' to add one."}
```

Each command produces exactly one `response`, in command order. `prompt` additionally streams events. The stream always ends with `agent_end`. Finish with `{"type":"shutdown"}` or close stdin. `SIGTERM` / `SIGINT` take the same graceful path.

While `running`, these commands are accepted: `abort`, `get_state`, `get_messages`, `get_last_assistant_text`, `get_query_result`, `get_artifact`, `shutdown`. These are rejected because they would race the running query: `prompt`, `new_session`, `set_thinking_level`, `reload_model`, `get_tables`. There is no steer/follow-up queue. A `prompt` carrying Pi's `streamingBehavior` is rejected whether or not a query is running.

`get_messages` returns committed turns only. A turn is committed when its `agent_end` has `status:"completed"`. Aborted or failed turns leave no trace in history or in the saved thread.

## Commands

### `prompt`

Start one agent run. The response means "accepted". The answer arrives as events. Only one run at a time.

```json
{"id":"q1","type":"prompt","message":"Top 5 customers by revenue this quarter"}
{"id":"q1","type":"response","command":"prompt","success":true}
```

Failures after acceptance (model API error, database error) are reported as `agent_end` with `status:"error"`, never as a second response.

### `abort`

Stop the running query. `agent_end` with `status:"aborted"` is emitted before the abort response. Idempotent: when idle it responds immediately with `aborted:false`. Stdin is still read while a run unwinds, so `get_state` can land between `abort` and `agent_end`.

### `new_session`

Clear history and start a fresh thread. Idle only.

### `get_state`

Same object as `ready` minus `protocolVersion`. Allowed while running (`state:"running"`).

### `get_messages`

Committed conversation as transcript messages. Not a pydantic-ai dump.

### `get_last_assistant_text`

`{"text": "…"}` or `{"text": null}` when there is no assistant message yet.

### `set_thinking_level`

Levels are `off`, `minimal`, `low`, `medium`, `high`, `maximum`. `off` disables reasoning. It is not an alias for medium. Idle only.

### `reload_model`

Re-read the saved model configuration and apply it to subsequent queries, keeping history. Idle only.

### `get_tables`

Tables across connected databases, for completion and schema panels. Idle only: listing tables hits live database connections.

### `get_query_result`

Fetch stored `execute_sql` rows by the `queryResult.id` seen on `tool_execution_end`, `agent_end`, or a `toolResult` message. Allowed while running. Page with `offset` and `limit` (default 500, max 5000) until `hasMore` is false.

```json
{"type":"get_query_result","resultId":"qr_01J9…","offset":0,"limit":500}
{"type":"response","command":"get_query_result","success":true,"data":{"result":{"id":"qr_01J9…","rowCount":5,"columns":["customer","revenue"]},"offset":0,"limit":500,"rows":[{"customer":"Acme","revenue":120340.5}],"hasMore":false}}
```

### `get_artifact`

Fetch an artifact descriptor by `artifact.id`. The payload includes `uri` so the client can read bytes itself. Allowed while running.

### `shutdown`

Abort any running query, end the session, respond, exit 0. Closing stdin without sending it does the same thing without a response.

## Events

| Event | When |
| --- | --- |
| `ready` | Once, first line, session accepting commands. |
| `agent_start` | A `prompt` began. Carries `promptId` when the command had an `id`. |
| `message_start` / `message_update` / `message_end` | One assistant message (one per model request in a tool loop). |
| `sql_update` | Cumulative SQL recovered from a streaming `execute_sql` tool call. |
| `tool_execution_start` / `tool_execution_end` | A tool ran. SQL tools include a `queryResult` handle, not the full grid. |
| `agent_end` | The run finished: `completed`, `aborted`, or `error`. Always last. |

`message_update.assistantMessageEvent.type` is one of `text_start`, `text_delta`, `text_end`, `thinking_start`, `thinking_delta`, `thinking_end`, `toolcall_start`, `toolcall_delta`, `toolcall_end`.

`agent_end` on `completed` includes the new transcript messages, the final answer text, usage, query-result and artifact descriptors, and the thread id. On `aborted` and `error` any open message is implicitly closed and nothing was committed.

## Types

All RPC-defined field names are camelCase. Timestamps are epoch milliseconds.

**State** (`ready`, `get_state`, mutating responses): `state`, `database` (`name`, `type`, `names`), `model` (`name`, `id`), `thinkingLevel`, `thinkingLevels`, `dangerousMode`, `csvToolResults`, `threadPersistence`, `threadId`, `messageCount`.

**Transcript message**

```
{role:"user", content: string, timestamp}
{role:"assistant", content: ContentBlock[], model, usage, stopReason, timestamp}
{role:"toolResult", toolCallId, toolName, content, isError, queryResult?, timestamp}

ContentBlock = {type:"text", text} | {type:"thinking", thinking}
             | {type:"toolCall", id, name, arguments}
```

**QueryResult** — `id`, `file`, `rowCount`, `columns`, `size`, `sha256`, `mediaType`, `databaseName`, `createdAt`. Fetch rows with `get_query_result`.

**Artifact** — `id`, `name`, `kind` (`image` \| `notebook` \| `file`), `mediaType`, `size`, `sha256`, `uri`.

## Minimal client

```python
import json
import subprocess

proc = subprocess.Popen(
    ["saber", "rpc", "-d", "analytics", "--no-thread"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
)

def send(command: dict) -> None:
    proc.stdin.write(json.dumps(command) + "\n")
    proc.stdin.flush()

first = json.loads(proc.stdout.readline())
if first["type"] != "ready":
    raise SystemExit(first["error"])

send({"id": 1, "type": "prompt", "message": "How many orders shipped late last month?"})
for raw in proc.stdout:
    event = json.loads(raw)
    if event["type"] == "message_update":
        delta = event.get("assistantMessageEvent", {})
        if delta.get("type") == "text_delta":
            print(delta["delta"], end="", flush=True)
    elif event["type"] == "sql_update":
        print(f"\n[sql] {event['sql']}")
    elif event["type"] == "agent_end":
        break

send({"type": "shutdown"})
proc.wait()
```

## What is not here

- `steer`, `follow_up`, queues: `SQLSaber.query` rejects overlapping runs. Wait for `agent_end`.
- `bash*`, extension UI, `compact*`, auto-retry, session trees (`fork` / `clone` / `get_tree`): threads are linear snapshots.
- `set_model` / `cycle_model`: model selection is `saber models set`. `reload_model` applies it live.
- Images on `prompt`: SQLSaber prompts are text.

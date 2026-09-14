"""RPC session: one SQLSaber, one JSONL conversation, ordered stdout."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import threading
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent

from sqlsaber import SQLSaber, SQLSaberResult
from sqlsaber.artifacts import ArtifactUnavailable
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.query_results import QueryResultUnavailable
from sqlsaber.sdk.errors import (
    RunInProgressError,
    SQLSaberClosedError,
    ThreadResumeError,
)

from .protocol import (
    MAX_STDIN_LINE,
    OVERSIZE_LINE,
    Abort,
    Aborted,
    AgentEnd,
    AgentStart,
    Command,
    Completed,
    Err,
    Event,
    Failed,
    GetArtifact,
    GetLastAssistantText,
    GetMessages,
    GetQueryResult,
    GetState,
    GetTables,
    Invalid,
    NewSession,
    Ok,
    Prompt,
    ReadCommand,
    Ready,
    ReloadModel,
    Response,
    RunOutcome,
    RunUsageSummary,
    SetThinkingLevel,
    Shutdown,
    StateSnapshot,
    encode,
    parse_command,
    query_result_page,
    state_json,
    artifact_json,
    message_json,
)
from .transcript import StreamTranslator, last_assistant_text, transcript

BUSY_ERROR = 'A query is already running. Send {"type":"abort"} or wait for agent_end.'
IDLE_ONLY_ERROR = 'A query is running. Send {"type":"abort"} or wait for agent_end.'
HARD_CANCEL_TIMEOUT = 5.0


@dataclass(frozen=True, slots=True)
class Idle:
    pass


@dataclass(slots=True)
class Running:
    task: asyncio.Task[None]
    abort: asyncio.Event
    prompt: str
    prompt_id: str | int | None
    abort_waiters: list[Abort] = field(default_factory=list)
    shutdown_waiters: list[Shutdown] = field(default_factory=list)
    watch: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class Closed:
    pass


type SessionState = Idle | Running | Closed


class _RunAborted(Exception):
    """Raised inside the event handler when the abort token is set."""


class LineReader(Protocol):
    async def readline(self) -> bytes:
        """One record including its terminator; ``b""`` at EOF."""
        ...


def _read_available(stream: Any) -> bytes:
    """Read whatever is ready without waiting to fill a buffer.

    ``BufferedReader.read(n)`` keeps pulling from the raw stream until it
    has ``n`` bytes or hits EOF. On a pipe that deadlocks an interactive
    JSONL client: the first command is shorter than the buffer, the client
    waits for a response, and this side waits for more input.

    ``read1`` issues at most one raw read, so a complete line unblocks.

    Args:
        stream: Binary stdin, typically ``sys.stdin.buffer``.

    Returns:
        Available bytes, or empty at EOF.
    """
    read1 = getattr(stream, "read1", None)
    try:
        chunk = read1(8192) if callable(read1) else stream.read(8192)
    except Exception:
        return b""
    return b"" if chunk is None else chunk


class ThreadedLineReader:
    """Portable stdin source: a daemon thread blocks on one raw read at a time.

    Enforces the 1 MiB line cap: an oversized line is discarded through the
    next ``\\n`` and surfaced as ``OVERSIZE_LINE``.
    """

    def __init__(self, stream: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._stream = stream
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._thread = threading.Thread(target=self._pump, args=(loop,), daemon=True)
        self._thread.start()

    def _put(self, loop: asyncio.AbstractEventLoop, line: bytes) -> None:
        loop.call_soon_threadsafe(self._queue.put_nowait, line)

    def _pump(self, loop: asyncio.AbstractEventLoop) -> None:
        buf = bytearray()
        oversized = False
        while True:
            chunk = _read_available(self._stream)
            if not chunk:
                if buf and not oversized:
                    self._put(loop, bytes(buf))
                elif oversized:
                    self._put(loop, OVERSIZE_LINE)
                self._put(loop, b"")
                return
            start = 0
            while start < len(chunk):
                newline = chunk.find(b"\n", start)
                if newline < 0:
                    piece = chunk[start:]
                    if oversized:
                        break
                    if len(buf) + len(piece) > MAX_STDIN_LINE:
                        oversized = True
                        buf.clear()
                    else:
                        buf.extend(piece)
                    break
                piece = chunk[start : newline + 1]
                if oversized or len(buf) + len(piece.rstrip(b"\r\n")) > MAX_STDIN_LINE:
                    self._put(loop, OVERSIZE_LINE)
                    oversized = False
                    buf.clear()
                else:
                    buf.extend(piece)
                    self._put(loop, bytes(buf))
                    buf.clear()
                start = newline + 1

    async def readline(self) -> bytes:
        return await self._queue.get()


class RpcSession:
    """One conversation over one SQLSaber, driven by parsed commands."""

    def __init__(
        self,
        saber: SQLSaber,
        *,
        write: Callable[[bytes], None],
        persist_thread: bool,
        abort_grace: float = 0.5,
    ) -> None:
        self._saber = saber
        self._write = write
        self._persist_thread = persist_thread
        self._abort_grace = abort_grace
        self._state: SessionState = Idle()
        self.closed = asyncio.Event()

    def emit_ready(self) -> None:
        self._emit(Ready(state=self._snapshot()))

    async def handle(self, command: Command) -> Response | None:
        """Exactly one response per command, or ``None`` when the reply is held.

        Abort/shutdown during a run return ``None`` so ``serve`` keeps reading
        stdin. Held replies are written after ``agent_end``.
        """
        match self._state, command:
            case _, Invalid(error=error, command=name, id=rid):
                return Err(name, rid, error)
            case Closed(), _:
                return Err(_command_name(command), command.id, "Session is closed")
            case _, Prompt() as prompt:
                return self._start_prompt(prompt)
            case _, Abort() as abort:
                return await self._abort(abort)
            case _, Shutdown() as shutdown:
                return await self._shutdown(shutdown)
            case Running(), GetTables() as tables:
                return Err("get_tables", tables.id, IDLE_ONLY_ERROR)
            case Running(), (
                NewSession() | SetThinkingLevel() | ReloadModel()
            ) as mutate:
                return Err(_command_name(mutate), mutate.id, IDLE_ONLY_ERROR)
            case Idle(), (NewSession() | SetThinkingLevel() | ReloadModel()) as mutate:
                return await self._mutate(mutate)
            case Idle(), GetTables() as tables:
                return await self._read(tables)
            case _, (
                GetState()
                | GetMessages()
                | GetLastAssistantText()
                | GetQueryResult()
                | GetArtifact()
            ) as read:
                return await self._read(read)
        return Err(
            _command_name(command),
            command.id,
            "Internal error: unhandled command",
        )

    async def close(self) -> None:
        """Idempotent. Aborts a running query, then enters Closed."""
        match self._state:
            case Closed():
                return
            case Running() as running:
                running.abort.set()
                self._ensure_watch(running)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(running.task),
                        timeout=self._abort_grace + HARD_CANCEL_TIMEOUT + 0.25,
                    )
                except TimeoutError:
                    running.task.cancel()
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(running.task, timeout=1.0)
                if isinstance(self._state, Closed):
                    return
                self._state = Closed()
                self.closed.set()
            case Idle():
                self._state = Closed()
                self.closed.set()

    def send(self, response: Response) -> None:
        self._write_bytes(encode(response))

    def _start_prompt(self, command: Prompt) -> Response:
        match self._state:
            case Running():
                return Err("prompt", command.id, BUSY_ERROR)
            case Idle():
                abort = asyncio.Event()
                task = asyncio.create_task(
                    self._run(command.message, abort, command.id)
                )
                self._state = Running(
                    task=task,
                    abort=abort,
                    prompt=command.message,
                    prompt_id=command.id,
                )
                return Ok("prompt", command.id)
        return Err("prompt", command.id, "Session is closed")

    async def _run(
        self,
        prompt: str,
        abort: asyncio.Event,
        prompt_id: str | int | None,
    ) -> None:
        outcome: RunOutcome = Failed("internal error")
        try:
            self._emit(AgentStart(prompt_id=prompt_id))
            result = await self._saber.query(
                prompt, event_stream_handler=self._translate
            )
            outcome = Completed(
                messages=tuple(transcript(result.new_messages)),
                text=result.text,
                usage=_run_usage(result),
                query_results=tuple(result.query_results),
                artifacts=tuple(result.artifacts),
                thread_id=self._saber.info.thread_id,
            )
        except _RunAborted:
            outcome = Aborted()
        except asyncio.CancelledError:
            outcome = Aborted()
        except Exception as exc:
            outcome = Failed(str(exc))
        finally:
            abort_waiters: list[Abort] = []
            shutdown_waiters: list[Shutdown] = []
            match self._state:
                case Running() as running:
                    abort_waiters = list(running.abort_waiters)
                    shutdown_waiters = list(running.shutdown_waiters)
                    running.abort_waiters.clear()
                    running.shutdown_waiters.clear()
            self._state = Idle()
            self._emit(AgentEnd(outcome))
            for waiter in abort_waiters:
                self.send(Ok("abort", waiter.id, {"aborted": True}))
            if shutdown_waiters:
                self._state = Closed()
                self.closed.set()
                for waiter in shutdown_waiters:
                    self.send(Ok("shutdown", waiter.id))

    async def _translate(
        self, ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]
    ) -> None:
        model = getattr(ctx, "model", None)
        model_name = getattr(model, "model_name", None)
        translator = StreamTranslator(
            model_name=str(model_name) if model_name is not None else None
        )
        self._raise_if_aborted()
        async for event in events:
            self._raise_if_aborted()
            for wire in translator.translate(event):
                self._emit(wire)
        self._raise_if_aborted()
        for wire in translator.finish():
            self._emit(wire)

    def _raise_if_aborted(self) -> None:
        match self._state:
            case Running(abort=token) if token.is_set():
                raise _RunAborted

    async def _abort(self, command: Abort) -> Response | None:
        match self._state:
            case Idle():
                return Ok("abort", command.id, {"aborted": False})
            case Running() as running:
                running.abort.set()
                running.abort_waiters.append(command)
                self._ensure_watch(running)
                return None
        return Err("abort", command.id, "Session is closed")

    async def _shutdown(self, command: Shutdown) -> Response | None:
        match self._state:
            case Idle():
                self._state = Closed()
                self.closed.set()
                return Ok("shutdown", command.id)
            case Running() as running:
                running.abort.set()
                running.shutdown_waiters.append(command)
                self._ensure_watch(running)
                return None
        return Ok("shutdown", command.id)

    def _ensure_watch(self, running: Running) -> None:
        if running.watch is not None and not running.watch.done():
            return
        running.watch = asyncio.create_task(self._watch_cancel(running))

    async def _watch_cancel(self, running: Running) -> None:
        task = running.task
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self._abort_grace)
            return
        except TimeoutError:
            task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=HARD_CANCEL_TIMEOUT
                )
            except TimeoutError:
                waiters: list[Abort] = []
                match self._state:
                    case Running() as current if current is running:
                        waiters = list(current.abort_waiters)
                        current.abort_waiters.clear()
                for waiter in waiters:
                    self.send(
                        Err(
                            "abort",
                            waiter.id,
                            "Run did not stop; it is still running",
                        )
                    )

    async def _read(self, command: ReadCommand | GetTables) -> Response:
        name = _command_name(command)
        try:
            match command:
                case GetState():
                    return Ok(name, command.id, state_json(self._snapshot()))
                case GetMessages():
                    messages = [
                        message_json(item) for item in transcript(self._saber.messages)
                    ]
                    return Ok(name, command.id, {"messages": messages})
                case GetLastAssistantText():
                    return Ok(
                        name,
                        command.id,
                        {"text": last_assistant_text(self._saber.messages)},
                    )
                case GetTables():
                    tables = [
                        {
                            "database": table.database_name,
                            "schema": table.schema_name,
                            "name": table.name,
                            "kind": table.kind,
                            "qualifiedName": table.qualified_name,
                        }
                        for table in await self._saber.list_tables()
                    ]
                    return Ok(name, command.id, {"tables": tables})
                case GetQueryResult(result_id=result_id, offset=offset, limit=limit):
                    loaded = await self._saber.get_query_result(result_id)
                    return Ok(
                        name,
                        command.id,
                        query_result_page(loaded, offset=offset, limit=limit),
                    )
                case GetArtifact(artifact_id=artifact_id):
                    loaded = await self._saber.get_artifact(artifact_id)
                    return Ok(
                        name,
                        command.id,
                        {"artifact": artifact_json(loaded.descriptor)},
                    )
        except Exception as exc:
            return Err(name, command.id, map_sdk_error(exc))
        return Err(name, command.id, "Internal error: unhandled read")

    async def _mutate(
        self, command: NewSession | SetThinkingLevel | ReloadModel
    ) -> Response:
        name = _command_name(command)
        try:
            match command:
                case NewSession():
                    await self._saber.new_thread()
                case SetThinkingLevel(level="off"):
                    self._saber.set_thinking(enabled=False)
                case SetThinkingLevel(level=level):
                    self._saber.set_thinking(enabled=True, level=ThinkingLevel(level))
                case ReloadModel():
                    self._saber.reload_model_settings()
            return Ok(name, command.id, state_json(self._snapshot()))
        except Exception as exc:
            return Err(name, command.id, map_sdk_error(exc))

    def _snapshot(self) -> StateSnapshot:
        info = self._saber.info
        thinking = "off" if not info.thinking.enabled else info.thinking.level.value
        phase: str = "running" if isinstance(self._state, Running) else "idle"
        return StateSnapshot(
            state="running" if phase == "running" else "idle",
            database_names=info.database_names,
            primary_database=info.primary_database_name,
            database_type=info.primary_database_type,
            model_name=info.model_name,
            model_id=info.model_id,
            thinking=thinking,
            dangerous_mode=info.dangerous_mode,
            csv_tool_results=info.csv_tool_results,
            thread_id=info.thread_id,
            thread_persistence=self._persist_thread,
            message_count=len(transcript(self._saber.messages)),
        )

    def _emit(self, event: Event) -> None:
        self._write_bytes(encode(event))

    def _write_bytes(self, record: bytes) -> None:
        try:
            self._write(record)
        except BrokenPipeError:
            if not self.closed.is_set():
                with contextlib.suppress(RuntimeError):
                    asyncio.get_running_loop().create_task(self.close())


def map_sdk_error(exc: BaseException) -> str:
    """Map an SDK/store exception onto a protocol error string.

    Args:
        exc: Raised error.

    Returns:
        Client-facing error text. ``error`` stays a string (Pi-shaped).
    """
    if isinstance(exc, RunInProgressError):
        return BUSY_ERROR
    if isinstance(exc, SQLSaberClosedError):
        return "SQLSaber is closed"
    if isinstance(exc, QueryResultUnavailable):
        return str(exc)
    if isinstance(exc, ArtifactUnavailable):
        return str(exc)
    if isinstance(exc, ThreadResumeError):
        return str(exc)
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Internal error: {exc}"


def _command_name(command: Command) -> str:
    match command:
        case Prompt():
            return "prompt"
        case Abort():
            return "abort"
        case NewSession():
            return "new_session"
        case GetState():
            return "get_state"
        case GetMessages():
            return "get_messages"
        case GetLastAssistantText():
            return "get_last_assistant_text"
        case SetThinkingLevel():
            return "set_thinking_level"
        case ReloadModel():
            return "reload_model"
        case GetTables():
            return "get_tables"
        case GetQueryResult():
            return "get_query_result"
        case GetArtifact():
            return "get_artifact"
        case Shutdown():
            return "shutdown"
        case Invalid(command=name):
            return name
    return "parse"


def _run_usage(result: SQLSaberResult) -> RunUsageSummary | None:
    usage = result.usage
    if usage is None:
        return None
    return RunUsageSummary(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        requests=usage.requests,
        tool_calls=usage.tool_calls,
        context_tokens=result.final_context_tokens,
    )


async def _next_line_or_closed(
    reader: LineReader, closed: asyncio.Event
) -> bytes | None:
    """Race one ``readline`` against the closed flag.

    Args:
        reader: Stdin source.
        closed: Set when the session enters Closed.

    Returns:
        The next line, ``b""`` at EOF, or ``None`` when closed won.
    """
    read_task = asyncio.create_task(reader.readline())
    closed_task = asyncio.create_task(closed.wait())
    try:
        _done, pending = await asyncio.wait(
            {read_task, closed_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if closed.is_set():
            return None
        return read_task.result()
    except asyncio.CancelledError:
        read_task.cancel()
        closed_task.cancel()
        raise


def _install_signals(session: RpcSession) -> Callable[[], None]:
    loop = asyncio.get_running_loop()
    signals = [
        sig
        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None))
        if sig is not None
    ]

    def _close() -> None:
        loop.create_task(session.close())

    installed: list[signal.Signals] = []
    for sig in signals:
        try:
            loop.add_signal_handler(sig, _close)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, OSError):
            break

    def _uninstall() -> None:
        for sig in installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, OSError):
                loop.remove_signal_handler(sig)

    return _uninstall


async def serve(
    saber: SQLSaber,
    *,
    reader: LineReader,
    write: Callable[[bytes], None],
    persist_thread: bool,
    abort_grace: float = 0.5,
) -> None:
    """Run one session to completion (shutdown, EOF, signal, or client gone).

    Args:
        saber: Conversation owner. Not closed by this function.
        reader: Stdin source.
        write: Sink for complete JSONL records.
        persist_thread: Whether this session saves a thread.
        abort_grace: Seconds to wait for cooperative abort before hard cancel.
    """
    session = RpcSession(
        saber, write=write, persist_thread=persist_thread, abort_grace=abort_grace
    )
    uninstall = _install_signals(session)
    session.emit_ready()
    try:
        while not session.closed.is_set():
            line = await _next_line_or_closed(reader, session.closed)
            if line is None:
                break
            if line == b"":
                await session.close()
                break
            if line.strip(b"\r\n") == b"":
                continue
            payload = line.rstrip(b"\r\n")
            if line == OVERSIZE_LINE or len(payload) > MAX_STDIN_LINE:
                session.send(Err("parse", None, "Input line exceeds 1 MiB"))
                continue
            response = await session.handle(parse_command(line))
            if response is not None:
                session.send(response)
    finally:
        uninstall()
        await session.close()

"""Codex App Server Runtime-B adapter using the installed stdio protocol."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.runtime import RuntimeCapabilities
from contracts.result import AgentEvent, AgentResult, RunHandle, RunStatus, Usage
from contracts.task import TaskContract

_RUNTIME_ID = "codex-app-server"
_TOOL_ITEM_TYPES = {
    "commandExecution",
    "fileChange",
    "mcpToolCall",
    "dynamicToolCall",
    "collabAgentToolCall",
    "webSearch",
}


def _where_codex() -> list[Path]:
    candidates: list[Path] = []
    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            result = subprocess.run(
                ["where.exe", "codex"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                candidates.extend(Path(line.strip()) for line in result.stdout.splitlines() if line.strip())
    resolved = shutil.which("codex")
    if resolved:
        candidates.append(Path(resolved))
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_codex_launch_command(candidates: Sequence[str | Path] | None = None) -> list[str]:
    """Resolve a Windows-native launch without relying on a Git-Bash job-control shim."""
    paths = [Path(value) for value in candidates] if candidates is not None else _where_codex()
    for candidate in paths:
        root = candidate.parent
        node = root / ("node.exe" if os.name == "nt" else "node")
        codex_js = root / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        if node.is_file() and codex_js.is_file():
            return [str(node), str(codex_js)]
    for candidate in paths:
        if candidate.suffix.casefold() == ".exe" and candidate.is_file():
            return [str(candidate)]
    if os.name == "nt":
        cmd = shutil.which("cmd.exe") or os.environ.get("COMSPEC")
        for candidate in paths:
            if candidate.suffix.casefold() == ".cmd" and candidate.is_file() and cmd:
                return [str(cmd), "/d", "/s", "/c", str(candidate)]
    raise FileNotFoundError("no Windows-native Codex launch path was resolved")


@dataclass(slots=True)
class _RunState:
    handle: RunHandle
    thread_id: str
    turn_id: str
    model: str | None
    provider: str | None
    cwd: str
    sandbox: Any
    approval_policy: Any
    queue: asyncio.Queue[AgentEvent] = field(default_factory=asyncio.Queue)
    terminal: asyncio.Event = field(default_factory=asyncio.Event)
    result: AgentResult | None = None
    usage: Usage | None = None
    cancel_requested: bool = False
    timeout_requested: bool = False
    approval_requests: list[str] = field(default_factory=list)
    agent_messages: list[str] = field(default_factory=list)
    completed_tools: list[dict[str, Any]] = field(default_factory=list)
    sequence: int = 0


class CodexAppServerAdapter:
    """One-process, one-active-turn V1 adapter for Codex App Server."""

    def __init__(
        self,
        *,
        launch_command: Sequence[str] | None = None,
        request_timeout: float = 30.0,
        wait_timeout: float = 120.0,
        interrupt_grace: float = 5.0,
        process_cwd: str | Path | None = None,
    ) -> None:
        self._launch_command = list(launch_command) if launch_command is not None else None
        self._request_timeout = request_timeout
        self._wait_timeout = wait_timeout
        self._interrupt_grace = interrupt_grace
        self._process_cwd = Path(process_cwd).resolve() if process_cwd is not None else None
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._request_id = 0
        self._write_lock = asyncio.Lock()
        self._runs: dict[str, _RunState] = {}
        self._backlog: list[dict[str, Any]] = []
        self._approval_backlog: dict[str, list[str]] = {}
        self._closing = False
        self._protocol_error: str | None = None
        self._stderr_seen = False
        self.runtime_version: str | None = None
        self.initialize_evidence: dict[str, Any] = {}
        self.initialized_sent = False

    async def connect(self) -> None:
        if self._process is not None and self._process.returncode is None and self.initialize_evidence:
            return
        command = self._launch_command or resolve_codex_launch_command()
        self._launch_command = list(command)
        self._closing = False
        self._protocol_error = None
        self._process = await asyncio.create_subprocess_exec(
            *command,
            "app-server",
            "--listen",
            "stdio://",
            cwd=str(self._process_cwd) if self._process_cwd is not None else None,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_stdout(), name="codex-app-server-stdout")
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name="codex-app-server-stderr")
        result = await self._request(
            "initialize",
            {
                "clientInfo": {"name": "adaptive-agent-orchestrator", "version": "0.1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        required = ("userAgent", "platformFamily", "platformOs", "codexHome")
        if not all(isinstance(result.get(name), str) and result[name] for name in required):
            await self.disconnect()
            raise RuntimeError("Codex initialize returned malformed required evidence")
        self.initialize_evidence = {
            "userAgent": result["userAgent"],
            "platformFamily": result["platformFamily"],
            "platformOs": result["platformOs"],
            "codexHomeObserved": True,
        }
        match = re.search(r"/(\d+\.\d+\.\d+)", result["userAgent"])
        self.runtime_version = match.group(1) if match else None
        await self._notify("initialized", None)
        self.initialized_sent = True
        if self._process.returncode is not None:
            raise RuntimeError("Codex app-server exited after initialize")

    async def disconnect(self) -> None:
        self._closing = True
        process, self._process = self._process, None
        if process is not None:
            if process.stdin is not None:
                with contextlib.suppress(OSError, RuntimeError):
                    process.stdin.close()
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._reader_task = None
        self._stderr_task = None
        error = ConnectionError("Codex app-server disconnected")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            streaming_events=True,
            mid_run_steer=False,
            native_delegation=False,
            cancellation=True,
            session_resume=False,
            max_concurrent_runs=1,
            filesystem_read=False,
            filesystem_write=False,
            shell=False,
            tests=False,
            web=False,
            background_execution=False,
            persistent_tasks=False,
            human_in_loop=False,
            native_kanban=False,
            structured_output=True,
            usage_observable=True,
            cost_observable=False,
        )

    async def submit(self, task: TaskContract) -> RunHandle:
        await self.connect()
        if any(state.result is None for state in self._runs.values()):
            raise RuntimeError("Codex Runtime-B V1 supports one active run")
        if task.workspace is None:
            raise RuntimeError("Codex Runtime-B requires an authoritative AAO workspace")
        cwd = str(Path(task.workspace.path).resolve())
        thread_result = await self._request(
            "thread/start",
            {
                "cwd": cwd,
                "ephemeral": True,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "dynamicTools": [],
                "environments": [],
                "experimentalRawEvents": False,
                "allowProviderModelFallback": False,
            },
        )
        thread = thread_result.get("thread")
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str) or not thread["id"]:
            raise RuntimeError("Codex thread/start returned no Thread.id")
        thread_id = thread["id"]
        effective_cwd = thread_result.get("cwd")
        if not isinstance(effective_cwd, str) or Path(effective_cwd).resolve() != Path(cwd):
            raise RuntimeError("Codex effective cwd does not match authoritative workspace")
        turn_result = await self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": task.prompt_preamble()}],
            },
        )
        turn = turn_result.get("turn")
        if not isinstance(turn, dict) or not isinstance(turn.get("id"), str) or not turn["id"]:
            raise RuntimeError("Codex turn/start returned no Turn.id")
        turn_id = turn["id"]
        handle = RunHandle(run_id=turn_id, task_id=task.id, session_id=thread_id)
        state = _RunState(
            handle=handle,
            thread_id=thread_id,
            turn_id=turn_id,
            model=thread_result.get("model") if isinstance(thread_result.get("model"), str) else None,
            provider=(
                thread_result.get("modelProvider")
                if isinstance(thread_result.get("modelProvider"), str)
                else None
            ),
            cwd=effective_cwd,
            sandbox=thread_result.get("sandbox"),
            approval_policy=thread_result.get("approvalPolicy"),
        )
        self._runs[turn_id] = state
        for method in self._approval_backlog.pop(turn_id, []):
            state.approval_requests.append(method)
            await state.queue.put(self._event(state, "approval_request", {"method": method}))
        if self._protocol_error is not None:
            await self._finish(
                state,
                native_status="protocolFailure",
                status=RunStatus.FAILED,
                turn=None,
                error=self._protocol_error,
            )
        backlog, self._backlog = self._backlog, []
        for notification in backlog:
            if not self._notification_matches(notification, state):
                self._backlog.append(notification)
                continue
            await self._handle_notification(notification, state)
        return handle

    async def events(
        self,
        handle: RunHandle,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        if after is not None:
            raise RuntimeError("Codex Runtime-B V1 does not support event resume cursors")
        state = self._state(handle)
        while True:
            event = await state.queue.get()
            yield event
            if event.type in {"completed", "error"}:
                return

    async def wait(self, handle: RunHandle) -> AgentResult:
        state = self._state(handle)
        if state.result is None:
            try:
                await asyncio.wait_for(state.terminal.wait(), timeout=self._wait_timeout)
            except TimeoutError:
                state.timeout_requested = True
                with contextlib.suppress(Exception):
                    await self._request(
                        "turn/interrupt",
                        {"threadId": state.thread_id, "turnId": state.turn_id},
                    )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(state.terminal.wait(), timeout=self._interrupt_grace)
                if state.result is None:
                    await self._finish(
                        state,
                        native_status="inProgress",
                        status=RunStatus.TIMEOUT,
                        turn=None,
                        error="Codex turn exceeded the host deadline",
                    )
        assert state.result is not None
        return state.result

    async def usage(self, handle: RunHandle) -> Usage:
        state = self._state(handle)
        return state.usage or Usage()

    async def cancel(self, handle: RunHandle) -> None:
        state = self._state(handle)
        if state.result is not None:
            return
        state.cancel_requested = True
        await self._request(
            "turn/interrupt",
            {"threadId": state.thread_id, "turnId": state.turn_id},
        )

    async def steer(self, handle: RunHandle, instruction: str) -> None:
        raise RuntimeError("Codex Runtime-B V1 does not support steer")

    def _state(self, handle: RunHandle) -> _RunState:
        try:
            state = self._runs[handle.run_id]
        except KeyError as exc:
            raise KeyError(f"unknown Codex turn: {handle.run_id}") from exc
        if state.thread_id != handle.session_id:
            raise ValueError("Codex handle thread identity mismatch")
        return state

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None or process.returncode is not None or process.stdin is None:
            raise ConnectionError("Codex app-server is not running")
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write({"id": request_id, "method": method, "params": params})
        try:
            result = await asyncio.wait_for(future, timeout=self._request_timeout)
        finally:
            self._pending.pop(request_id, None)
        if not isinstance(result, dict):
            raise RuntimeError(f"Codex {method} returned a non-object result")
        return result

    async def _notify(self, method: str, params: dict[str, Any] | None) -> None:
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = params
        await self._write(message)

    async def _write(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.returncode is not None or process.stdin is None:
            raise ConnectionError("Codex app-server stdin is unavailable")
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        async with self._write_lock:
            process.stdin.write(payload)
            await process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                raw = await self._process.stdout.readline()
                if not raw:
                    if not self._closing:
                        await self._fail_protocol("Codex app-server stdout closed")
                    return
                try:
                    message = json.loads(raw.decode(errors="replace"))
                except json.JSONDecodeError:
                    await self._fail_protocol("Codex app-server emitted malformed JSON")
                    return
                if not isinstance(message, dict):
                    await self._fail_protocol("Codex app-server emitted a non-object message")
                    return
                response_id = message.get("id")
                if response_id in self._pending and "method" not in message:
                    future = self._pending[response_id]
                    if "error" in message:
                        error = message.get("error") or {}
                        future.set_exception(RuntimeError(str(error.get("message", "Codex request failed"))))
                    else:
                        future.set_result(message.get("result"))
                    continue
                if "method" in message and "id" in message:
                    await self._deny_server_request(message)
                    continue
                if "method" in message:
                    await self._dispatch_notification(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                await self._fail_protocol(f"Codex protocol reader failed: {type(exc).__name__}")

    async def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        try:
            while await self._process.stderr.readline():
                self._stderr_seen = True
        except asyncio.CancelledError:
            raise

    async def _deny_server_request(self, message: dict[str, Any]) -> None:
        method = str(message.get("method"))
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        turn_id = params.get("turnId")
        state = self._runs.get(turn_id) if isinstance(turn_id, str) else None
        if state is not None:
            state.approval_requests.append(method)
            await state.queue.put(self._event(state, "approval_request", {"method": method}))
        elif isinstance(turn_id, str):
            self._approval_backlog.setdefault(turn_id, []).append(method)
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            await self._write({"id": message["id"], "result": {"decision": "cancel"}})
        else:
            await self._write(
                {
                    "id": message["id"],
                    "error": {
                        "code": -32000,
                        "message": "AAO Runtime-B denies interactive server requests",
                    },
                }
            )

    async def _dispatch_notification(self, message: dict[str, Any]) -> None:
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        turn_id = params.get("turnId")
        if not isinstance(turn_id, str):
            turn = params.get("turn")
            turn_id = turn.get("id") if isinstance(turn, dict) else None
        state = self._runs.get(turn_id) if isinstance(turn_id, str) else None
        if state is None:
            self._backlog.append(message)
            return
        if not self._notification_matches(message, state):
            return
        await self._handle_notification(message, state)

    @staticmethod
    def _notification_matches(message: dict[str, Any], state: _RunState) -> bool:
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        turn = params.get("turn")
        if turn_id is None and isinstance(turn, dict):
            turn_id = turn.get("id")
        return thread_id == state.thread_id and turn_id == state.turn_id

    async def _handle_notification(self, message: dict[str, Any], state: _RunState) -> None:
        method = message.get("method")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "thread/tokenUsage/updated":
            token_usage = params.get("tokenUsage")
            last = token_usage.get("last") if isinstance(token_usage, dict) else None
            if isinstance(last, dict):
                state.usage = Usage(
                    input_tokens=self._integer_or_none(last.get("inputTokens")),
                    output_tokens=self._integer_or_none(last.get("outputTokens")),
                    cached_tokens=self._integer_or_none(last.get("cachedInputTokens")),
                    total_tokens=self._integer_or_none(last.get("totalTokens")),
                    estimated_cost_usd=None,
                )
                await state.queue.put(self._event(state, "usage", state.usage.model_dump()))
            return
        if method == "model/rerouted" and isinstance(params.get("toModel"), str):
            state.model = params["toModel"]
            return
        if method == "item/started":
            item = params.get("item")
            if isinstance(item, dict) and item.get("type") in _TOOL_ITEM_TYPES:
                await state.queue.put(
                    self._event(state, "tool_start", {"item_id": item.get("id"), "type": item.get("type")})
                )
            return
        if method == "item/completed":
            item = params.get("item")
            if isinstance(item, dict):
                if item.get("type") == "agentMessage":
                    if isinstance(item.get("text"), str):
                        state.agent_messages.append(item["text"])
                    await state.queue.put(
                        self._event(state, "message", {"item_id": item.get("id"), "text": item.get("text")})
                    )
                elif item.get("type") in _TOOL_ITEM_TYPES:
                    normalized_tool = self._normalize_tool(item)
                    state.completed_tools.append(normalized_tool)
                    await state.queue.put(self._event(state, "tool_complete", normalized_tool))
            return
        if method != "turn/completed":
            return
        turn = params.get("turn")
        if not isinstance(turn, dict) or turn.get("id") != state.turn_id:
            return
        native_status = turn.get("status")
        if native_status == "completed":
            status = RunStatus.COMPLETED
        elif native_status == "failed":
            status = RunStatus.FAILED
        elif native_status == "interrupted" and state.timeout_requested:
            status = RunStatus.TIMEOUT
        elif native_status == "interrupted" and state.cancel_requested:
            status = RunStatus.CANCELLED
        else:
            status = RunStatus.FAILED
        error = None
        turn_error = turn.get("error")
        if isinstance(turn_error, dict) and isinstance(turn_error.get("message"), str):
            error = turn_error["message"]
        if status == RunStatus.FAILED and error is None:
            error = f"Codex turn ended with status {native_status!r}"
        await self._finish(state, native_status=str(native_status), status=status, turn=turn, error=error)

    async def _finish(
        self,
        state: _RunState,
        *,
        native_status: str,
        status: RunStatus,
        turn: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        if state.result is not None:
            return
        output: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        file_paths: list[str] | None = None
        items_view = None
        if isinstance(turn, dict):
            items = turn.get("items")
            items_view = turn.get("itemsView", "full")
            if isinstance(items, list) and items_view == "full":
                messages = [
                    item.get("text")
                    for item in items
                    if isinstance(item, dict)
                    and item.get("type") == "agentMessage"
                    and isinstance(item.get("text"), str)
                ]
                output = messages[-1] if messages else None
                normalized = [
                    self._normalize_tool(item)
                    for item in items
                    if isinstance(item, dict) and item.get("type") in _TOOL_ITEM_TYPES
                ]
                tool_calls = normalized
                file_paths = sorted(
                    {
                        path
                        for item in items
                        if isinstance(item, dict) and item.get("type") == "fileChange"
                        for path in self._file_paths(item)
                    }
                )
        if output is None and state.agent_messages:
            output = state.agent_messages[-1]
        state.result = AgentResult(
            run_id=state.turn_id,
            task_id=state.handle.task_id,
            status=status,
            usage=state.usage,
            files_changed=file_paths,
            summary=output,
            tool_calls=tool_calls,
            model=state.model,
            provider=state.provider,
            runtime_version=self.runtime_version,
            error=error,
            provenance={
                "runtime": _RUNTIME_ID,
                "thread_id": state.thread_id,
                "turn_id": state.turn_id,
                "native_turn_status": native_status,
                "effective_cwd": state.cwd,
                "sandbox_config": state.sandbox,
                "approval_policy": state.approval_policy,
                "approval_requests": list(state.approval_requests),
                "tool_evidence_completeness": "partial" if tool_calls is not None else "unavailable",
                "runtime_file_evidence_completeness": (
                    "partial" if file_paths is not None else "unavailable"
                ),
                "turn_items_view": items_view,
                "stderr_diagnostics_observed": self._stderr_seen,
            },
        )
        event_type = "completed" if status == RunStatus.COMPLETED else "error"
        payload = {
            "native_status": native_status,
            "summary": output,
            "error": error,
            "files_changed": file_paths,
        }
        await state.queue.put(self._event(state, event_type, payload))
        state.terminal.set()

    async def _fail_protocol(self, message: str) -> None:
        self._protocol_error = message
        error = RuntimeError(message)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        for state in self._runs.values():
            if state.result is None:
                await self._finish(
                    state,
                    native_status="protocolFailure",
                    status=RunStatus.FAILED,
                    turn=None,
                    error=message,
                )

    def _event(self, state: _RunState, event_type: str, payload: dict[str, Any]) -> AgentEvent:
        state.sequence += 1
        return AgentEvent(
            id=f"{state.turn_id}:{state.sequence}",
            run_id=state.turn_id,
            type=event_type,
            payload=payload,
        )

    @staticmethod
    def _integer_or_none(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    @staticmethod
    def _file_paths(item: dict[str, Any]) -> list[str]:
        return [
            change["path"]
            for change in item.get("changes", [])
            if isinstance(change, dict) and isinstance(change.get("path"), str)
        ]

    @classmethod
    def _normalize_tool(cls, item: dict[str, Any]) -> dict[str, Any]:
        item_type = str(item.get("type"))
        normalized: dict[str, Any] = {
            "item_id": item.get("id"),
            "type": item_type,
            "status": item.get("status"),
        }
        if item_type == "commandExecution":
            normalized.update(
                {
                    "command": item.get("command"),
                    "cwd": item.get("cwd"),
                    "exit_code": item.get("exitCode"),
                    "duration_ms": item.get("durationMs"),
                }
            )
        elif item_type == "fileChange":
            normalized["paths"] = cls._file_paths(item)
        elif item_type == "mcpToolCall":
            normalized.update(
                {
                    "server": item.get("server"),
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments"),
                    "duration_ms": item.get("durationMs"),
                }
            )
        elif item_type == "dynamicToolCall":
            normalized.update(
                {
                    "namespace": item.get("namespace"),
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments"),
                    "duration_ms": item.get("durationMs"),
                    "success": item.get("success"),
                }
            )
        elif item_type == "collabAgentToolCall":
            normalized.update(
                {
                    "tool": item.get("tool"),
                    "receiver_thread_ids": item.get("receiverThreadIds"),
                }
            )
        elif item_type == "webSearch":
            normalized["query"] = item.get("query")
        return normalized

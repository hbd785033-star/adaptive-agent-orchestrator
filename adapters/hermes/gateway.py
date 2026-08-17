"""
HermesAdapter — AgentRuntime implementation over Hermes TUI Gateway (WebSocket).

Connection lifecycle:
  1. Call connect() to establish WebSocket and authenticate.
  2. submit() creates a session, submits the task, returns a RunHandle.
  3. events() subscribes to the run's event stream with cursor support for reconnect.
  4. Disconnect/reconnect is handled transparently by _ensure_connected().

Protocol:
  JSON-RPC 2.0 over WebSocket, as defined by Hermes TUI Gateway docs.
  All method names mirror the Gateway API (session.create, prompt.submit, etc.).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from websockets.exceptions import ConnectionClosed

from adapters.runtime import RuntimeCapabilities
from contracts.result import (
    AgentEvent,
    AgentResult,
    CompletedPayload,
    ErrorPayload,
    RunHandle,
    RunStatus,
    ToolCompletePayload,
    Usage,
)
from contracts.task import TaskContract

# ── Event type → typed payload parser ────────────────────────────────────────

_TYPED_PARSERS = {
    "tool_complete": ToolCompletePayload,
    "completed": CompletedPayload,
    "error": ErrorPayload,
}


def _build_ws_url(url: str, session_token: str | None = None) -> str:
    """Normalize the installed Hermes dashboard WebSocket contract."""
    parts = urlsplit(url)
    if parts.scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError("Hermes Gateway URL must be an absolute ws:// or wss:// URL")

    path = parts.path.rstrip("/") or "/api/ws"
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    if session_token:
        query = [(key, value) for key, value in query if key != "token"]
        query.append(("token", session_token))
    return urlunsplit(
        (parts.scheme, parts.netloc, path, urlencode(query), parts.fragment)
    )


def _normalize_gateway_event(data: dict) -> tuple[str, str | None, dict]:
    """Normalize installed Hermes event frames to the AAO event contract."""
    params = data.get("params") or {}
    session_id = str(params.get("session_id") or "")
    event_type = str(params.get("type") or "")
    payload = params.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {"text": str(payload)}

    if event_type in {"message.delta", "message.start"}:
        return session_id, "message", payload
    if event_type == "message.complete":
        if payload.get("status") == "error":
            return session_id, "error", {
                "code": "hermes_message_complete_error",
                "message": str(payload.get("text") or "Hermes turn failed"),
            }
        return session_id, "completed", {
            "summary": str(payload.get("text") or ""),
            "files_changed": payload.get("files_changed", []),
            "tests_run": payload.get("tests_run", False),
            "unresolved_risks": payload.get("unresolved_risks", []),
        }
    if event_type in {"tool.start", "tool.progress"}:
        return session_id, "tool_start", payload
    if event_type == "tool.complete":
        return session_id, "tool_complete", payload
    if event_type == "session.usage":
        return session_id, "usage", payload
    if event_type in {"error", "turn.error"}:
        return session_id, "error", payload
    return session_id, None, payload


def _usage_from_evidence(evidence: dict[str, Any]) -> Usage:
    nested = evidence.get("usage")
    if isinstance(nested, dict):
        evidence = nested
    input_tokens = evidence.get("input_tokens", evidence.get("input"))
    output_tokens = evidence.get("output_tokens", evidence.get("output"))
    total_tokens = evidence.get("total_tokens", evidence.get("total"))
    if (
        "total_tokens" not in evidence
        and "total" not in evidence
        and input_tokens is not None
        and output_tokens is not None
    ):
        total_tokens = input_tokens + output_tokens
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=evidence.get("estimated_cost_usd"),
    )


def _parse_event(raw: dict) -> AgentEvent:
    event = AgentEvent(
        id=raw.get("id", uuid.uuid4().hex),
        run_id=raw.get("run_id", ""),
        timestamp=datetime.fromisoformat(raw["timestamp"]) if "timestamp" in raw else datetime.utcnow(),
        type=raw["type"],
        payload=raw.get("payload", {}),
    )
    parser = _TYPED_PARSERS.get(event.type)
    if parser:
        with contextlib.suppress(Exception):
            event.typed_payload = parser(**event.payload)
            # keep raw payload if parse fails; don't break the stream
    return event


# ── Adapter ───────────────────────────────────────────────────────────────────

class HermesAdapter:
    """
    Connects to a running Hermes instance via TUI Gateway WebSocket.

    Parameters
    ----------
    url:
        WebSocket URL, e.g. "ws://localhost:4999"
    api_key:
        Optional auth token for Hermes Gateway.
    reconnect_delay:
        Seconds to wait before reconnect attempts.
    """

    def __init__(
        self,
        url: str = "ws://localhost:4999",
        api_key: str | None = None,
        reconnect_delay: float = 2.0,
        wait_timeout: float = 120.0,
    ) -> None:
        self._url = url
        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN")
        )
        self._reconnect_delay = reconnect_delay
        self._wait_timeout = wait_timeout
        self._ws: Any | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._event_queues: dict[str, asyncio.Queue] = {}  # correlation id → queue
        self._session_to_run: dict[str, str] = {}
        self._streaming_runs: set[str] = set()
        self._handles: dict[str, RunHandle] = {}
        self._run_state: dict[str, dict[str, Any]] = {}
        self._completed_results: dict[str, AgentResult] = {}
        self._completion_events: dict[str, asyncio.Event] = {}
        self._subscription_locks: dict[str, asyncio.Lock] = {}
        self._subscribed_runs: set[str] = set()
        self._recv_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self._shutdown = False
        self._connection_generation = 0

    # ── Connection management ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish WebSocket connection and start the receive loop."""
        self._shutdown = False
        connect_url = _build_ws_url(self._url, self._api_key)
        self._ws = await websockets.connect(connect_url, additional_headers={})
        self._connection_generation += 1
        self._connected.set()
        self._recv_task = asyncio.create_task(self._recv_loop(), name="hermes-recv")

    async def disconnect(self) -> None:
        if self._shutdown and self._ws is None:
            return
        self._shutdown = True
        self._connection_generation += 1
        self._connected.clear()
        reconnect_task = self._reconnect_task
        self._reconnect_task = None
        if reconnect_task and reconnect_task is not asyncio.current_task():
            reconnect_task.cancel()
        await self._fail_active_runs("Hermes Gateway disconnected")
        self._fail_pending(ConnectionError("Hermes Gateway disconnected"))
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        websocket, self._ws = self._ws, None
        if websocket:
            await websocket.close()

    async def _ensure_connected(self) -> None:
        if self._shutdown:
            raise ConnectionError("Hermes Gateway adapter is explicitly disconnected")
        if not self._connected.is_set():
            await self.connect()

    async def _reconnect(self) -> None:
        try:
            self._connected.clear()
            await asyncio.sleep(self._reconnect_delay)
            if self._shutdown:
                return
            with contextlib.suppress(Exception):
                await self.connect()
                # retry handled by caller if this also fails
        finally:
            if self._reconnect_task is asyncio.current_task():
                self._reconnect_task = None

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    # ── Receive loop (fan-out) ────────────────────────────────────────────────

    async def _handle_connection_closed(self) -> None:
        self._connection_generation += 1
        self._connected.clear()
        await self._fail_active_runs(
            "Hermes Gateway disconnected; run resume is unsupported"
        )
        self._fail_pending(ConnectionError("Hermes Gateway disconnected"))
        if not self._shutdown and (
            self._reconnect_task is None or self._reconnect_task.done()
        ):
            self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _recv_loop(self) -> None:
        assert self._ws
        try:
            async for message in self._ws:
                data = json.loads(message)
                msg_id = data.get("id")
                # JSON-RPC response → resolve pending future
                if msg_id and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if "error" in data:
                        fut.set_exception(RuntimeError(data["error"]["message"]))
                    else:
                        fut.set_result(data.get("result"))
                # Installed Hermes WebSocket events use the JSON-RPC event
                # envelope and correlate by the target session id.
                elif data.get("method") == "event":
                    session_id, event_type, payload = _normalize_gateway_event(data)
                    run_id = self._session_to_run.get(session_id)
                    if run_id and event_type:
                        raw = {
                            "run_id": run_id,
                            "session_id": session_id,
                            "type": event_type,
                            "payload": payload,
                        }
                        queue = self._event_queues.setdefault(run_id, asyncio.Queue())
                        await queue.put(raw)
                        handle = self._handles.get(run_id)
                        if handle is not None:
                            self._record_event(handle, _parse_event(raw))
                # Legacy run-scoped event envelope retained for existing adapters.
                elif data.get("type") == "event":
                    run_id = data.get("run_id")
                    if run_id:
                        queue = self._event_queues.setdefault(run_id, asyncio.Queue())
                        raw = data.get("event", data)
                        await queue.put(raw)
                        handle = self._handles.get(run_id)
                        if handle is not None:
                            self._record_event(handle, _parse_event(raw))
        except ConnectionClosed:
            await self._handle_connection_closed()
        else:
            await self._handle_connection_closed()

    async def _fail_active_runs(self, message: str) -> None:
        """Make every known active run terminal and wake any stream consumer."""
        for run_id, queue in self._event_queues.items():
            if run_id in self._completed_results:
                continue
            raw = {
                "run_id": run_id,
                "type": "error",
                "payload": {"message": message},
            }
            await queue.put(raw)
            handle = self._handles.get(run_id)
            if handle is not None:
                self._record_event(handle, _parse_event(raw))

    # ── RPC helper ────────────────────────────────────────────────────────────

    async def _call(self, method: str, params: dict) -> Any:
        await self._ensure_connected()
        assert self._ws
        msg_id = uuid.uuid4().hex
        payload = json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        try:
            await self._ws.send(payload)
            return await asyncio.wait_for(fut, timeout=30.0)
        finally:
            self._pending.pop(msg_id, None)

    # ── AgentRuntime protocol ─────────────────────────────────────────────────

    async def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            streaming_events=True,
            mid_run_steer=True,
            native_delegation=False,
            cancellation=True,
            session_resume=False,
            max_concurrent_runs=8,
        )

    async def submit(self, task: TaskContract) -> RunHandle:
        # Hermes TUI creates a session, then acknowledges prompt.submit with
        # {"status": "streaming"}; the terminal answer arrives as events.
        session = await self._call("session.create", {"label": f"task-{task.id[:8]}"})
        session_id = str(session["session_id"])
        correlation_id = f"tui:{session_id}"
        submit_generation = self._connection_generation
        was_connected = self._connected.is_set()

        self._event_queues.setdefault(correlation_id, asyncio.Queue())
        self._session_to_run[session_id] = correlation_id
        handle = RunHandle(
            run_id=correlation_id,
            task_id=task.id,
            session_id=session_id,
        )
        self._handles[correlation_id] = handle
        self._run_state[correlation_id] = {
            "files_changed": [],
            "summary": "",
            "error": None,
            "usage": None,
            "status": RunStatus.RUNNING,
        }

        try:
            result = await self._call(
                "prompt.submit",
                {
                    "session_id": session_id,
                    "text": task.prompt_preamble(),
                    "background": True,
                },
            )
        except Exception:
            self._session_to_run.pop(session_id, None)
            self._event_queues.pop(correlation_id, None)
            self._handles.pop(correlation_id, None)
            self._run_state.pop(correlation_id, None)
            raise

        backend_run_id = result.get("run_id") if isinstance(result, dict) else None
        if backend_run_id:
            # Preserve compatibility with runtimes that return a native run id.
            provisional_queue = self._event_queues.pop(correlation_id)
            target_queue = self._event_queues.setdefault(
                str(backend_run_id), provisional_queue
            )
            if target_queue is not provisional_queue:
                while not provisional_queue.empty():
                    target_queue.put_nowait(provisional_queue.get_nowait())
            self._handles.pop(correlation_id, None)
            self._run_state[backend_run_id] = self._run_state.pop(correlation_id)
            handle = RunHandle(
                run_id=str(backend_run_id),
                task_id=task.id,
                session_id=session_id,
            )
            self._handles[handle.run_id] = handle
            self._session_to_run[session_id] = handle.run_id
        elif isinstance(result, dict) and result.get("status") == "streaming":
            self._streaming_runs.add(correlation_id)
        else:
            raise RuntimeError("Hermes prompt.submit returned no accepted streaming acknowledgement")

        disconnected_during_submit = (
            self._shutdown
            or self._connection_generation != submit_generation
            or (was_connected and not self._connected.is_set())
        )
        if disconnected_during_submit:
            self._record_event(handle, _parse_event({
                "run_id": handle.run_id,
                "type": "error",
                "payload": {
                    "message": "Hermes Gateway disconnected before run registration"
                },
            }))
        # A terminal push may arrive before prompt.submit returns.
        self._record_buffered_events(handle)
        return handle

    def _record_buffered_events(self, handle: RunHandle) -> None:
        """Record a queue snapshot without consuming it from stream clients."""
        queue = self._event_queues.setdefault(handle.run_id, asyncio.Queue())
        buffered: list[dict[str, Any]] = []
        while True:
            try:
                buffered.append(queue.get_nowait())
                queue.task_done()
            except asyncio.QueueEmpty:
                break
        for raw in buffered:
            queue.put_nowait(raw)
            self._record_event(handle, _parse_event(raw))

    def _rpc_session_params(self, handle: RunHandle) -> dict[str, str]:
        if handle.run_id in self._streaming_runs and handle.session_id:
            return {"session_id": handle.session_id}
        return {"run_id": handle.run_id}

    async def _subscribe(
        self, handle: RunHandle, *, after: str | None = None
    ) -> None:
        """Subscribe once per run so waiters and stream clients cannot race."""
        if handle.run_id in self._streaming_runs:
            return
        lock = self._subscription_locks.setdefault(handle.run_id, asyncio.Lock())
        async with lock:
            if after is None and handle.run_id in self._subscribed_runs:
                return
            params: dict[str, Any] = self._rpc_session_params(handle)
            if after:
                params["after"] = after
            await self._call("session.subscribe", params)
            if after is None:
                self._subscribed_runs.add(handle.run_id)
            self._record_buffered_events(handle)

    async def status(self, handle: RunHandle) -> RunStatus:
        result = await self._call("session.status", self._rpc_session_params(handle))
        _status_map = {
            "pending": RunStatus.PENDING,
            "running": RunStatus.RUNNING,
            "completed": RunStatus.COMPLETED,
            "failed": RunStatus.FAILED,
            "cancelled": RunStatus.CANCELLED,
            "approval_required": RunStatus.APPROVAL_REQUIRED,
        }
        return _status_map.get(result.get("status", ""), RunStatus.RUNNING)

    def _record_event(self, handle: RunHandle, event: AgentEvent) -> None:
        if (
            event.type in ("completed", "error")
            and handle.run_id in self._completed_results
        ):
            self._completion_events.setdefault(handle.run_id, asyncio.Event()).set()
            return
        state = self._run_state.setdefault(handle.run_id, {
            "files_changed": [], "summary": "", "error": None,
            "usage": None, "status": RunStatus.RUNNING,
        })
        if event.type == "completed":
            if isinstance(event.typed_payload, CompletedPayload):
                state["files_changed"].extend(event.typed_payload.files_changed)
                state["summary"] = event.typed_payload.summary
            else:
                state["files_changed"].extend(event.payload.get("files_changed", []))
                state["summary"] = event.payload.get("summary", "")
            state["status"] = RunStatus.COMPLETED
        elif event.type == "tool_complete" and isinstance(event.typed_payload, ToolCompletePayload):
            state["files_changed"].extend(event.typed_payload.files_written)
        elif event.type == "usage":
            state["usage"] = _usage_from_evidence(event.payload)
        elif event.type == "error":
            state["error"] = (
                event.typed_payload.message
                if isinstance(event.typed_payload, ErrorPayload)
                else event.payload.get("message", "Hermes run failed")
            )
            state["status"] = RunStatus.FAILED

        if event.type in ("completed", "error"):
            self._completed_results[handle.run_id] = AgentResult(
                run_id=handle.run_id,
                task_id=handle.task_id,
                status=state["status"],
                usage=state["usage"],
                files_changed=sorted(set(state["files_changed"])),
                summary=state["summary"],
                error=state["error"],
            )
            self._completion_events.setdefault(handle.run_id, asyncio.Event()).set()

    async def wait(self, handle: RunHandle) -> AgentResult:
        if handle.run_id not in self._completed_results:
            try:
                await self._subscribe(handle)
            except Exception:
                if handle.run_id not in self._completed_results:
                    raise
            completion = self._completion_events.setdefault(
                handle.run_id, asyncio.Event()
            )
            if handle.run_id in self._completed_results:
                completion.set()
            try:
                await asyncio.wait_for(completion.wait(), timeout=self._wait_timeout)
            except TimeoutError as exc:
                raise TimeoutError("streaming runtime wait timed out") from exc
        return self._completed_results[handle.run_id]

    async def result(self, run_id: str) -> AgentResult:
        """Backward-compatible shim; protocol callers should use wait(handle)."""
        handle = self._handles.get(run_id, RunHandle(run_id=run_id, task_id=""))
        return await self.wait(handle)

    async def usage(self, handle: RunHandle) -> Usage:
        result = await self._call("session.usage", self._rpc_session_params(handle))
        return _usage_from_evidence(result)

    async def cancel(self, handle: RunHandle) -> None:
        await self._call("session.interrupt", self._rpc_session_params(handle))

    async def steer(self, handle: RunHandle, instruction: str) -> None:
        params = self._rpc_session_params(handle)
        params["text"] = instruction
        await self._call(
            "session.steer",
            params,
        )

    async def events(
        self,
        handle: RunHandle,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        queue = self._event_queues.setdefault(handle.run_id, asyncio.Queue())
        # A completed result can still have an early buffered stream to deliver.
        if handle.run_id in self._completed_results and queue.empty():
            return
        if handle.run_id not in self._completed_results:
            await self._subscribe(handle, after=after)
        while True:
            try:
                raw = await asyncio.wait_for(queue.get(), timeout=self._wait_timeout)
            except TimeoutError as exc:
                raise TimeoutError("streaming runtime wait timed out") from exc
            event = _parse_event(raw)
            self._record_event(handle, event)
            yield event
            if event.type in ("completed", "error"):
                break

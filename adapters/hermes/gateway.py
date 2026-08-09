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
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

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
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._reconnect_delay = reconnect_delay
        self._ws: Any | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._event_queues: dict[str, asyncio.Queue] = {}  # run_id → queue
        self._handles: dict[str, RunHandle] = {}
        self._run_state: dict[str, dict[str, Any]] = {}
        self._completed_results: dict[str, AgentResult] = {}
        self._recv_task: asyncio.Task | None = None
        self._connected = asyncio.Event()

    # ── Connection management ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish WebSocket connection and start the receive loop."""
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._ws = await websockets.connect(self._url, additional_headers=headers)
        self._connected.set()
        self._recv_task = asyncio.create_task(self._recv_loop(), name="hermes-recv")

    async def disconnect(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        self._connected.clear()

    async def _ensure_connected(self) -> None:
        if not self._connected.is_set():
            await self.connect()

    async def _reconnect(self) -> None:
        self._connected.clear()
        await asyncio.sleep(self._reconnect_delay)
        with contextlib.suppress(Exception):
            await self.connect()
            # retry handled by caller if this also fails

    # ── Receive loop (fan-out) ────────────────────────────────────────────────

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
                # Push event to run-specific queue
                elif data.get("type") == "event":
                    run_id = data.get("run_id")
                    if run_id:
                        queue = self._event_queues.setdefault(run_id, asyncio.Queue())
                        await queue.put(data.get("event", data))
        except ConnectionClosed:
            for run_id, queue in self._event_queues.items():
                if run_id not in self._completed_results:
                    await queue.put({
                        "run_id": run_id,
                        "type": "error",
                        "payload": {
                            "message": "Hermes Gateway disconnected; run resume is unsupported"
                        },
                    })
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("Hermes Gateway disconnected"))
            self._pending.clear()
            asyncio.create_task(self._reconnect())

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
        # 1. Create a new session
        session = await self._call("session.create", {"label": f"task-{task.id[:8]}"})
        session_id = session["session_id"]

        # 2. Submit prompt with constraint preamble
        result = await self._call(
            "prompt.submit",
            {
                "session_id": session_id,
                "text": task.prompt_preamble(),
                "background": True,
            },
        )
        run_id = result["run_id"]

        # Register event queue for this run
        self._event_queues[run_id] = asyncio.Queue()

        handle = RunHandle(run_id=run_id, task_id=task.id, session_id=session_id)
        self._handles[run_id] = handle
        self._run_state[run_id] = {
            "files_changed": [],
            "summary": "",
            "error": None,
            "usage": Usage(),
            "status": RunStatus.RUNNING,
        }
        return handle

    async def status(self, handle: RunHandle) -> RunStatus:
        result = await self._call("session.status", {"run_id": handle.run_id})
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
        state = self._run_state.setdefault(handle.run_id, {
            "files_changed": [], "summary": "", "error": None,
            "usage": Usage(), "status": RunStatus.RUNNING,
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
            state["usage"] = Usage(
                input_tokens=event.payload.get("input_tokens", 0),
                output_tokens=event.payload.get("output_tokens", 0),
                total_tokens=event.payload.get(
                    "total_tokens",
                    event.payload.get("input_tokens", 0) + event.payload.get("output_tokens", 0),
                ),
                estimated_cost_usd=event.payload.get("estimated_cost_usd"),
            )
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

    async def wait(self, handle: RunHandle) -> AgentResult:
        if handle.run_id not in self._completed_results:
            async for _ in self.events(handle):
                pass
        return self._completed_results[handle.run_id]

    async def result(self, run_id: str) -> AgentResult:
        """Backward-compatible shim; protocol callers should use wait(handle)."""
        handle = self._handles.get(run_id, RunHandle(run_id=run_id, task_id=""))
        return await self.wait(handle)

    async def usage(self, handle: RunHandle) -> Usage:
        result = await self._call("session.usage", {"run_id": handle.run_id})
        return Usage(
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            estimated_cost_usd=result.get("estimated_cost_usd"),
        )

    async def cancel(self, handle: RunHandle) -> None:
        await self._call("session.interrupt", {"run_id": handle.run_id})

    async def steer(self, handle: RunHandle, instruction: str) -> None:
        await self._call(
            "session.steer",
            {"run_id": handle.run_id, "text": instruction},
        )

    async def events(
        self,
        handle: RunHandle,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # Subscribe on Gateway
        if handle.run_id in self._completed_results:
            return
        params: dict[str, Any] = {"run_id": handle.run_id}
        if after:
            params["after"] = after
        await self._call("session.subscribe", params)

        if handle.run_id not in self._event_queues:
            self._event_queues[handle.run_id] = asyncio.Queue()

        queue = self._event_queues[handle.run_id]
        while True:
            raw = await queue.get()
            event = _parse_event(raw)
            self._record_event(handle, event)
            yield event
            if event.type in ("completed", "error"):
                break

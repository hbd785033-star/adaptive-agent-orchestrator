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
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterator

import websockets
from websockets.exceptions import ConnectionClosed

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
        try:
            event.typed_payload = parser(**event.payload)
        except Exception:
            pass  # keep raw payload; don't break the stream
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
        try:
            await self.connect()
        except Exception:
            pass  # retry handled by caller

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
                    if run_id and run_id in self._event_queues:
                        await self._event_queues[run_id].put(data.get("event", data))
        except ConnectionClosed:
            asyncio.create_task(self._reconnect())

    # ── RPC helper ────────────────────────────────────────────────────────────

    async def _call(self, method: str, params: dict) -> Any:
        await self._ensure_connected()
        assert self._ws
        msg_id = uuid.uuid4().hex
        payload = json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        await self._ws.send(payload)
        return await asyncio.wait_for(fut, timeout=30.0)

    # ── AgentRuntime protocol ─────────────────────────────────────────────────

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

        return RunHandle(run_id=run_id, task_id=task.id, session_id=session_id)

    async def status(self, run_id: str) -> RunStatus:
        result = await self._call("session.status", {"run_id": run_id})
        _status_map = {
            "pending": RunStatus.PENDING,
            "running": RunStatus.RUNNING,
            "completed": RunStatus.COMPLETED,
            "failed": RunStatus.FAILED,
            "cancelled": RunStatus.CANCELLED,
            "approval_required": RunStatus.APPROVAL_REQUIRED,
        }
        return _status_map.get(result.get("status", ""), RunStatus.RUNNING)

    async def result(self, run_id: str) -> AgentResult:
        # Drain event stream to build result
        files_changed: list[str] = []
        summary = ""
        error_msg: str | None = None
        usage = Usage()
        final_status = RunStatus.COMPLETED

        async for event in self.events(run_id):
            if event.type == "completed" and isinstance(event.typed_payload, CompletedPayload):
                p = event.typed_payload
                files_changed = p.files_changed
                summary = p.summary
            elif event.type == "tool_complete" and isinstance(event.typed_payload, ToolCompletePayload):
                files_changed.extend(event.typed_payload.files_written)
            elif event.type == "usage":
                usage = Usage(
                    input_tokens=event.payload.get("input_tokens", 0),
                    output_tokens=event.payload.get("output_tokens", 0),
                    total_tokens=event.payload.get("total_tokens", 0),
                )
            elif event.type == "error":
                if isinstance(event.typed_payload, ErrorPayload):
                    error_msg = event.typed_payload.message
                final_status = RunStatus.FAILED

        return AgentResult(
            run_id=run_id,
            task_id="",  # caller fills this from RunHandle
            status=final_status,
            usage=usage,
            files_changed=list(set(files_changed)),
            summary=summary,
            error=error_msg,
        )

    async def usage(self, run_id: str) -> Usage:
        result = await self._call("session.usage", {"run_id": run_id})
        return Usage(
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            estimated_cost_usd=result.get("estimated_cost_usd"),
        )

    async def cancel(self, run_id: str) -> None:
        await self._call("session.interrupt", {"run_id": run_id})

    async def steer(self, run_id: str, text: str) -> None:
        await self._call("session.steer", {"run_id": run_id, "text": text})

    async def events(
        self,
        run_id: str,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # Subscribe on Gateway
        params: dict[str, Any] = {"run_id": run_id}
        if after:
            params["after"] = after
        await self._call("session.subscribe", params)

        if run_id not in self._event_queues:
            self._event_queues[run_id] = asyncio.Queue()

        queue = self._event_queues[run_id]
        while True:
            raw = await queue.get()
            event = _parse_event(raw)
            yield event
            if event.type in ("completed", "error"):
                break

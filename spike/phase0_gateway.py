"""
Phase 0 — Hermes Gateway compatibility spike.

Validates the full session lifecycle against a live Hermes instance.
Run BEFORE writing any Orchestrator code that depends on the adapter.

Usage:
    python spike/phase0_gateway.py --url ws://localhost:4999 [--api-key TOKEN]

Output:
    YAML compatibility report written to spike/phase0_report.yaml
    Exit code 0 = all blockers passed, 1 = one or more blockers failed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
import websockets
from websockets.exceptions import ConnectionClosed


# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str = "unknown"    # pass | fail | partial | skip
    detail: str = ""
    blocker: bool = True
    elapsed_ms: int = 0


@dataclass
class SpikeReport:
    hermes_url: str
    timestamp: str = ""
    blockers: list[CheckResult] = field(default_factory=list)
    nice_to_have: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        if result.blocker:
            self.blockers.append(result)
        else:
            self.nice_to_have.append(result)

    def all_blockers_passed(self) -> bool:
        return all(r.status == "pass" for r in self.blockers)

    def to_yaml(self) -> str:
        d = {
            "hermes_url": self.hermes_url,
            "timestamp": self.timestamp,
            "result": "PASS" if self.all_blockers_passed() else "FAIL",
            "blockers": {r.name: {"status": r.status, "detail": r.detail, "elapsed_ms": r.elapsed_ms}
                         for r in self.blockers},
            "nice_to_have": {r.name: {"status": r.status, "detail": r.detail, "elapsed_ms": r.elapsed_ms}
                             for r in self.nice_to_have},
        }
        return yaml.dump(d, default_flow_style=False, allow_unicode=True)


# ── Minimal JSON-RPC client ───────────────────────────────────────────────────

class GatewayClient:
    def __init__(self, url: str, api_key: str | None = None):
        self.url = url
        self.api_key = api_key
        self.ws = None
        self._pending: dict[str, asyncio.Future] = {}
        self._events: list[dict] = []
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.ws = await websockets.connect(self.url, additional_headers=headers, open_timeout=10)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def disconnect(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _recv_loop(self) -> None:
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                msg_id = data.get("id")
                if msg_id and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if "error" in data:
                        fut.set_exception(RuntimeError(str(data["error"])))
                    else:
                        fut.set_result(data.get("result"))
                else:
                    self._events.append(data)
        except (ConnectionClosed, asyncio.CancelledError):
            pass

    async def call(self, method: str, params: dict, timeout: float = 15.0) -> dict:
        msg_id = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        await self.ws.send(json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}))
        return await asyncio.wait_for(fut, timeout=timeout)

    async def wait_for_event(self, event_type: str, run_id: str, timeout: float = 30.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for e in self._events:
                if e.get("type") == event_type and e.get("run_id") == run_id:
                    return e
                if e.get("event", {}).get("type") == event_type:
                    return e
            await asyncio.sleep(0.2)
        return None


# ── Spike checks ──────────────────────────────────────────────────────────────

async def check_websocket_connect(url: str, api_key: str | None) -> CheckResult:
    t0 = time.monotonic()
    try:
        client = GatewayClient(url, api_key)
        await client.connect()
        await client.disconnect()
        return CheckResult("websocket_connect", "pass", "connected successfully",
                           elapsed_ms=int((time.monotonic() - t0) * 1000))
    except Exception as e:
        return CheckResult("websocket_connect", "fail", str(e),
                           elapsed_ms=int((time.monotonic() - t0) * 1000))


async def check_session_create(client: GatewayClient) -> tuple[CheckResult, str | None]:
    t0 = time.monotonic()
    try:
        result = await client.call("session.create", {"label": "phase0-spike"})
        session_id = result.get("session_id")
        if not session_id:
            return CheckResult("session_create", "fail", f"no session_id in response: {result}",
                               elapsed_ms=int((time.monotonic() - t0) * 1000)), None
        return CheckResult("session_create", "pass", f"session_id={session_id}",
                           elapsed_ms=int((time.monotonic() - t0) * 1000)), session_id
    except Exception as e:
        return CheckResult("session_create", "fail", str(e),
                           elapsed_ms=int((time.monotonic() - t0) * 1000)), None


async def check_task_submit(client: GatewayClient, session_id: str) -> tuple[CheckResult, str | None]:
    t0 = time.monotonic()
    try:
        result = await client.call("prompt.submit", {
            "session_id": session_id,
            "text": "Echo back the text: PHASE0_TEST_PING",
            "background": True,
        })
        run_id = result.get("run_id")
        if not run_id:
            return CheckResult("task_submit", "fail", f"no run_id: {result}",
                               elapsed_ms=int((time.monotonic() - t0) * 1000)), None
        return CheckResult("task_submit", "pass", f"run_id={run_id}",
                           elapsed_ms=int((time.monotonic() - t0) * 1000)), run_id
    except Exception as e:
        return CheckResult("task_submit", "fail", str(e),
                           elapsed_ms=int((time.monotonic() - t0) * 1000)), None


async def check_event_stream(client: GatewayClient, run_id: str) -> CheckResult:
    t0 = time.monotonic()
    try:
        # Subscribe
        await client.call("session.subscribe", {"run_id": run_id}, timeout=10)
        # Wait for any event
        event = await client.wait_for_event("completed", run_id, timeout=60)
        if event:
            return CheckResult("event_stream", "pass", "received completed event",
                               elapsed_ms=int((time.monotonic() - t0) * 1000))
        # Check if we got any events at all
        if client._events:
            return CheckResult("event_stream", "partial",
                               f"got {len(client._events)} events but not 'completed'",
                               elapsed_ms=int((time.monotonic() - t0) * 1000))
        return CheckResult("event_stream", "fail", "no events received within 60s",
                           elapsed_ms=int((time.monotonic() - t0) * 1000))
    except Exception as e:
        return CheckResult("event_stream", "fail", str(e),
                           elapsed_ms=int((time.monotonic() - t0) * 1000))


async def check_session_usage(client: GatewayClient, run_id: str) -> CheckResult:
    t0 = time.monotonic()
    try:
        result = await client.call("session.usage", {"run_id": run_id})
        has_tokens = "input_tokens" in result or "total_tokens" in result
        status = "pass" if has_tokens else "partial"
        return CheckResult("usage_metrics", status, str(result), blocker=False,
                           elapsed_ms=int((time.monotonic() - t0) * 1000))
    except Exception as e:
        return CheckResult("usage_metrics", "fail", str(e), blocker=False,
                           elapsed_ms=int((time.monotonic() - t0) * 1000))


async def check_reconnect(url: str, api_key: str | None, session_id: str) -> CheckResult:
    """Disconnect and reconnect; verify session is still accessible."""
    t0 = time.monotonic()
    try:
        client2 = GatewayClient(url, api_key)
        await client2.connect()
        result = await client2.call("session.status", {"session_id": session_id})
        await client2.disconnect()
        if result:
            return CheckResult("reconnect_with_cursor", "pass",
                               f"session still accessible after reconnect: {result}",
                               elapsed_ms=int((time.monotonic() - t0) * 1000))
        return CheckResult("reconnect_with_cursor", "fail", "empty status on reconnect",
                           elapsed_ms=int((time.monotonic() - t0) * 1000))
    except Exception as e:
        return CheckResult("reconnect_with_cursor", "fail", str(e),
                           elapsed_ms=int((time.monotonic() - t0) * 1000))


async def check_steer(client: GatewayClient, run_id: str) -> CheckResult:
    t0 = time.monotonic()
    try:
        await client.call("session.steer", {"run_id": run_id, "text": "phase0 steer test"})
        return CheckResult("subagent_steer", "pass", "steer accepted", blocker=False,
                           elapsed_ms=int((time.monotonic() - t0) * 1000))
    except Exception as e:
        return CheckResult("subagent_steer", "partial", f"steer not supported or errored: {e}",
                           blocker=False, elapsed_ms=int((time.monotonic() - t0) * 1000))


async def check_interrupt(client: GatewayClient) -> CheckResult:
    t0 = time.monotonic()
    try:
        # Submit a long-running task to interrupt
        session_r = await client.call("session.create", {"label": "phase0-interrupt-test"})
        interrupt_session = session_r.get("session_id", "")
        run_r = await client.call("prompt.submit", {
            "session_id": interrupt_session,
            "text": "Count from 1 to 10000 slowly.",
            "background": True,
        })
        interrupt_run_id = run_r.get("run_id", "")
        await asyncio.sleep(1)
        await client.call("session.interrupt", {"run_id": interrupt_run_id})
        return CheckResult("subagent_interrupt", "pass", "interrupt accepted", blocker=False,
                           elapsed_ms=int((time.monotonic() - t0) * 1000))
    except Exception as e:
        return CheckResult("subagent_interrupt", "partial", f"interrupt errored: {e}",
                           blocker=False, elapsed_ms=int((time.monotonic() - t0) * 1000))


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_spike(url: str, api_key: str | None) -> SpikeReport:
    from datetime import datetime, timezone
    report = SpikeReport(hermes_url=url, timestamp=datetime.now(timezone.utc).isoformat())

    print(f"\n🔍 Phase 0 — Hermes Gateway Spike")
    print(f"   Target: {url}\n")

    # ⓪ WebSocket connect (standalone — needs no session)
    r = await check_websocket_connect(url, api_key)
    report.add(r)
    _print_check(r)
    if r.status == "fail":
        print("\n❌  Cannot connect — aborting remaining checks.")
        return report

    # All subsequent checks share one client
    client = GatewayClient(url, api_key)
    await client.connect()

    try:
        # ① session.create
        r, session_id = await check_session_create(client)
        report.add(r)
        _print_check(r)

        # ② prompt.submit
        run_id: str | None = None
        if session_id:
            r, run_id = await check_task_submit(client, session_id)
            report.add(r)
            _print_check(r)

        # ③ event stream
        if run_id:
            r = await check_event_stream(client, run_id)
            report.add(r)
            _print_check(r)

        # ④ session.usage
        if run_id:
            r = await check_session_usage(client, run_id)
            report.add(r)
            _print_check(r)

        # ⑤ steer
        if run_id:
            r = await check_steer(client, run_id)
            report.add(r)
            _print_check(r)

        # ⑥ interrupt
        r = await check_interrupt(client)
        report.add(r)
        _print_check(r)

    finally:
        await client.disconnect()

    # ⑦ reconnect
    if session_id:
        r = await check_reconnect(url, api_key, session_id)
        report.add(r)
        _print_check(r)

    return report


def _print_check(r: CheckResult) -> None:
    icon = "✅" if r.status == "pass" else ("⚠️ " if r.status == "partial" else "❌")
    label = "[BLOCKER]" if r.blocker else "[optional]"
    print(f"  {icon} {r.name:30s} {label}  {r.elapsed_ms}ms")
    if r.status not in ("pass",):
        print(f"     → {r.detail[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 Hermes Gateway spike")
    parser.add_argument("--url", default="ws://localhost:4999", help="Hermes TUI Gateway WebSocket URL")
    parser.add_argument("--api-key", default=None, help="Optional API key / bearer token")
    parser.add_argument("--out", default="spike/phase0_report.yaml", help="Output report path")
    args = parser.parse_args()

    report = asyncio.run(run_spike(args.url, args.api_key))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_yaml())

    print(f"\n{'=' * 55}")
    if report.all_blockers_passed():
        print("✅  All blockers PASSED — Hermes adapter is viable.")
        print(f"   Report: {out_path}")
        sys.exit(0)
    else:
        failed = [r.name for r in report.blockers if r.status != "pass"]
        print(f"❌  {len(failed)} blocker(s) FAILED: {failed}")
        print(f"   Fix these before building the Orchestrator.")
        print(f"   Report: {out_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

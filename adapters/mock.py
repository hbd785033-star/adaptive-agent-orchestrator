"""
MockHermesAdapter — deterministic fake runtime for unit tests.

Behaviour is driven by a scenario dict so tests can specify
what events the adapter should emit without a live Hermes process.

API matches adapters/runtime.py AgentRuntime Protocol (RunHandle-based).
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from adapters.runtime import RuntimeCapabilities
from contracts.result import (
    AgentEvent,
    AgentResult,
    CompletedPayload,
    RunHandle,
    RunStatus,
    Usage,
)
from contracts.task import TaskContract
from orchestrator.cost import estimate_cost


class _RunState:
    def __init__(self, task_id: str, scenario: dict[str, Any]) -> None:
        self.task_id = task_id
        self.scenario = scenario
        self.status = RunStatus.PENDING
        self._events_exhausted = False


class MockHermesAdapter:
    """
    Usage in tests::

        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", files_changed=["src/foo.py"])

        handle = await adapter.submit(task)
        result = await adapter.wait(handle)
        assert result.status == RunStatus.COMPLETED
    """

    def __init__(self) -> None:
        self._runs: dict[str, _RunState] = {}
        self._scenario_queue: list[dict[str, Any]] = []

    def enqueue_scenario(
        self,
        outcome: str = "pass",          # "pass" | "fail" | "approval_required"
        files_changed: list[str] | None = None,
        summary: str = "Mock task completed",
        error_message: str | None = None,
        input_tokens: int = 500,
        output_tokens: int = 200,
        model: str = "claude-sonnet-4",
        extra_events: list[dict] | None = None,
    ) -> None:
        self._scenario_queue.append(
            dict(
                outcome=outcome,
                files_changed=files_changed or [],
                summary=summary,
                error_message=error_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                extra_events=extra_events or [],
            )
        )

    # ── AgentRuntime protocol (RunHandle-based) ───────────────────────────────

    async def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            streaming_events=True,
            mid_run_steer=False,
            native_delegation=False,
            cancellation=True,
            session_resume=False,
            max_concurrent_runs=8,
        )

    async def submit(self, task: TaskContract) -> RunHandle:
        scenario = self._scenario_queue.pop(0) if self._scenario_queue else {}
        run_id = uuid.uuid4().hex
        self._runs[run_id] = _RunState(task_id=task.id, scenario=scenario)
        return RunHandle(run_id=run_id, task_id=task.id)

    async def wait(self, handle: RunHandle) -> AgentResult:
        """Block until run is terminal (drain events), return final result."""
        state = self._runs[handle.run_id]
        # Drain event iterator to drive state transitions
        async for _ in self.events(handle):
            pass
        return self._build_result(handle.run_id, state)

    async def usage(self, handle: RunHandle) -> Usage:
        s = self._runs[handle.run_id].scenario
        return Usage(
            input_tokens=s.get("input_tokens", 0),
            output_tokens=s.get("output_tokens", 0),
            total_tokens=s.get("input_tokens", 0) + s.get("output_tokens", 0),
            estimated_cost_usd=estimate_cost(
                s.get("model", "claude-sonnet-4"),
                s.get("input_tokens", 0),
                s.get("output_tokens", 0),
            ),
        )

    async def cancel(self, handle: RunHandle) -> None:
        self._runs[handle.run_id].status = RunStatus.CANCELLED

    async def steer(self, handle: RunHandle, instruction: str) -> None:
        # No-op in mock; tests can inspect steering calls via subclass
        pass

    async def events(
        self,
        handle: RunHandle,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        state = self._runs[handle.run_id]
        scenario = state.scenario

        def _evt(type_: str, payload: dict) -> AgentEvent:
            return AgentEvent(
                id=uuid.uuid4().hex,
                run_id=handle.run_id,
                timestamp=datetime.utcnow(),
                type=type_,
                payload=payload,
            )

        state.status = RunStatus.RUNNING

        # Emit any caller-supplied extra events first
        for e in scenario.get("extra_events", []):
            await asyncio.sleep(0)
            yield _evt(e["type"], e.get("payload", {}))

        outcome = scenario.get("outcome", "pass")

        if outcome == "approval_required":
            state.status = RunStatus.APPROVAL_REQUIRED
            yield _evt("approval_request", {"reason": "mock approval required"})

        # Simulate tool call
        await asyncio.sleep(0)
        yield _evt("tool_start", {"tool_name": "mock_tool"})
        await asyncio.sleep(0)
        yield _evt(
            "tool_complete",
            {
                "tool_name": "mock_tool",
                "files_written": scenario.get("files_changed", []),
                "exit_code": 0,
                "duration_ms": 100,
            },
        )

        # Usage event
        await asyncio.sleep(0)
        yield _evt(
            "usage",
            {
                "input_tokens": scenario.get("input_tokens", 500),
                "output_tokens": scenario.get("output_tokens", 200),
            },
        )

        # Terminal event
        await asyncio.sleep(0)
        if outcome == "fail":
            state.status = RunStatus.FAILED
            yield _evt("error", {"code": "mock_error", "message": scenario.get("error_message", "mock failure")})
        else:
            state.status = RunStatus.COMPLETED
            yield _evt(
                "completed",
                CompletedPayload(
                    summary=scenario.get("summary", ""),
                    files_changed=scenario.get("files_changed", []),
                    tests_run=True,
                    unresolved_risks=[],
                ).model_dump(),
            )

    # ── Legacy shim (backward compat with tests using run_id strings) ─────────

    async def result(self, run_id: str) -> AgentResult:
        """Backward-compat shim — prefer wait(handle)."""
        handle = RunHandle(run_id=run_id, task_id=self._runs[run_id].task_id)
        return await self.wait(handle)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_result(self, run_id: str, state: _RunState) -> AgentResult:
        scenario = state.scenario
        return AgentResult(
            run_id=run_id,
            task_id=state.task_id,
            status=state.status,
            usage=Usage(
                input_tokens=scenario.get("input_tokens", 0),
                output_tokens=scenario.get("output_tokens", 0),
                total_tokens=scenario.get("input_tokens", 0) + scenario.get("output_tokens", 0),
                estimated_cost_usd=estimate_cost(
                    scenario.get("model", "claude-sonnet-4"),
                    scenario.get("input_tokens", 0),
                    scenario.get("output_tokens", 0),
                ),
            ),
            files_changed=scenario.get("files_changed", []),
            summary=scenario.get("summary", ""),
            error=scenario.get("error_message"),
        )

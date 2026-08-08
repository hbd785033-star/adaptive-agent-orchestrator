"""
MockHermesAdapter — deterministic fake runtime for unit tests.

Behaviour is driven by a scenario dict so tests can specify
what events the adapter should emit without a live Hermes process.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from contracts.result import (
    AgentEvent,
    AgentResult,
    CompletedPayload,
    RunHandle,
    RunStatus,
    Usage,
)
from contracts.task import TaskContract


class MockHermesAdapter:
    """
    Usage in tests:

        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", files_changed=["src/foo.py"])

        handle = await adapter.submit(task)
        result = await adapter.result(handle.run_id)
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
                extra_events=extra_events or [],
            )
        )

    # ── AgentRuntime protocol ────────────────────────────────────────────────

    async def submit(self, task: TaskContract) -> RunHandle:
        scenario = self._scenario_queue.pop(0) if self._scenario_queue else {}
        run_id = uuid.uuid4().hex
        self._runs[run_id] = _RunState(task_id=task.id, scenario=scenario)
        return RunHandle(run_id=run_id, task_id=task.id)

    async def status(self, run_id: str) -> RunStatus:
        return self._runs[run_id].status

    async def result(self, run_id: str) -> AgentResult:
        state = self._runs[run_id]
        # Drain the event iterator to drive state transitions
        async for _ in self.events(run_id):
            pass
        return AgentResult(
            run_id=run_id,
            task_id=state.task_id,
            status=state.status,
            usage=Usage(
                input_tokens=state.scenario.get("input_tokens", 0),
                output_tokens=state.scenario.get("output_tokens", 0),
                total_tokens=state.scenario.get("input_tokens", 0) + state.scenario.get("output_tokens", 0),
            ),
            files_changed=state.scenario.get("files_changed", []),
            summary=state.scenario.get("summary", ""),
            error=state.scenario.get("error_message"),
        )

    async def usage(self, run_id: str) -> Usage:
        s = self._runs[run_id].scenario
        return Usage(
            input_tokens=s.get("input_tokens", 0),
            output_tokens=s.get("output_tokens", 0),
            total_tokens=s.get("input_tokens", 0) + s.get("output_tokens", 0),
        )

    async def cancel(self, run_id: str) -> None:
        self._runs[run_id].status = RunStatus.CANCELLED

    async def steer(self, run_id: str, text: str) -> None:
        # No-op in mock; tests can inspect steering calls via subclass
        pass

    async def events(
        self,
        run_id: str,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        state = self._runs[run_id]
        scenario = state.scenario

        def _evt(type_: str, payload: dict) -> AgentEvent:
            return AgentEvent(
                id=uuid.uuid4().hex,
                run_id=run_id,
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
            return

        if outcome == "fail":
            state.status = RunStatus.FAILED
            yield _evt(
                "error",
                {
                    "code": "mock_error",
                    "message": scenario.get("error_message", "Mock failure"),
                    "recoverable": False,
                },
            )
            return

        # Pass
        yield _evt(
            "tool_complete",
            {
                "tool_name": "mock_tool",
                "files_written": scenario.get("files_changed", []),
                "exit_code": 0,
                "duration_ms": 100,
            },
        )
        yield _evt(
            "usage",
            {
                "input_tokens": scenario.get("input_tokens", 0),
                "output_tokens": scenario.get("output_tokens", 0),
            },
        )

        completed_payload = CompletedPayload(
            summary=scenario.get("summary", ""),
            files_changed=scenario.get("files_changed", []),
            tests_run=True,
        )
        yield _evt("completed", completed_payload.model_dump())
        state.status = RunStatus.COMPLETED


class _RunState:
    __slots__ = ("task_id", "scenario", "status")

    def __init__(self, task_id: str, scenario: dict) -> None:
        self.task_id = task_id
        self.scenario = scenario
        self.status = RunStatus.PENDING

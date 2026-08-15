"""
AgentRuntime Protocol — the single interface every adapter must implement.

All orchestrator code talks to this; concrete implementations live in
adapters/hermes/ and adapters/mock.py.

Design notes
------------
- Methods accept RunHandle objects, not raw run_id strings.
  This prevents accidental cross-run ID confusion in concurrent scenarios.
- capabilities() is cheap / cached; the engine can call it before task execution
  to know which optional features (steer, delegation, streaming) are available.
- events() is the primary result path; result()/wait() is a convenience
  wrapper for callers that don't need streaming.
"""
from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from contracts.result import AgentEvent, AgentResult, RunHandle, Usage
from contracts.task import TaskContract


@dataclasses.dataclass(frozen=True)
class RuntimeCapabilities:
    """
    Feature flags reported by a runtime implementation.
    Engine and DelegationExecutor consult this before using optional operations.
    """
    streaming_events: bool = True      # supports events() generator
    mid_run_steer: bool = False        # supports steer() while running
    native_delegation: bool = False    # runtime handles fan-out natively
    cancellation: bool = True          # supports cancel()
    session_resume: bool = False       # can reconnect and resume (cursor)
    max_concurrent_runs: int = 8       # advisory — how many parallel submits are safe

    filesystem_read: bool = False
    filesystem_write: bool = False
    shell: bool = False
    tests: bool = False
    web: bool = False
    background_execution: bool = False
    persistent_tasks: bool = False
    human_in_loop: bool = False
    native_kanban: bool = False
    structured_output: bool = False
    usage_observable: bool = False
    cost_observable: bool = False


@runtime_checkable
class AgentRuntime(Protocol):

    async def capabilities(self) -> RuntimeCapabilities:
        """
        Return feature flags for this runtime instance.
        Should be cheap / cached — called before each task execution.
        """
        ...

    async def submit(self, task: TaskContract) -> RunHandle:
        """Submit a task; return a handle immediately (non-blocking)."""
        ...

    async def wait(self, handle: RunHandle) -> AgentResult:
        """Block until run is terminal, then return the result."""
        ...

    async def events(
        self,
        handle: RunHandle,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Subscribe to the event stream for a run.

        Parameters
        ----------
        handle:
            The run to subscribe to.
        after:
            Cursor — resume from this event ID on reconnect (WebSocket drop/reconnect).
            Pass None to stream from the beginning.

        Yields
        ------
        AgentEvent instances in chronological order.
        Stream ends when a ``completed`` or ``error`` event is emitted.
        """
        ...

    async def usage(self, handle: RunHandle) -> Usage:
        """Return accumulated token/cost usage for this run."""
        ...

    async def cancel(self, handle: RunHandle) -> None:
        """Request cancellation of a running task."""
        ...

    async def steer(self, handle: RunHandle, instruction: str) -> None:
        """
        Send a mid-run steering message to the agent.
        Only call after checking capabilities().mid_run_steer == True.
        """
        ...

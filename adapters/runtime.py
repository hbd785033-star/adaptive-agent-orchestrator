"""
AgentRuntime Protocol — the single interface every adapter must implement.

All orchestrator code talks to this; concrete implementations live in
adapters/hermes/ and adapters/mock.py.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from contracts.result import AgentResult, AgentEvent, RunHandle, RunStatus, Usage
from contracts.task import TaskContract


@runtime_checkable
class AgentRuntime(Protocol):
    async def submit(self, task: TaskContract) -> RunHandle:
        """Submit a task and return a handle immediately (non-blocking)."""
        ...

    async def status(self, run_id: str) -> RunStatus:
        """Poll current run status."""
        ...

    async def result(self, run_id: str) -> AgentResult:
        """Block until run is terminal, then return the result."""
        ...

    async def usage(self, run_id: str) -> Usage:
        """Return accumulated token/cost usage for this run."""
        ...

    async def cancel(self, run_id: str) -> None:
        """Request cancellation of a running task."""
        ...

    async def steer(self, run_id: str, text: str) -> None:
        """
        Send a mid-run steering message to the agent.
        Corresponds to session.steer / subagent.steer in Hermes TUI Gateway.
        """
        ...

    def events(
        self,
        run_id: str,
        *,
        after: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Subscribe to the event stream for a run.

        Parameters
        ----------
        after:
            Cursor — resume from this event ID on reconnect.
            If None, stream from the beginning.

        Yields
        ------
        AgentEvent instances in chronological order.
        The stream ends when a "completed" or "error" event is emitted.
        """
        ...

"""
TaskStatus state machine + SQLite persistence.

States:
    RECEIVED → PROFILED → ROUTED → RUNNING → EVALUATING → COMPLETED
                                           ↘ FAILED → RETRYING → ROUTED (retry)
                                                     ↘ ABANDONED
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from contracts.task import TaskContract
from storage.database import Database


class TaskStatus(StrEnum):
    RECEIVED   = "received"
    PROFILED   = "profiled"
    ROUTED     = "routed"
    RUNNING    = "running"
    EVALUATING = "evaluating"
    COMPLETED  = "completed"
    FAILED     = "failed"
    RETRYING   = "retrying"
    ABANDONED  = "abandoned"


# Valid transitions — guards prevent illegal state jumps
_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.RECEIVED:   {TaskStatus.PROFILED,   TaskStatus.ABANDONED},
    TaskStatus.PROFILED:   {TaskStatus.ROUTED,     TaskStatus.ABANDONED},
    TaskStatus.ROUTED:     {TaskStatus.RUNNING,    TaskStatus.ABANDONED, TaskStatus.FAILED},
    TaskStatus.RUNNING:    {TaskStatus.EVALUATING, TaskStatus.FAILED},
    TaskStatus.EVALUATING: {TaskStatus.COMPLETED,  TaskStatus.FAILED},
    TaskStatus.FAILED:     {TaskStatus.RETRYING,   TaskStatus.ABANDONED},
    TaskStatus.RETRYING:   {TaskStatus.ROUTED,     TaskStatus.ABANDONED, TaskStatus.FAILED},
    TaskStatus.COMPLETED:  set(),
    TaskStatus.ABANDONED:  set(),
}


class IllegalTransitionError(Exception):
    pass


class TaskRecord:
    """In-memory task state; synced to DB on every transition."""

    def __init__(
        self,
        task: TaskContract,
        db: Database,
    ) -> None:
        self.task = task
        self.db = db
        self.status = TaskStatus.RECEIVED
        self.route: str | None = None
        self.run_id: str | None = None
        self.retry_count: int = 0
        self.created_at = datetime.utcnow()

    async def transition(self, new_status: TaskStatus, **extra_fields: Any) -> None:
        allowed = _TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise IllegalTransitionError(
                f"{self.status} → {new_status} is not a valid transition"
            )
        self.status = new_status
        update: dict[str, Any] = {"status": new_status.value}
        update.update(extra_fields)
        await self.db.upsert_task(self.task.id, **update)

    async def mark_routed(self, route: str) -> None:
        self.route = route
        await self.transition(TaskStatus.ROUTED, route=route)

    async def mark_running(self, run_id: str) -> None:
        self.run_id = run_id
        await self.transition(TaskStatus.RUNNING)
        await self.db.insert_run(run_id, self.task.id)

    async def mark_evaluating(self) -> None:
        await self.transition(TaskStatus.EVALUATING)

    async def mark_completed(self) -> None:
        await self.transition(TaskStatus.COMPLETED)

    async def mark_failed(self, error: str = "") -> None:
        await self.transition(TaskStatus.FAILED)
        if self.run_id:
            await self.db.update_run(self.run_id, status="failed",
                                     finished_at=datetime.utcnow().isoformat(),
                                     error=error)

    async def mark_retry(self) -> None:
        self.retry_count += 1
        await self.transition(TaskStatus.RETRYING, retry_count=self.retry_count)

    async def mark_abandoned(self, reason: str = "") -> None:
        await self.transition(TaskStatus.ABANDONED)


class StateMachine:
    """
    Factory + registry.

    Usage:
        sm = StateMachine(db)
        record = await sm.create(task)
        await record.mark_routed("delegation")
        ...
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._records: dict[str, TaskRecord] = {}

    async def create(self, task: TaskContract) -> TaskRecord:
        record = TaskRecord(task, self._db)
        self._records[task.id] = record
        await self._db.upsert_task(
            task.id,
            task_type=task.task_type.value,
            goal=task.goal,
            risk=task.risk.value,
            complexity=task.complexity,
            status=TaskStatus.RECEIVED.value,
            created_at=datetime.utcnow().isoformat(),
        )
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

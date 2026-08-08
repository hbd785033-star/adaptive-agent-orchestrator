"""Telemetry — structured append-only event log."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class TelemetryEvent:
    """
    One structured event emitted during task execution.

    Every event is written to:
      - structlog (stdout / log file)
      - Database.append_telemetry() via the telemetry recorder
    """
    __slots__ = ("id", "task_id", "run_id", "event_type", "payload", "recorded_at")

    def __init__(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.task_id = task_id
        self.run_id = run_id
        self.event_type = event_type
        self.payload = payload
        self.recorded_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "recorded_at": self.recorded_at.isoformat(),
        }


class TelemetryRecorder:
    """
    Emits TelemetryEvents to the database and structlog.

    Usage:
        recorder = TelemetryRecorder(db)
        await recorder.record(task_id, "task_routed", {"route": "delegation"})
    """

    def __init__(self, db=None) -> None:
        self._db = db

    async def record(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> TelemetryEvent:
        event = TelemetryEvent(task_id, event_type, payload, run_id)
        log.info(event_type, **event.to_dict())
        if self._db is not None:
            await self._db.append_telemetry(task_id, run_id, event_type, payload)
        return event

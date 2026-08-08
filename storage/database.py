"""Async SQLite database layer."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import aiosqlite

from storage.migrations.v1 import ALL_STATEMENTS, SCHEMA_VERSION


_DEFAULT_DB = Path("data/orchestrator.db")


class Database:
    def __init__(self, path: Path = _DEFAULT_DB):
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._migrate()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def _migrate(self) -> None:
        assert self._db
        # Check current version
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        async with self._db.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ) as cur:
            row = await cur.fetchone()
            current = row["v"] if row and row["v"] else 0

        if current < SCHEMA_VERSION:
            for stmt in ALL_STATEMENTS:
                for s in stmt.strip().split(";"):
                    s = s.strip()
                    if s:
                        await self._db.execute(s)
            await self._db.execute(
                "INSERT OR REPLACE INTO schema_version VALUES (?, ?)",
                (SCHEMA_VERSION, _now()),
            )
            await self._db.commit()

    # ── Operational writes ────────────────────────────────────────────────────

    async def upsert_task(self, task_id: str, **fields) -> None:
        assert self._db
        fields["updated_at"] = _now()
        if not await self._task_exists(task_id):
            fields.setdefault("created_at", _now())
            fields.setdefault("status", "received")
            cols = ", ".join(fields.keys())
            placeholders = ", ".join("?" for _ in fields)
            await self._db.execute(
                f"INSERT INTO tasks (id, {cols}) VALUES (?, {placeholders})",
                (task_id, *fields.values()),
            )
        else:
            set_clause = ", ".join(f"{k}=?" for k in fields)
            await self._db.execute(
                f"UPDATE tasks SET {set_clause} WHERE id=?",
                (*fields.values(), task_id),
            )
        await self._db.commit()

    async def update_task_status(self, task_id: str, status: str) -> None:
        await self.upsert_task(task_id, status=status)

    async def insert_run(self, run_id: str, task_id: str, session_id: str | None = None) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO runs (id, task_id, session_id, status, started_at) VALUES (?,?,?,?,?)",
            (run_id, task_id, session_id, "pending", _now()),
        )
        await self._db.commit()

    async def update_run(self, run_id: str, **fields) -> None:
        assert self._db
        set_clause = ", ".join(f"{k}=?" for k in fields)
        await self._db.execute(
            f"UPDATE runs SET {set_clause} WHERE id=?",
            (*fields.values(), run_id),
        )
        await self._db.commit()

    # ── Audit appends (never update) ──────────────────────────────────────────

    async def append_telemetry(self, task_id: str, run_id: str | None, event_type: str, payload: dict) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO telemetry_events (id, task_id, run_id, event_type, payload, recorded_at) "
            "VALUES (?,?,?,?,?,?)",
            (_uid(), task_id, run_id, event_type, json.dumps(payload), _now()),
        )
        await self._db.commit()

    async def append_routing_decision(
        self, task_id: str, policy_version: str, route: str, reasons: list[str]
    ) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO routing_decisions (id, task_id, policy_version, route, reasons, decided_at) "
            "VALUES (?,?,?,?,?,?)",
            (_uid(), task_id, policy_version, route, json.dumps(reasons), _now()),
        )
        await self._db.commit()

    async def append_eval_result(self, task_id: str, run_id: str, overall: str, checks: list) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO eval_results (id, task_id, run_id, overall, checks, evaluated_at) "
            "VALUES (?,?,?,?,?,?)",
            (_uid(), task_id, run_id, overall, json.dumps(checks), _now()),
        )
        await self._db.commit()

    async def append_usage(self, task_id: str, run_id: str, input_tokens: int, output_tokens: int, estimated_cost: float | None) -> None:
        assert self._db
        await self._db.execute(
            "INSERT INTO usage_records "
            "(id, task_id, run_id, input_tokens, output_tokens, total_tokens, estimated_cost, recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (_uid(), task_id, run_id, input_tokens, output_tokens,
             input_tokens + output_tokens, estimated_cost, _now()),
        )
        await self._db.commit()

    # ── Reads ─────────────────────────────────────────────────────────────────

    async def get_task(self, task_id: str) -> dict | None:
        assert self._db
        async with self._db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_routing_history(self, policy_version: str | None = None) -> list[dict]:
        assert self._db
        if policy_version:
            sql = "SELECT * FROM routing_decisions WHERE policy_version=? ORDER BY decided_at"
            params = (policy_version,)
        else:
            sql = "SELECT * FROM routing_decisions ORDER BY decided_at"
            params = ()
        async with self._db.execute(sql, params) as cur:
            return [dict(r) async for r in cur]

    async def _task_exists(self, task_id: str) -> bool:
        assert self._db
        async with self._db.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)) as cur:
            return await cur.fetchone() is not None


def _now() -> str:
    return datetime.utcnow().isoformat()

def _uid() -> str:
    return uuid.uuid4().hex

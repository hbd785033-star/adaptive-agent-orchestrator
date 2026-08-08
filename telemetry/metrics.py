"""Task completion metrics — queried from the audit tables."""
from __future__ import annotations

from dataclasses import dataclass
from storage.database import Database


@dataclass
class RoutingMetrics:
    policy_version: str
    route: str
    total: int
    passed: int
    failed: int

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


async def routing_success_by_version(db: Database, policy_version: str | None = None) -> list[RoutingMetrics]:
    """
    Compare routing success rates across policy versions.
    Uses the append-only routing_decisions + eval_results tables.
    """
    conn = db._db
    assert conn, "Database not connected"

    sql = """
        SELECT
            r.policy_version,
            r.route,
            COUNT(*) AS total,
            SUM(CASE WHEN e.overall = 'pass' THEN 1 ELSE 0 END) AS passed,
            SUM(CASE WHEN e.overall = 'fail' THEN 1 ELSE 0 END) AS failed
        FROM routing_decisions r
        LEFT JOIN eval_results e ON r.task_id = e.task_id
        {where}
        GROUP BY r.policy_version, r.route
        ORDER BY r.policy_version, r.route
    """
    where = "WHERE r.policy_version = ?" if policy_version else ""
    params = (policy_version,) if policy_version else ()

    async with conn.execute(sql.format(where=where), params) as cur:
        rows = await cur.fetchall()

    return [
        RoutingMetrics(
            policy_version=row["policy_version"],
            route=row["route"],
            total=row["total"],
            passed=row["passed"] or 0,
            failed=row["failed"] or 0,
        )
        for row in rows
    ]

"""
SQLite schema and migrations.

Two conceptual groups:
  Operational:  tasks, runs, subtasks       — mutable, tracks live state
  Audit:        telemetry_events, eval_results,
                routing_decisions, usage_records — append-only, never updated
"""
from __future__ import annotations

SCHEMA_VERSION = 1

# ── Operational tables ────────────────────────────────────────────────────────

CREATE_TASKS = """
CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    task_type       TEXT NOT NULL,
    goal            TEXT NOT NULL,
    risk            INTEGER NOT NULL DEFAULT 1,
    complexity      INTEGER NOT NULL DEFAULT 1,
    status          TEXT NOT NULL DEFAULT 'received',
    route           TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    parent_task_id  TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""

CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL REFERENCES tasks(id),
    session_id      TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    error           TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
"""

CREATE_SUBTASKS = """
CREATE TABLE IF NOT EXISTS subtasks (
    id              TEXT PRIMARY KEY,
    parent_task_id  TEXT NOT NULL REFERENCES tasks(id),
    child_id        TEXT NOT NULL,
    worktree_path   TEXT,
    branch          TEXT,
    status          TEXT NOT NULL DEFAULT 'allocated',
    created_at      TEXT NOT NULL
);
"""

# ── Audit tables (append-only) ────────────────────────────────────────────────

CREATE_TELEMETRY_EVENTS = """
CREATE TABLE IF NOT EXISTS telemetry_events (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    run_id          TEXT,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,  -- JSON
    recorded_at     TEXT NOT NULL
);
"""

CREATE_EVAL_RESULTS = """
CREATE TABLE IF NOT EXISTS eval_results (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    overall         TEXT NOT NULL,
    checks          TEXT NOT NULL,  -- JSON array
    evaluated_at    TEXT NOT NULL
);
"""

CREATE_ROUTING_DECISIONS = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    route           TEXT NOT NULL,
    reasons         TEXT NOT NULL,  -- JSON array
    decided_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routing_policy ON routing_decisions(policy_version);
"""

CREATE_USAGE_RECORDS = """
CREATE TABLE IF NOT EXISTS usage_records (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    estimated_cost  REAL,
    recorded_at     TEXT NOT NULL
);
"""

CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""

ALL_STATEMENTS: list[str] = [
    CREATE_SCHEMA_VERSION,
    CREATE_TASKS,
    CREATE_RUNS,
    CREATE_SUBTASKS,
    CREATE_TELEMETRY_EVENTS,
    CREATE_EVAL_RESULTS,
    CREATE_ROUTING_DECISIONS,
    CREATE_USAGE_RECORDS,
]

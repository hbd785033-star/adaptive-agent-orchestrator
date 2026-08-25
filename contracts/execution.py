"""Versioned success criteria, outcomes, and portable execution records."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

CriterionType = Literal[
    "pytest",
    "command",
    "file_exists",
    "file_contains",
    "git_diff",
    "registered",
    "output_equals",
]


class SuccessCriterion(BaseModel):
    type: CriterionType
    target: str | None = None
    value: str | None = None
    command: list[str] | str | None = None
    name: str | None = None
    timeout_s: float = Field(default=120, gt=0)


class CriterionResult(BaseModel):
    criterion: SuccessCriterion
    passed: bool
    detail: str = ""


class TaskOutcome(BaseModel):
    completed: bool = False
    criteria: list[CriterionResult] = Field(default_factory=list)

    @computed_field
    @property
    def passed(self) -> bool:
        return self.completed and bool(self.criteria) and all(item.passed for item in self.criteria)


class ExecutionRecord(BaseModel):
    """Provider-neutral AAO output consumed by independent evaluators."""

    model_config = {"extra": "forbid"}

    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    run_id: str | None = None
    model: str | None = None
    provider: str | None = None
    harness: str = "adaptive-agent-orchestrator"
    status: Literal["completed", "failed", "cancelled", "timeout"]
    started_at: datetime
    finished_at: datetime
    latency_seconds: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    tool_calls: list[dict[str, Any]] | None = None
    files_changed: list[str] | None = None
    output: str | None = None
    workspace_root: str | None = None
    isolation_level: Literal["none", "workspace", "os"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def export(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

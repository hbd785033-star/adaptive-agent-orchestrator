"""Eval Gate result types."""
from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
from datetime import datetime


class EvalStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class EvalCheck(BaseModel):
    name: str
    status: EvalStatus
    detail: str = ""
    blocker: bool = True  # FAIL on a blocker → overall FAIL


class EvalResult(BaseModel):
    task_id: str
    run_id: str
    overall: EvalStatus
    checks: list[EvalCheck] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def aggregate(cls, task_id: str, run_id: str, checks: list[EvalCheck]) -> "EvalResult":
        failed_blockers = [c for c in checks if c.blocker and c.status == EvalStatus.FAIL]
        overall = EvalStatus.FAIL if failed_blockers else EvalStatus.PASS
        return cls(task_id=task_id, run_id=run_id, overall=overall, checks=checks)

    def failed_checks(self) -> list[EvalCheck]:
        return [c for c in self.checks if c.status == EvalStatus.FAIL]

"""Runtime handles and results returned from AgentRuntime."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVAL_REQUIRED = "approval_required"


class RunHandle(BaseModel):
    run_id: str
    task_id: str
    session_id: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=_sum_observed(self.input_tokens, other.input_tokens),
            output_tokens=_sum_observed(self.output_tokens, other.output_tokens),
            total_tokens=_sum_observed(self.total_tokens, other.total_tokens),
            estimated_cost_usd=_sum_observed(
                self.estimated_cost_usd, other.estimated_cost_usd
            ),
        )


def _sum_observed(left: int | float | None, right: int | float | None):
    return None if left is None or right is None else left + right


class AgentEvent(BaseModel):
    """Normalised event from any AgentRuntime implementation."""
    id: str
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    type: Literal[
        "message",
        "tool_start",
        "tool_complete",
        "approval_request",
        "clarify_request",
        "subagent_started",
        "subagent_completed",
        "usage",
        "error",
        "completed",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)

    # Typed payload — populated by adapter layer after parsing raw payload
    typed_payload: Any | None = Field(default=None, exclude=True)


# ── Typed payloads for events that Eval Gate inspects ────────────────────────

class ToolCompletePayload(BaseModel):
    tool_name: str
    files_written: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    duration_ms: int = 0


class CompletedPayload(BaseModel):
    summary: str = ""
    files_changed: list[str] = Field(default_factory=list)
    tests_run: bool = False
    unresolved_risks: list[str] = Field(default_factory=list)


class ErrorPayload(BaseModel):
    code: str
    message: str
    recoverable: bool = False


# ── Final result ──────────────────────────────────────────────────────────────

class AgentResult(BaseModel):
    run_id: str | None
    task_id: str
    status: RunStatus
    usage: Usage | None = None
    files_changed: list[str] = Field(default_factory=list)
    summary: str = ""
    unresolved_risks: list[str] = Field(default_factory=list)
    tests_run: bool = False
    error: str | None = None
    completed_at: datetime = Field(default_factory=datetime.utcnow)

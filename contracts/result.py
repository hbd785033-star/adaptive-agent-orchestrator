"""Runtime handles and results returned from AgentRuntime."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class RunStatus(str, Enum):
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
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=(
                (self.estimated_cost_usd or 0) + (other.estimated_cost_usd or 0)
            ),
        )


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
    run_id: str
    task_id: str
    status: RunStatus
    usage: Usage = Field(default_factory=Usage)
    files_changed: list[str] = Field(default_factory=list)
    summary: str = ""
    unresolved_risks: list[str] = Field(default_factory=list)
    tests_run: bool = False
    error: str | None = None
    completed_at: datetime = Field(default_factory=datetime.utcnow)

"""
Delegation contracts — formal types for multi-agent execution results.

These are the stable interfaces that Engine, Eval, Telemetry, and future
Model Council integrations all read from. Define them early so all
downstream consumers share the same shape.
"""
from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from contracts.evaluation import EvalResult
from contracts.result import AgentResult, Usage


class ChildExecution(BaseModel):
    """Result of one child agent within a delegation run."""

    child_id: str
    run_id: str

    status: Literal["completed", "failed", "cancelled", "timeout"] = "failed"

    result: AgentResult | None = None
    eval_result: EvalResult | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    retry_count: int = 0
    duration_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return (
            self.status == "completed"
            and self.eval_result is not None
            and self.eval_result.overall.value == "pass"
        )

    @property
    def usage(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
        )


class DelegationResult(BaseModel):
    """Aggregate result of a complete delegation execution."""

    parent_task_id: str
    children: list[ChildExecution] = Field(default_factory=list)

    # Filled after all children finish
    aggregate_result: AgentResult | None = None
    started_at_ms: int = Field(default_factory=lambda: int(time.monotonic() * 1000))
    finished_at_ms: int = 0

    @property
    def successful(self) -> int:
        return sum(1 for c in self.children if c.succeeded)

    @property
    def failed(self) -> int:
        return len(self.children) - self.successful

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.children)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.children)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.estimated_cost_usd for c in self.children)

    @property
    def duration_ms(self) -> int:
        return self.finished_at_ms - self.started_at_ms

    @property
    def overall_status(self) -> Literal["completed", "partial_failed", "failed"]:
        if self.failed == 0:
            return "completed"
        if self.successful > 0:
            return "partial_failed"
        return "failed"

    def get_child(self, child_id: str) -> ChildExecution | None:
        return next((c for c in self.children if c.child_id == child_id), None)

    def failed_children(self) -> list[ChildExecution]:
        return [c for c in self.children if not c.succeeded]

    def successful_children(self) -> list[ChildExecution]:
        return [c for c in self.children if c.succeeded]


class ExecutionPlan(BaseModel):
    """
    Router output — richer than just a route string.
    Engine and DelegationExecutor consume this.
    """

    route: Literal["single", "delegation"]
    children: int = 1                           # always 1 for single
    workspace_mode: Literal["readonly", "shared", "isolated"] = "shared"
    max_retries: int = 1
    reasons: list[str] = Field(default_factory=list)
    policy_version: str = "routing-v1.0"

    def to_dict(self) -> dict:
        return self.model_dump()

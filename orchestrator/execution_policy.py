"""Shared R2 execution authorization for single and delegated runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from contracts.task import TaskContract


class ApprovalOutcome(StrEnum):
    ALLOWED = "allowed"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    outcome: ApprovalOutcome
    reason: str = ""


@dataclass(frozen=True)
class ExecutionPolicy:
    always_require_actions: set[str] = field(default_factory=set)
    require_approval_above_calls: int = 5
    max_total_calls: int = 8
    approval_timeout_s: float = 120

    def authorize_event(
        self,
        task: TaskContract,
        event: dict,
        *,
        calls_used: int,
        approval: bool | None = None,
    ) -> PolicyDecision:
        """Authorize one next call/event; exact limit means no further call fits."""
        if calls_used >= self.max_total_calls:
            return PolicyDecision(False, ApprovalOutcome.BUDGET_EXCEEDED, "call budget exhausted")
        action = str(event.get("action", ""))
        requires = (
            action in self.always_require_actions
            or calls_used >= self.require_approval_above_calls
        )
        if not requires:
            return PolicyDecision(True, ApprovalOutcome.ALLOWED)
        if approval is True:
            return PolicyDecision(True, ApprovalOutcome.APPROVED)
        if approval is False:
            return PolicyDecision(False, ApprovalOutcome.DENIED, "approval denied")
        return PolicyDecision(False, ApprovalOutcome.TIMEOUT, "approval missing or timed out")

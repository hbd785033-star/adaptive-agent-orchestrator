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

    def reject_runtime_approval_request(
        self,
        task: TaskContract,
        *,
        calls_used: int,
        approval: bool | None = None,
    ) -> PolicyDecision:
        """Runtime events are observational; an approval request cannot resume in R2."""
        del task, calls_used, approval
        return PolicyDecision(
            False,
            ApprovalOutcome.DENIED,
            "runtime approval requests fail closed; no resume capability",
        )

    def authorize_submission(
        self,
        task: TaskContract,
        *,
        calls_used: int,
        planned_actions: set[str] | None = None,
        approval: bool | None = None,
    ) -> PolicyDecision:
        """Authorize known AAO-owned inputs before runtime submission."""
        del task
        if calls_used >= self.max_total_calls:
            return PolicyDecision(False, ApprovalOutcome.BUDGET_EXCEEDED, "call budget exhausted")
        protected_action = bool(
            (planned_actions or set()).intersection(self.always_require_actions)
        )
        protected_call = calls_used >= self.require_approval_above_calls
        if not protected_action and not protected_call:
            return PolicyDecision(True, ApprovalOutcome.ALLOWED)
        if approval is True:
            return PolicyDecision(True, ApprovalOutcome.APPROVED)
        if approval is False:
            return PolicyDecision(False, ApprovalOutcome.DENIED, "approval denied")
        return PolicyDecision(False, ApprovalOutcome.TIMEOUT, "approval missing or timed out")

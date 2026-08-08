"""
Budget Gate and Approval Gate — v1.

Budget Gate: hard limits on calls, children, depth, retries.
Approval Gate: CLI yes/no for high-risk or over-budget tasks.

Both gates are synchronous decision points; they do NOT modify task state
themselves — the caller is responsible for persisting the outcome.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import yaml

from contracts.task import TaskContract, RiskLevel


# ── Budget ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BudgetConfig:
    max_children: int = 2
    max_depth: int = 1
    max_retries: int = 1
    max_total_calls: int = 8
    require_approval_above_calls: int = 5


@dataclasses.dataclass
class BudgetState:
    """Mutable budget tracker — one instance per task execution."""
    task_id: str
    config: BudgetConfig
    children_used: int = 0
    retries_used: int = 0
    calls_used: int = 0
    depth_used: int = 0

    def check_children(self) -> "BudgetViolation | None":
        if self.children_used >= self.config.max_children:
            return BudgetViolation(
                "max_children",
                f"children={self.children_used} >= limit={self.config.max_children}",
            )
        return None

    def check_retries(self) -> "BudgetViolation | None":
        if self.retries_used >= self.config.max_retries:
            return BudgetViolation(
                "max_retries",
                f"retries={self.retries_used} >= limit={self.config.max_retries}",
            )
        return None

    def check_calls(self) -> "BudgetViolation | None":
        if self.calls_used >= self.config.max_total_calls:
            return BudgetViolation(
                "max_total_calls",
                f"calls={self.calls_used} >= limit={self.config.max_total_calls}",
            )
        return None

    def needs_approval_for_calls(self) -> bool:
        return self.calls_used >= self.config.require_approval_above_calls


@dataclasses.dataclass
class BudgetViolation:
    field: str
    detail: str


# ── Approval Gate ─────────────────────────────────────────────────────────────

class ApprovalGate:
    """
    v1 implementation: CLI yes/no prompt.
    v2 will replace this with persistent async approval (LangGraph interrupt).
    """

    def __init__(self, policy_path: str | Path = "policies/default.yaml") -> None:
        raw = yaml.safe_load(Path(policy_path).read_text())
        cfg = raw.get("approval", {})
        self._always_require: set[str] = set(cfg.get("always_require", []))
        self._require_for_risk: set[int] = set(cfg.get("require_for_risk_levels", [3, 4]))

    def requires_approval(self, task: TaskContract, actions: list[str] | None = None) -> tuple[bool, str]:
        """
        Returns (needs_approval: bool, reason: str).
        """
        if task.risk.value in self._require_for_risk:
            return True, f"risk={task.risk.name} requires approval"

        triggered = self._always_require & set(actions or [])
        if triggered:
            return True, f"actions {triggered} always require approval"

        return False, ""

    def prompt_user(self, reason: str, task: TaskContract) -> bool:
        """
        Blocking CLI prompt. Returns True if user approves.
        In non-interactive contexts (CI, tests) this should be bypassed
        by injecting an ApprovalGate subclass.
        """
        print(f"\n⚠️  Approval required: {reason}")
        print(f"   Task: {task.goal[:80]}")
        answer = input("   Proceed? [y/N] ").strip().lower()
        return answer in ("y", "yes")

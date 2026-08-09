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

import yaml

from contracts.task import TaskContract

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
    calls_reserved: int = 0
    depth_used: int = 0

    def check_children(self) -> BudgetViolation | None:
        if self.children_used >= self.config.max_children:
            return BudgetViolation(
                "max_children",
                f"children={self.children_used} >= limit={self.config.max_children}",
            )
        return None

    def check_retries(self) -> BudgetViolation | None:
        if self.retries_used >= self.config.max_retries:
            return BudgetViolation(
                "max_retries",
                f"retries={self.retries_used} >= limit={self.config.max_retries}",
            )
        return None

    def can_submit_calls(self, count: int = 1) -> BudgetViolation | None:
        """Preflight check: would starting ``count`` calls exceed the limit?"""
        projected = self.calls_used + self.calls_reserved + count
        if count < 1:
            raise ValueError("count must be positive")
        if projected > self.config.max_total_calls:
            return BudgetViolation(
                "max_total_calls",
                f"calls={self.calls_used} reserved={self.calls_reserved} requested={count} "
                f"> limit={self.config.max_total_calls}",
            )
        return None

    def reserve_calls(self, count: int = 1) -> BudgetViolation | None:
        violation = self.can_submit_calls(count)
        if violation is None:
            self.calls_reserved += count
        return violation

    def commit_reserved_call(self) -> None:
        if self.calls_reserved < 1:
            raise RuntimeError("no reserved call to commit")
        self.calls_reserved -= 1
        self.calls_used += 1

    def release_reserved_call(self) -> None:
        if self.calls_reserved < 1:
            raise RuntimeError("no reserved call to release")
        self.calls_reserved -= 1

    def is_over_budget(self) -> BudgetViolation | None:
        """Postflight invariant: using exactly the limit is valid."""
        if self.calls_used > self.config.max_total_calls:
            return BudgetViolation(
                "max_total_calls",
                f"calls={self.calls_used} > limit={self.config.max_total_calls}",
            )
        return None

    def check_calls(self) -> BudgetViolation | None:
        """Backward-compatible preflight alias."""
        return self.can_submit_calls(1)

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

    def prompt_user(self, reason: str, task: TaskContract, timeout_s: int = 120) -> bool:
        """
        Blocking CLI prompt with timeout. Returns True if user approves.
        Defaults to DENY on timeout — never hangs a pipeline indefinitely.
        In non-interactive contexts (CI, tests) inject an ApprovalGate subclass.
        """
        import threading

        print(f"\n⚠️  Approval required: {reason}")
        print(f"   Task: {task.goal[:80]}")
        print(f"   Proceed? [y/N]  (auto-DENY in {timeout_s}s) ", end="", flush=True)

        answer: list[str] = []
        answered = threading.Event()

        def _read() -> None:
            try:
                answer.append(input())
                answered.set()
            except EOFError:
                answered.set()  # non-interactive / pipe → treat as no

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        if not answered.wait(timeout_s):
            print("\n   [timeout — auto-denied]")
            return False

        # answer is empty when stdin was closed (EOF) before any input was read
        return bool(answer) and answer[0].strip().lower() in ("y", "yes")

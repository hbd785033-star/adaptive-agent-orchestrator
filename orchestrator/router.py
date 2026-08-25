"""
Rule-based Router — v1.

Classifies a TaskContract as either "single" or "delegation".
Decision + reasons are written to routing_decisions (append-only).

Rules are read from policies/default.yaml at startup.
The policy_version field is stored with every decision so A/B
comparison queries work: SELECT route, eval_passed FROM ... WHERE policy_version='x'.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import yaml

from contracts.execution import SuccessCriterion
from contracts.task import TaskContract

Route = Literal["single", "delegation"]


@dataclasses.dataclass
class RoutingDecision:
    route: Route
    reasons: list[str]
    policy_version: str

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "reasons": self.reasons,
            "policy_version": self.policy_version,
        }


# ── Token estimator (heuristic v1) ───────────────────────────────────────────

def _criterion_estimation_text(criterion: str | SuccessCriterion) -> str:
    """Return deterministic criterion text without evaluating the criterion."""
    if isinstance(criterion, str):
        return criterion
    return criterion.model_dump_json(exclude_none=True)


def estimate_input_tokens(task: TaskContract) -> int:
    """
    Rough character-to-token ratio. Accurate enough to gate routing.
    Replace with tiktoken if precision matters later.
    """  # heuristic v1
    criteria_text = " ".join(
        _criterion_estimation_text(criterion) for criterion in task.success_criteria
    )
    text = task.goal + str(task.context) + criteria_text
    return int(len(text) / 3.5)


# ── Router ────────────────────────────────────────────────────────────────────

class RuleRouter:
    def __init__(self, policy_path: str | Path = "policies/default.yaml") -> None:
        raw = yaml.safe_load(Path(policy_path).read_text())
        self._policy_version: str = raw.get("policy_version", "routing-v1.0")
        cfg = raw.get("routing", {})
        delg = cfg.get("delegation", {})
        single = cfg.get("single", {})
        constraints = cfg.get("constraints", {})

        # Delegation triggers
        self._min_independent_subtasks: int = delg.get("min_independent_subtasks", 2)
        self._min_estimated_input_tokens: int = delg.get("min_estimated_input_tokens", 8000)
        self._allowed_task_types: set[str] = set(delg.get("allowed_task_types", []))

        # Single triggers
        self._max_complexity: int = single.get("max_complexity", 2)
        self._max_affected_modules: int = single.get("max_affected_modules", 1)

        # Constraints
        self._sequential_dep_forces_single: bool = constraints.get(
            "sequential_dependency_forces_single", True
        )

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def route(
        self,
        task: TaskContract,
        *,
        independent_subtask_count: int = 0,
        has_sequential_dependency: bool = False,
        affected_module_count: int = 1,
    ) -> RoutingDecision:
        """
        Evaluate routing rules in priority order.
        First matching rule wins; all reasons are recorded for telemetry.

        Parameters
        ----------
        task:
            The TaskContract to route.
        independent_subtask_count:
            Number of parallelisable subtasks detected by the profiler.
        has_sequential_dependency:
            True if subtasks must run in order (cannot parallelise).
        affected_module_count:
            Number of distinct modules / top-level paths the task touches.
        """
        reasons: list[str] = []

        # ── Hard single rules (checked first) ────────────────────────────────

        # An explicit, validated subtask plan is an execution plan rather than a
        # profiler hint. Dependency edges are handled by the DAG scheduler.
        if task.subtasks:
            reasons.append(f"explicit_subtasks={len(task.subtasks)} → delegation")
            if has_sequential_dependency:
                reasons.append("dependency_edges=true → scheduled in DAG waves")
            return RoutingDecision("delegation", reasons, self._policy_version)

        if self._sequential_dep_forces_single and has_sequential_dependency:
            reasons.append("sequential_dependency=true → forces single")
            return RoutingDecision("single", reasons, self._policy_version)

        # ── Delegation triggers (before single complexity check) ──────────────
        # task_type and subtask count are checked before complexity/module
        # thresholds so that an explicit delegation task_type is not swallowed
        # by a low-complexity single rule.

        delegation_signals: list[str] = []

        if task.task_type.value in self._allowed_task_types:
            delegation_signals.append(f"task_type={task.task_type.value} in allowed_task_types")

        if independent_subtask_count >= self._min_independent_subtasks:
            delegation_signals.append(
                f"independent_subtasks={independent_subtask_count} >= {self._min_independent_subtasks}"
            )

        estimated_tokens = estimate_input_tokens(task)
        if estimated_tokens >= self._min_estimated_input_tokens:
            delegation_signals.append(
                f"estimated_input_tokens={estimated_tokens} >= {self._min_estimated_input_tokens}"
            )

        if delegation_signals:
            reasons.extend(delegation_signals)
            return RoutingDecision("delegation", reasons, self._policy_version)

        # ── Single complexity/module rule (fallback) ──────────────────────────

        if task.complexity <= self._max_complexity and affected_module_count <= self._max_affected_modules:
            reasons.append(
                f"complexity={task.complexity} <= {self._max_complexity}"
                f" AND affected_modules={affected_module_count} <= {self._max_affected_modules}"
                " → single"
            )
            return RoutingDecision("single", reasons, self._policy_version)

        # ── Default ───────────────────────────────────────────────────────────
        reasons.append("no delegation triggers matched → default single")
        return RoutingDecision("single", reasons, self._policy_version)

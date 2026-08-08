"""
Task Profiler — extracts routing signals from a TaskContract.

The profiler does NOT make routing decisions; it returns measurable
attributes (subtask count, module count, sequential dependency flag)
that the RuleRouter consumes.

v1: rule-based heuristics over the task goal and context.
v2: replace with a lightweight LLM call or a trained classifier
    once telemetry has accumulated enough labelled examples.
"""
from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass
class TaskProfile:
    """Routing signals extracted from a TaskContract."""
    estimated_input_tokens: int
    independent_subtask_count: int
    has_sequential_dependency: bool
    affected_module_count: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# Keywords that suggest parallel independent subtasks
_PARALLEL_SIGNALS = [
    r"\band\b.*\band\b",         # "fix X and add Y and update Z"
    r"\bseparate(ly)?\b",
    r"\bindepend(ent|ently)?\b",
    r"\bparallel\b",
    r"\bsimultaneous(ly)?\b",
    r"\b(also|additionally|furthermore)\b",
    r"(?i)both\s+\w+\s+and\s+\w+",
]

# Keywords that suggest sequential dependency
_SEQUENTIAL_SIGNALS = [
    r"\bafter\b",
    r"\bthen\b",
    r"\bfirst\b.+\bthen\b",
    r"\bonce\b.+\b(done|complete|finished)\b",
    r"\bdepend(s)? on\b",
    r"\bfollow(ing|ed by)?\b",
]


def _count_path_groups(allowed_paths: list[str]) -> int:
    """Estimate distinct module count from allowed_paths globs."""
    tops = set()
    for p in allowed_paths:
        top = p.split("/")[0].replace("**", "").replace("*", "").strip()
        if top:
            tops.add(top)
    return max(len(tops), 1) if allowed_paths else 1


class TaskProfiler:
    """
    Extract routing signals from a TaskContract.

    Usage:
        profiler = TaskProfiler()
        profile = profiler.profile(task)
    """

    def profile(self, task) -> TaskProfile:  # task: TaskContract (avoid circular import)
        from orchestrator.router import estimate_input_tokens

        goal_lower = task.goal.lower()
        context_text = str(task.context)

        # Token estimate
        estimated_tokens = estimate_input_tokens(task)

        # Parallel signals
        parallel_hits = sum(
            1 for pattern in _PARALLEL_SIGNALS
            if re.search(pattern, goal_lower)
        )
        # Heuristic: each parallel signal → +1 subtask (minimum 1)
        independent_count = max(1, parallel_hits + (1 if parallel_hits else 0))

        # Sequential dependency — overrides parallel signals
        has_sequential = any(
            re.search(p, goal_lower) for p in _SEQUENTIAL_SIGNALS
        )

        # Module count from allowed_paths
        module_count = _count_path_groups(task.allowed_paths)

        return TaskProfile(
            estimated_input_tokens=estimated_tokens,
            independent_subtask_count=independent_count,
            has_sequential_dependency=has_sequential,
            affected_module_count=module_count,
        )

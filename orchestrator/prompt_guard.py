"""
Prompt Guard — encodes TaskContract constraints into the agent prompt.

This is the v1 "execution-before" guard: we can't intercept tool calls
in real-time (that's v2), so we embed the constraints clearly in the
goal/context so the agent knows its boundaries upfront.

The guard produces an *augmented* TaskContract whose `goal` field
carries the original goal plus a structured constraints block that
a capable LLM will follow. The original contract is never mutated.

Post-execution verification is done by DeterministicEvalGate (paths,
secrets, budget checks) — this guard and the eval gate are complementary.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from contracts.task import TaskContract, TaskType, WorkspaceSpec

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_constraints(
    task: TaskContract,
    worktree_path: Path | None = None,
    child_id: str | None = None,
) -> TaskContract:
    """
    Return a *copy* of `task` with constraints encoded into the goal text.

    Args:
        task:          Original contract (not mutated).
        worktree_path: If provided, overrides task.workspace.path and injects
                       the exact directory the agent must work in.
        child_id:      Optional child identifier for multi-agent prompts.
    """
    augmented = deepcopy(task)

    constraints_block = _build_constraints_block(task, worktree_path, child_id)
    if constraints_block:
        augmented.goal = task.goal + "\n\n" + constraints_block

    # Inject workspace into context so downstream serialisers include it
    if worktree_path is not None:
        ws = WorkspaceSpec(
            path=str(worktree_path),
            branch=f"agent/{task.id}/{child_id}" if child_id else f"agent/{task.id}",
        )
        augmented.workspace = ws

    return augmented


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_constraints_block(
    task: TaskContract,
    worktree_path: Path | None,
    child_id: str | None,
) -> str:
    """Build a human-readable constraints section to append to the goal."""
    lines: list[str] = []

    # Header
    lines.append("---")
    lines.append("## ORCHESTRATOR CONSTRAINTS — READ BEFORE ACTING")
    lines.append("")
    lines.append("These constraints are enforced both in your prompt and by an")
    lines.append("automated post-execution gate. Violating them will cause the")
    lines.append("task to fail and trigger a retry.")
    lines.append("")

    if task.task_type in {TaskType.CODE_REVIEW, TaskType.PARALLEL_RESEARCH}:
        lines.append("### READ-ONLY EXECUTION")
        lines.append("Do not create, modify, delete, move, or rename any repository file.")
        lines.append("")

    # Workspace
    effective_path = str(worktree_path) if worktree_path else (
        task.workspace.path if task.workspace else None
    )
    if effective_path:
        lines.append("### Workspace")
        lines.append(f"Work exclusively inside: `{effective_path}`")
        if child_id:
            lines.append(f"Your agent ID: `{child_id}`")
        lines.append("")

    # Allowed paths
    if task.allowed_paths:
        lines.append("### Allowed file paths (you may ONLY read/write these)")
        for p in task.allowed_paths:
            lines.append(f"  - {p}")
        lines.append("")
        lines.append("Any file outside these paths MUST NOT be modified.")
        lines.append("")

    # Forbidden actions
    if task.forbidden_actions:
        lines.append("### Forbidden actions (do NOT perform these under any circumstance)")
        for a in task.forbidden_actions:
            lines.append(f"  - {a}")
        lines.append("")

    # Success criteria
    if task.success_criteria:
        lines.append("### Success criteria (your output MUST satisfy all of these)")
        for i, c in enumerate(task.success_criteria, 1):
            rendered = c if isinstance(c, str) else c.model_dump_json(exclude_none=True)
            lines.append(f"  {i}. {rendered}")
        lines.append("")

    # Output schema
    if task.output_schema:
        lines.append("### Required output fields (include all of these in your final summary)")
        for f in task.output_schema:
            lines.append(f"  - {f}")
        lines.append("")

    # If nothing was added beyond the header, return empty string
    has_content = bool(
        task.allowed_paths or task.forbidden_actions
        or task.success_criteria or task.output_schema
        or effective_path
    )
    if not has_content:
        return ""

    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Delegation helper — split a contract into N child contracts
# ---------------------------------------------------------------------------

def split_for_delegation(
    task: TaskContract,
    worktrees: list[tuple[str, Path | None]],
) -> list[TaskContract]:
    """
    Produce one augmented child contract per worktree.

    Each child gets:
    - The same goal + constraints block
    - Its own workspace path injected
    - A context note identifying it as child N of M
    """
    children = []
    n = len(worktrees)
    if task.subtasks and len(task.subtasks) != n:
        raise ValueError(
            f"explicit subtask count ({len(task.subtasks)}) must match child count ({n})"
        )
    for i, (child_id, wt_path) in enumerate(worktrees, 1):
        base = deepcopy(task)
        if task.subtasks:
            subtask = task.subtasks[i - 1]
            base.goal = subtask.goal
            if subtask.allowed_paths:
                base.allowed_paths = list(subtask.allowed_paths)
            base.parent_task_id = task.id
            base.subtasks = []
            base.context["_subtask_id"] = subtask.id
            base.context["_dependencies"] = list(subtask.dependencies)
            base.context["_expected_output"] = subtask.expected_output
            base.context["_decomposition_mode"] = "explicit"
        else:
            base.context["_decomposition_mode"] = "replicated_goal"
        child = inject_constraints(base, worktree_path=wt_path, child_id=child_id)
        # Inject _child_id so DelegationExecutor can correlate results
        child.context["_child_id"] = child_id
        # Prepend delegation context
        delegation_header = (
            f"[Delegation context: you are sub-agent {i}/{n} (id={child_id})."
            f" Coordinate with other sub-agents by staying within your workspace"
            f" and allowed paths only.]\n\n"
        )
        child.goal = delegation_header + child.goal
        children.append(child)
    return children

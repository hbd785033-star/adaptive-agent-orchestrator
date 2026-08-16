"""Structured TaskContract -> cross-runtime planning intake.

Recovery provenance
-------------------
This Phase 3B.1 / 3B.1.1 file is a source-constrained reconstruction from the
preserved Architecture Pivot checkpoint. It is not claimed byte-for-byte
identical to the lost worktree.

Rules intentionally use structured TaskContract fields only. `goal` prose and
`context` are not routing-hint channels.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from contracts.requirements import TaskRequirements
from contracts.task import TaskContract, TaskType
from contracts.task_profile import ComplexityLevel, TaskProfile


class TaskIntake(BaseModel):
    """Typed planning inputs derived conservatively from caller intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: TaskProfile
    requirements: TaskRequirements


def _reasoning_complexity(task: TaskContract) -> ComplexityLevel:
    # TaskContract.complexity is the only structured reasoning/complexity signal
    # available on the legacy caller contract.
    if task.complexity >= 5:
        return ComplexityLevel.HIGH
    if task.complexity >= 3:
        return ComplexityLevel.MEDIUM
    return ComplexityLevel.LOW


def _dependency_shape(task: TaskContract) -> tuple[bool, bool]:
    """Return (parallelizable, cross_role_dependencies) from explicit DAG only."""
    if not task.subtasks:
        return False, False

    cross_role_dependencies = any(subtask.dependencies for subtask in task.subtasks)

    # Phase 3B.1.1 correction:
    # parallelizable is based on the current explicit DAG ready frontier, not
    # merely "subtask_count >= 2". At intake time the initial ready frontier is
    # the set of nodes with no dependencies.
    ready_frontier = [
        subtask for subtask in task.subtasks if not subtask.dependencies
    ]
    parallelizable = len(ready_frontier) >= 2
    return parallelizable, cross_role_dependencies


def _execution_complexity(task: TaskContract) -> ComplexityLevel:
    """Conservative structured-only estimate of execution coordination shape."""
    if task.subtasks:
        if any(subtask.dependencies for subtask in task.subtasks) or len(task.subtasks) >= 3:
            return ComplexityLevel.HIGH
        if len(task.subtasks) >= 2:
            return ComplexityLevel.MEDIUM

    if task.task_type == TaskType.MULTI_FILE_REFACTOR:
        return ComplexityLevel.MEDIUM
    if len(task.allowed_paths) >= 2:
        return ComplexityLevel.MEDIUM
    return ComplexityLevel.LOW


def derive_task_profile(task: TaskContract) -> TaskProfile:
    """Derive task shape without NLP, hidden context hints, or runtime selection."""
    parallelizable, cross_role_dependencies = _dependency_shape(task)
    return TaskProfile(
        reasoning_complexity=_reasoning_complexity(task),
        execution_complexity=_execution_complexity(task),
        persistent_execution=False,
        long_running=False,
        parallelizable=parallelizable,
        cross_role_dependencies=cross_role_dependencies,
        human_in_loop=False,
        decision_diversity=False,
        external_effects=False,
    )


def derive_task_requirements(task: TaskContract) -> TaskRequirements:
    """Derive only capabilities justified by structured task type/scope."""
    write_types = {
        TaskType.CODE_FIX,
        TaskType.MULTI_FILE_REFACTOR,
        TaskType.TEST_AND_IMPLEMENT,
    }
    code_types = {
        TaskType.CODE_FIX,
        TaskType.MULTI_FILE_REFACTOR,
        TaskType.TEST_AND_IMPLEMENT,
        TaskType.CODE_REVIEW,
    }

    filesystem_write = task.task_type in write_types
    filesystem_read = (
        task.task_type in code_types
        or bool(task.allowed_paths)
        or any(subtask.allowed_paths for subtask in task.subtasks)
    )
    shell = task.task_type in write_types
    tests = task.task_type in write_types
    web = task.task_type == TaskType.PARALLEL_RESEARCH

    return TaskRequirements(
        filesystem_read=filesystem_read,
        filesystem_write=filesystem_write,
        shell=shell,
        tests=tests,
        web=web,
        background_execution=False,
        persistent_tasks=False,
        human_in_loop=False,
    )


def intake_task(task: TaskContract) -> TaskIntake:
    """Build typed planning inputs from structured caller intent."""
    return TaskIntake(
        profile=derive_task_profile(task),
        requirements=derive_task_requirements(task),
    )

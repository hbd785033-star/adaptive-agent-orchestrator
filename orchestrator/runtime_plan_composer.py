"""Pure RuntimePlan composition from completed upstream decisions."""
from __future__ import annotations

from contracts.execution_mode import ExecutionModeDecision
from contracts.runtime_plan import RuntimePlan
from contracts.runtime_selection import RuntimeSelectionDecision


def compose_runtime_plan(
    selection: RuntimeSelectionDecision,
    execution_mode: ExecutionModeDecision,
    *,
    plan_policy_version: str,
    planner: str | None = None,
    reviewer: str | None = None,
    approval_required: bool = False,
) -> RuntimePlan:
    """Compose a RuntimePlan without rerouting, execution, health checks, or I/O."""
    if not plan_policy_version.strip():
        raise ValueError("plan_policy_version must not be blank")
    if selection.selected_runtime is None:
        raise ValueError("cannot compose RuntimePlan without selected runtime")
    if execution_mode.runtime is None or execution_mode.mode is None:
        raise ValueError("cannot compose RuntimePlan without execution mode")
    if selection.selected_runtime != execution_mode.runtime:
        raise ValueError("selection runtime must match execution mode runtime")

    reasons = [
        *(f"selection: {reason}" for reason in selection.reasons),
        *(f"execution_mode: {reason}" for reason in execution_mode.reasons),
    ]
    return RuntimePlan(
        planner=planner,
        executor=selection.selected_runtime,
        execution_mode=execution_mode.mode,
        reviewer=reviewer,
        fallback=None,
        policy_version=plan_policy_version,
        selection_policy_version=selection.policy_version,
        execution_mode_policy_version=execution_mode.policy_version,
        reasons=reasons,
        approval_required=approval_required,
    )

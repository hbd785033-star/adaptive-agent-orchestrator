"""Deterministic execution mode decisions for a selected runtime."""
from __future__ import annotations

from adapters.runtime import RuntimeCapabilities
from contracts.execution_mode import (
    ExecutionModeDecision,
    ExecutionModeDecisionCode,
    ExecutionModePolicy,
)
from contracts.runtime_selection import RuntimeSelectionDecision
from contracts.task_profile import ComplexityLevel, TaskProfile


def _kanban_reason(task_profile: TaskProfile) -> str | None:
    if task_profile.persistent_execution:
        return "persistent_execution=True"
    if task_profile.long_running:
        return "long_running=True"
    if task_profile.cross_role_dependencies:
        return "cross_role_dependencies=True"
    if task_profile.execution_complexity == ComplexityLevel.HIGH and task_profile.parallelizable:
        return "execution_complexity=high and parallelizable=True"
    return None


def select_execution_mode(
    task_profile: TaskProfile,
    runtime_selection: RuntimeSelectionDecision,
    runtime_capabilities: RuntimeCapabilities,
    policy: ExecutionModePolicy,
) -> ExecutionModeDecision:
    """Select only the execution mode; do not build RuntimePlan or execute runtime."""
    if runtime_selection.selected_runtime is None:
        return ExecutionModeDecision(
            runtime=None,
            mode=None,
            policy_version=policy.policy_version,
            decision_code=ExecutionModeDecisionCode.NO_SELECTED_RUNTIME,
            reasons=["no runtime was selected"],
        )

    runtime = runtime_selection.selected_runtime

    # V1 integration boundary: Codex and Claude Code own their internal topology.
    # Do not expand this into a runtime registry or AAO orchestration layer here.
    if runtime in {"codex", "claude_code"}:
        return ExecutionModeDecision(
            runtime=runtime,
            mode="native",
            policy_version=policy.policy_version,
            decision_code=ExecutionModeDecisionCode.SELECTED_NATIVE,
            reasons=[f"selected runtime owns its native execution topology: {runtime}"],
        )

    if runtime == "hermes":
        kanban_reason = _kanban_reason(task_profile)
        if kanban_reason and policy.allow_kanban and runtime_capabilities.native_kanban:
            return ExecutionModeDecision(
                runtime=runtime,
                mode="kanban",
                policy_version=policy.policy_version,
                decision_code=ExecutionModeDecisionCode.SELECTED_KANBAN,
                reasons=[
                    f"selected kanban execution because {kanban_reason} and runtime supports native_kanban"
                ],
            )
        if task_profile.parallelizable and policy.allow_delegate and runtime_capabilities.native_delegation:
            return ExecutionModeDecision(
                runtime=runtime,
                mode="delegate",
                policy_version=policy.policy_version,
                decision_code=ExecutionModeDecisionCode.SELECTED_DELEGATE,
                reasons=[
                    "selected delegate execution because task is parallelizable and runtime supports native_delegation"
                ],
            )
        return ExecutionModeDecision(
            runtime=runtime,
            mode="direct",
            policy_version=policy.policy_version,
            decision_code=ExecutionModeDecisionCode.SELECTED_DIRECT,
            reasons=[f"selected direct execution for runtime: {runtime}"],
        )

    return ExecutionModeDecision(
        runtime=runtime,
        mode="direct",
        policy_version=policy.policy_version,
        decision_code=ExecutionModeDecisionCode.SELECTED_DIRECT,
        reasons=[f"selected generic direct execution for runtime: {runtime}"],
    )

from __future__ import annotations

from datetime import UTC, datetime

from adapters.runtime import RuntimeCapabilities
from contracts.execution_mode import ExecutionModePolicy
from contracts.requirements import TaskRequirements
from contracts.runtime_health import RuntimeHealth
from contracts.runtime_selection import RuntimeSelectionPolicy
from contracts.task_profile import TaskProfile
from orchestrator.candidate_filter import RuntimeCandidate
from orchestrator.planning_pipeline import plan_runtime


def candidate(runtime: str, *, status: str = "available", **caps: bool) -> RuntimeCandidate:
    return RuntimeCandidate(
        runtime=runtime,
        capabilities=RuntimeCapabilities(**caps),
        health=RuntimeHealth(
            runtime=runtime,
            status=status,
            checked_at=datetime.now(UTC),
        ),
    )


def test_planning_pipeline_composes_selected_candidate_only() -> None:
    result = plan_runtime(
        TaskProfile(),
        TaskRequirements(filesystem_write=True),
        [
            candidate("hermes", filesystem_write=False),
            candidate("codex", filesystem_write=True),
        ],
        RuntimeSelectionPolicy(
            policy_version="selection-test-v1",
            runtime_priority=("hermes", "codex"),
        ),
        ExecutionModePolicy(policy_version="mode-test-v1"),
        plan_policy_version="plan-test-v1",
        reviewer="claude_code",
    )

    assert [assessment.eligible for assessment in result.assessments] == [False, True]
    assert result.selection.selected_runtime == "codex"
    assert result.mode is not None
    assert result.mode.runtime == "codex"
    assert result.mode.mode == "native"
    assert result.plan is not None
    assert result.plan.executor == "codex"
    assert result.plan.reviewer == "claude_code"
    assert result.plan.selection_policy_version == "selection-test-v1"
    assert result.plan.execution_mode_policy_version == "mode-test-v1"


def test_no_selection_does_not_fabricate_mode_or_plan() -> None:
    result = plan_runtime(
        TaskProfile(),
        TaskRequirements(filesystem_write=True),
        [candidate("hermes", filesystem_write=False)],
        RuntimeSelectionPolicy(
            policy_version="selection-test-v1",
            runtime_priority=("hermes",),
        ),
        ExecutionModePolicy(policy_version="mode-test-v1"),
        plan_policy_version="plan-test-v1",
    )

    assert result.selection.selected_runtime is None
    assert result.mode is None
    assert result.plan is None


def test_unsupported_selected_runtime_keeps_observed_mode_decision_but_no_plan() -> None:
    result = plan_runtime(
        TaskProfile(),
        TaskRequirements(),
        [candidate("future_runtime")],
        RuntimeSelectionPolicy(
            policy_version="selection-test-v1",
            runtime_priority=("future_runtime",),
        ),
        ExecutionModePolicy(policy_version="mode-test-v1"),
        plan_policy_version="plan-test-v1",
    )

    assert result.selection.selected_runtime == "future_runtime"
    assert result.mode is not None
    assert result.mode.mode is None
    assert result.plan is None


def test_selection_time_degraded_fallback_is_not_execution_fallback() -> None:
    result = plan_runtime(
        TaskProfile(),
        TaskRequirements(),
        [candidate("codex", status="degraded")],
        RuntimeSelectionPolicy(
            policy_version="selection-test-v1",
            runtime_priority=("codex",),
            allow_degraded_fallback=True,
        ),
        ExecutionModePolicy(policy_version="mode-test-v1"),
        plan_policy_version="plan-test-v1",
    )

    assert result.selection.used_degraded_fallback is True
    assert result.plan is not None
    assert result.plan.fallback is None

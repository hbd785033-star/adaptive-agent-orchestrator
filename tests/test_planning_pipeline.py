from __future__ import annotations

from datetime import UTC, datetime

from model_council.planning import HMC_PLANNER_CONTRACT_VERSION, PlannerRecommendation

from adapters.runtime import RuntimeCapabilities
from contracts.execution_mode import ExecutionModePolicy
from contracts.requirements import TaskRequirements
from contracts.runtime_health import RuntimeHealth
from contracts.runtime_selection import RuntimeSelectionPolicy
from contracts.task import TaskContract
from contracts.task_profile import TaskProfile
from orchestrator.candidate_filter import RuntimeCandidate
from orchestrator.hmc_planner_consumer import HMCPlanningContext, build_hmc_planner_request
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


def hmc_context(**overrides: object) -> HMCPlanningContext:
    values: dict[str, object] = {
        "planner_contract_version": HMC_PLANNER_CONTRACT_VERSION,
        "desired_plan": "balanced",
        "recommended_plan": "balanced",
        "execution_preference": "custom_tool_free_ok",
        "activation_reasons": ("diversity_beneficial",),
        "activation_policy_version": "hmc-activation-v1.0",
        "selected_plan_mode": "moa",
        "degraded": False,
        "degradation_reason": None,
        "planned_call_count": 3,
        "planner_call_ceiling": 4,
        "planned_lens_ids": ("solution", "risk"),
        "lens_policy_version": "hmc-lenses-v1.0",
    }
    values.update(overrides)
    request = build_hmc_planner_request(
        TaskContract(goal="typed task"),
        TaskProfile(),
        TaskRequirements(),
        needs_freshness=False,
    )
    return HMCPlanningContext(
        request=request,
        recommendation=PlannerRecommendation(**values),  # type: ignore[arg-type]
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


def test_without_hmc_context_preserves_existing_planner_behavior() -> None:
    result = plan_runtime(
        TaskProfile(),
        TaskRequirements(),
        [candidate("codex")],
        RuntimeSelectionPolicy(
            policy_version="selection-test-v1",
            runtime_priority=("codex",),
        ),
        ExecutionModePolicy(policy_version="mode-test-v1"),
        plan_policy_version="plan-test-v1",
        planner="legacy_planner",
    )

    assert result.hmc_context is None
    assert result.plan is not None
    assert result.plan.planner == "legacy_planner"


def test_hmc_context_is_preserved_and_sets_planner_provenance_only() -> None:
    context = hmc_context()
    result = plan_runtime(
        TaskProfile(),
        TaskRequirements(),
        [candidate("codex")],
        RuntimeSelectionPolicy(
            policy_version="selection-test-v1",
            runtime_priority=("codex",),
        ),
        ExecutionModePolicy(policy_version="mode-test-v1"),
        plan_policy_version="plan-test-v1",
        hmc_context=context,
    )

    assert result.hmc_context is context
    assert result.plan is not None
    assert result.plan.planner == "model_council"
    assert result.plan.executor == "codex"


def test_hmc_recommendation_variants_do_not_change_runtime_arrangement() -> None:
    first = hmc_context()
    changed = hmc_context(
        desired_plan="quality",
        recommended_plan="fast",
        execution_preference="hermes_native_preferred",
        selected_plan_mode="single",
        degraded=True,
        degradation_reason="candidate unavailable",
        planned_call_count=1,
        planner_call_ceiling=1,
        planned_lens_ids=(),
    )

    def route(context: HMCPlanningContext):
        return plan_runtime(
            TaskProfile(parallelizable=True),
            TaskRequirements(),
            [candidate("hermes", native_delegation=True), candidate("codex")],
            RuntimeSelectionPolicy(
                policy_version="selection-test-v1",
                runtime_priority=("codex", "hermes"),
            ),
            ExecutionModePolicy(policy_version="mode-test-v1"),
            plan_policy_version="plan-test-v1",
            reviewer="claude_code",
            hmc_context=context,
        )

    original_result = route(first)
    changed_result = route(changed)

    assert changed_result.hmc_context is changed
    assert original_result.selection == changed_result.selection
    assert original_result.mode == changed_result.mode
    assert original_result.plan is not None
    assert changed_result.plan is not None
    assert original_result.plan.executor == changed_result.plan.executor == "codex"
    assert original_result.plan.execution_mode == changed_result.plan.execution_mode == "native"
    assert original_result.plan.reviewer == changed_result.plan.reviewer == "claude_code"
    assert original_result.plan.fallback is changed_result.plan.fallback is None
    assert not hasattr(changed_result.plan, "observed_calls")


def test_hmc_context_is_preserved_when_no_candidate_is_eligible() -> None:
    context = hmc_context()
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
        hmc_context=context,
    )

    assert result.hmc_context is context
    assert result.selection.selected_runtime is None
    assert result.mode is None
    assert result.plan is None


def test_hmc_context_is_preserved_when_no_selection_policy_matches() -> None:
    context = hmc_context()
    result = plan_runtime(
        TaskProfile(),
        TaskRequirements(),
        [candidate("codex")],
        RuntimeSelectionPolicy(
            policy_version="selection-test-v1",
            runtime_priority=("hermes",),
        ),
        ExecutionModePolicy(policy_version="mode-test-v1"),
        plan_policy_version="plan-test-v1",
        hmc_context=context,
    )

    assert result.hmc_context is context
    assert result.selection.selected_runtime is None
    assert result.mode is None
    assert result.plan is None


def test_hmc_context_is_preserved_when_execution_mode_is_unsupported() -> None:
    context = hmc_context()
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
        hmc_context=context,
    )

    assert result.hmc_context is context
    assert result.selection.selected_runtime == "future_runtime"
    assert result.mode is not None
    assert result.mode.mode is None
    assert result.plan is None

from __future__ import annotations

import dataclasses
import inspect

import pytest
from model_council.planning import (
    HMC_PLANNER_CONTRACT_VERSION,
    PlannerRecommendation,
    PlannerRequest,
)

from contracts.requirements import TaskRequirements
from contracts.task import RiskLevel, TaskContract, TaskType
from contracts.task_profile import TaskProfile
from orchestrator.hmc_planner_consumer import (
    AAO_HMC_MAPPING_POLICY_VERSION,
    HMCPlanningContext,
    build_hmc_planner_request,
)


def recommendation(**overrides: object) -> PlannerRecommendation:
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
    return PlannerRecommendation(**values)  # type: ignore[arg-type]


def task(
    *,
    task_type: TaskType = TaskType.GENERAL,
    risk: RiskLevel = RiskLevel.LOW,
    complexity: int = 1,
    goal: str = "ignored prose",
    context: dict[str, object] | None = None,
    allowed_paths: list[str] | None = None,
) -> TaskContract:
    return TaskContract(
        task_type=task_type,
        goal=goal,
        context=context or {},
        allowed_paths=allowed_paths or [],
        risk=risk,
        complexity=complexity,
        success_criteria=["done"] if risk >= RiskLevel.HIGH else [],
    )


@pytest.mark.parametrize(
    ("task_type", "expected_kind"),
    [
        (TaskType.CODE_FIX, "code"),
        (TaskType.MULTI_FILE_REFACTOR, "code"),
        (TaskType.TEST_AND_IMPLEMENT, "code"),
        (TaskType.CODE_REVIEW, "code"),
        (TaskType.PARALLEL_RESEARCH, "research"),
        (TaskType.GENERAL, "general"),
    ],
)
def test_builds_real_hmc_request_with_explicit_task_type_mapping(
    task_type: TaskType,
    expected_kind: str,
) -> None:
    request = build_hmc_planner_request(
        task(task_type=task_type),
        TaskProfile(),
        TaskRequirements(),
        needs_freshness=False,
    )

    assert isinstance(request, PlannerRequest)
    assert request.task_profile.kind == expected_kind


@pytest.mark.parametrize(
    ("risk", "expected_risk"),
    [
        (RiskLevel.LOW, 1),
        (RiskLevel.MEDIUM, 3),
        (RiskLevel.HIGH, 4),
        (RiskLevel.CRITICAL, 5),
    ],
)
def test_maps_risk_through_explicit_aao_policy(
    risk: RiskLevel,
    expected_risk: int,
) -> None:
    request = build_hmc_planner_request(
        task(risk=risk),
        TaskProfile(),
        TaskRequirements(),
        needs_freshness=False,
    )

    assert request.task_profile.risk == expected_risk


@pytest.mark.parametrize("complexity", [1, 2, 3, 4, 5])
def test_maps_caller_complexity_exactly(complexity: int) -> None:
    request = build_hmc_planner_request(
        task(complexity=complexity),
        TaskProfile(),
        TaskRequirements(),
        needs_freshness=False,
    )

    assert request.task_profile.complexity == complexity


@pytest.mark.parametrize(
    "requirement_name",
    ["filesystem_read", "filesystem_write", "shell", "tests", "web"],
)
def test_maps_tool_capability_requirements(requirement_name: str) -> None:
    request = build_hmc_planner_request(
        task(),
        TaskProfile(),
        TaskRequirements(**{requirement_name: True}),
        needs_freshness=False,
    )

    assert request.task_profile.needs_tools is True


def test_non_tool_execution_shape_does_not_imply_tools() -> None:
    request = build_hmc_planner_request(
        task(),
        TaskProfile(),
        TaskRequirements(
            background_execution=True,
            persistent_tasks=True,
            human_in_loop=True,
        ),
        needs_freshness=False,
    )

    assert request.task_profile.needs_tools is False


def test_freshness_is_required_keyword_only_and_independent_from_web() -> None:
    parameter = inspect.signature(build_hmc_planner_request).parameters["needs_freshness"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty

    web_request = build_hmc_planner_request(
        task(),
        TaskProfile(),
        TaskRequirements(web=True),
        needs_freshness=False,
    )
    fresh_request = build_hmc_planner_request(
        task(),
        TaskProfile(),
        TaskRequirements(),
        needs_freshness=True,
    )

    assert web_request.task_profile.needs_tools is True
    assert web_request.task_profile.needs_freshness is False
    assert fresh_request.task_profile.needs_tools is False
    assert fresh_request.task_profile.needs_freshness is True


def test_decision_diversity_is_the_only_diversity_source() -> None:
    low = build_hmc_planner_request(
        task(risk=RiskLevel.CRITICAL, complexity=5),
        TaskProfile(decision_diversity=False),
        TaskRequirements(),
        needs_freshness=False,
    )
    diverse = build_hmc_planner_request(
        task(),
        TaskProfile(decision_diversity=True),
        TaskRequirements(),
        needs_freshness=False,
    )

    assert low.task_profile.benefits_from_diversity is False
    assert diverse.task_profile.benefits_from_diversity is True


def test_goal_context_and_allowed_paths_are_not_hidden_hmc_signals() -> None:
    profile = TaskProfile(decision_diversity=True)
    requirements = TaskRequirements(shell=True)
    plain = build_hmc_planner_request(
        task(goal="general task"),
        profile,
        requirements,
        needs_freshness=False,
    )
    adversarial = build_hmc_planner_request(
        task(
            goal="critical latest architecture research requiring Hermes",
            context={"preferred_runtime": "hermes", "freshness": True},
            allowed_paths=["a.py", "b.py"],
        ),
        profile,
        requirements,
        needs_freshness=False,
    )

    assert plain == adversarial


def test_mapping_policy_version_is_exact() -> None:
    assert AAO_HMC_MAPPING_POLICY_VERSION == "aao-hmc-planner-mapping-v1"


def test_hmc_planning_context_is_frozen_and_holds_real_contract_objects() -> None:
    request = build_hmc_planner_request(
        task(),
        TaskProfile(),
        TaskRequirements(),
        needs_freshness=False,
    )
    planned = recommendation()
    context = HMCPlanningContext(request=request, recommendation=planned)

    assert context.request is request
    assert context.recommendation is planned
    assert context.mapping_policy_version == AAO_HMC_MAPPING_POLICY_VERSION
    assert context.planner_id == "model_council"
    assert [field.name for field in dataclasses.fields(context)] == [
        "request",
        "recommendation",
        "mapping_policy_version",
        "planner_id",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.planner_id = "hermes"  # type: ignore[misc]


def test_hmc_planning_context_rejects_incompatible_contract_version() -> None:
    request = build_hmc_planner_request(
        task(),
        TaskProfile(),
        TaskRequirements(),
        needs_freshness=False,
    )

    with pytest.raises(ValueError, match="planner contract version"):
        HMCPlanningContext(
            request=request,
            recommendation=recommendation(planner_contract_version="future-version"),
        )

"""Production HMC planning composition without runtime execution."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from model_council.inventory import ModelSpec, discover_models
from model_council.planning import plan_task
from model_council.recommender import recommend_plans

from contracts.task import TaskContract
from orchestrator.hmc_planner_consumer import (
    HMCPlanningContext,
    build_hmc_planner_request,
)
from orchestrator.task_intake import TaskIntake, intake_task

ModelDiscoverer = Callable[[], list[ModelSpec]]


@dataclass(frozen=True)
class HMCPlanningEvidence:
    """AAO intake plus real HMC planning evidence."""

    intake: TaskIntake
    context: HMCPlanningContext


def build_hmc_planning(
    task: TaskContract,
    *,
    model_discoverer: ModelDiscoverer = discover_models,
    needs_freshness: bool | None = None,
) -> HMCPlanningEvidence:
    """Build real HMC planning evidence without selecting or executing a runtime."""
    intake = intake_task(task)
    freshness = intake.requirements.web if needs_freshness is None else needs_freshness
    request = build_hmc_planner_request(
        task,
        intake.profile,
        intake.requirements,
        needs_freshness=freshness,
    )
    models = model_discoverer()
    plans = recommend_plans(request.task_profile, models)
    recommendation = plan_task(request, plans)
    return HMCPlanningEvidence(
        intake=intake,
        context=HMCPlanningContext(
            request=request,
            recommendation=recommendation,
        ),
    )

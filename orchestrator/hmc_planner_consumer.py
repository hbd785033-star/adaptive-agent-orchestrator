"""AAO-owned typed mapping to the HMC planner contract."""

from __future__ import annotations

from dataclasses import dataclass

from model_council.analysis import TaskProfile as HMCTaskProfile
from model_council.planning import (
    HMC_PLANNER_CONTRACT_VERSION,
    PlannerRecommendation,
    PlannerRequest,
)

from contracts.requirements import TaskRequirements
from contracts.task import RiskLevel, TaskContract, TaskType
from contracts.task_profile import TaskProfile as AAOTaskProfile

AAO_HMC_MAPPING_POLICY_VERSION = "aao-hmc-planner-mapping-v1"

_TASK_KIND_BY_TYPE = {
    TaskType.CODE_FIX: "code",
    TaskType.MULTI_FILE_REFACTOR: "code",
    TaskType.TEST_AND_IMPLEMENT: "code",
    TaskType.CODE_REVIEW: "code",
    TaskType.PARALLEL_RESEARCH: "research",
    TaskType.GENERAL: "general",
}

_HMC_RISK_BY_AAO_RISK = {
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 3,
    RiskLevel.HIGH: 4,
    RiskLevel.CRITICAL: 5,
}

_HMC_COMPLEXITY_BY_AAO_COMPLEXITY = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}


@dataclass(frozen=True)
class HMCPlanningContext:
    """Real HMC planning evidence plus AAO-owned mapping provenance."""

    request: PlannerRequest
    recommendation: PlannerRecommendation
    mapping_policy_version: str = AAO_HMC_MAPPING_POLICY_VERSION
    planner_id: str = "model_council"

    def __post_init__(self) -> None:
        if not isinstance(self.request, PlannerRequest):
            raise ValueError("request must be a real HMC PlannerRequest")
        if not isinstance(self.recommendation, PlannerRecommendation):
            raise ValueError("recommendation must be a real HMC PlannerRecommendation")
        if (
            self.request.contract_version != HMC_PLANNER_CONTRACT_VERSION
            or self.recommendation.planner_contract_version
            != HMC_PLANNER_CONTRACT_VERSION
        ):
            raise ValueError("planner contract version must match hmc-planner-v1.0")
        if self.mapping_policy_version != AAO_HMC_MAPPING_POLICY_VERSION:
            raise ValueError("mapping policy version must match AAO policy")
        if self.planner_id != "model_council":
            raise ValueError("planner_id must be model_council")


def build_hmc_planner_request(
    task: TaskContract,
    task_profile: AAOTaskProfile,
    requirements: TaskRequirements,
    *,
    needs_freshness: bool,
) -> PlannerRequest:
    """Map explicit AAO facts to the real typed HMC request contract."""
    if type(needs_freshness) is not bool:
        raise ValueError("needs_freshness must be an explicit boolean")

    needs_tools = any(
        (
            requirements.filesystem_read,
            requirements.filesystem_write,
            requirements.shell,
            requirements.tests,
            requirements.web,
        )
    )
    return PlannerRequest(
        task_profile=HMCTaskProfile(
            kind=_TASK_KIND_BY_TYPE[task.task_type],
            complexity=_HMC_COMPLEXITY_BY_AAO_COMPLEXITY[task.complexity],
            risk=_HMC_RISK_BY_AAO_RISK[task.risk],
            needs_tools=needs_tools,
            needs_freshness=needs_freshness,
            benefits_from_diversity=task_profile.decision_diversity,
        )
    )

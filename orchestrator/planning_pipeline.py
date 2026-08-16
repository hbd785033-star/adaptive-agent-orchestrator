"""Pure cross-runtime planning pipeline.

Recovery provenance
-------------------
This Phase 3A file is a source-constrained reconstruction from the preserved
Architecture Pivot checkpoint. It is not claimed byte-for-byte identical to
the lost worktree.

The pipeline is intentionally pure:
TaskProfile + TaskRequirements + explicit runtime candidates + explicit policies
-> assessments -> selection -> exact selected candidate -> execution mode -> RuntimePlan.

No runtime execution, probing, filesystem/network I/O, HMC invocation, AE
invocation, workspace mutation, or fallback execution occurs here.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from contracts.execution_mode import ExecutionModeDecision, ExecutionModePolicy
from contracts.requirements import TaskRequirements
from contracts.runtime_plan import RuntimePlan
from contracts.runtime_selection import RuntimeSelectionDecision, RuntimeSelectionPolicy
from contracts.task_profile import TaskProfile
from orchestrator.candidate_filter import (
    CandidateAssessment,
    RuntimeCandidate,
    assess_candidates,
)
from orchestrator.execution_mode_policy import select_execution_mode
from orchestrator.hmc_planner_consumer import HMCPlanningContext
from orchestrator.runtime_plan_composer import compose_runtime_plan
from orchestrator.runtime_selector import select_runtime


class PlanningResult(BaseModel):
    """Explainable planning output; absence remains absence rather than fake success."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessments: list[CandidateAssessment]
    selection: RuntimeSelectionDecision
    mode: ExecutionModeDecision | None = None
    plan: RuntimePlan | None = None
    hmc_context: HMCPlanningContext | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> PlanningResult:
        if self.selection.selected_runtime is None:
            if self.mode is not None or self.plan is not None:
                raise ValueError("no runtime selection requires mode=None and plan=None")
            return self

        if self.mode is None:
            if self.plan is not None:
                raise ValueError("plan requires an execution-mode decision")
            return self

        if self.mode.runtime != self.selection.selected_runtime:
            raise ValueError("mode runtime must match selected runtime")

        if self.mode.mode is None:
            if self.plan is not None:
                raise ValueError("unsupported/no execution mode must not create RuntimePlan")
            return self

        if self.plan is None:
            raise ValueError("selected executable mode requires RuntimePlan")
        if self.plan.executor != self.selection.selected_runtime:
            raise ValueError("RuntimePlan executor must match selected runtime")
        if self.plan.execution_mode != self.mode.mode:
            raise ValueError("RuntimePlan mode must match execution-mode decision")
        if (
            self.hmc_context is not None
            and self.plan.planner != self.hmc_context.planner_id
        ):
            raise ValueError("RuntimePlan planner must match HMC planning provenance")
        return self


def plan_runtime(
    task_profile: TaskProfile,
    requirements: TaskRequirements,
    candidates: list[RuntimeCandidate],
    selection_policy: RuntimeSelectionPolicy,
    execution_mode_policy: ExecutionModePolicy,
    *,
    plan_policy_version: str,
    planner: str | None = None,
    reviewer: str | None = None,
    approval_required: bool = False,
    hmc_context: HMCPlanningContext | None = None,
) -> PlanningResult:
    """Run the deterministic planning chain without executing any runtime."""
    assessments = assess_candidates(requirements, candidates)
    selection = select_runtime(assessments, selection_policy)

    if selection.selected_runtime is None:
        return PlanningResult(
            assessments=assessments,
            selection=selection,
            mode=None,
            plan=None,
            hmc_context=hmc_context,
        )

    matches = [
        candidate
        for candidate in candidates
        if candidate.runtime == selection.selected_runtime
    ]
    if len(matches) != 1:
        # select_runtime already rejects duplicate assessment identities; keep this
        # exact-resolution gate so the pipeline never fabricates capabilities.
        raise ValueError("selected runtime must resolve to exactly one supplied candidate")

    selected_candidate = matches[0]
    mode = select_execution_mode(
        task_profile,
        selection,
        selected_candidate.capabilities,
        execution_mode_policy,
    )

    if mode.mode is None:
        return PlanningResult(
            assessments=assessments,
            selection=selection,
            mode=mode,
            plan=None,
            hmc_context=hmc_context,
        )

    plan = compose_runtime_plan(
        selection,
        mode,
        plan_policy_version=plan_policy_version,
        planner=hmc_context.planner_id if hmc_context is not None else planner,
        reviewer=reviewer,
        approval_required=approval_required,
    )
    return PlanningResult(
        assessments=assessments,
        selection=selection,
        mode=mode,
        plan=plan,
        hmc_context=hmc_context,
    )

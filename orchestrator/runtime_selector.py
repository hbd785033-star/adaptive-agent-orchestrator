"""Deterministic runtime selection from candidate assessments."""
from __future__ import annotations

from contracts.runtime_selection import (
    RuntimeSelectionDecision,
    RuntimeSelectionDecisionCode,
    RuntimeSelectionPolicy,
)
from orchestrator.candidate_filter import CandidateAssessment


def select_runtime(
    assessments: list[CandidateAssessment],
    policy: RuntimeSelectionPolicy,
) -> RuntimeSelectionDecision:
    """Select a runtime by policy order from healthy eligible candidates only."""
    candidate_runtimes = [assessment.runtime for assessment in assessments]
    if len(candidate_runtimes) != len(set(candidate_runtimes)):
        raise ValueError("duplicate candidate runtime")

    healthy_by_runtime = {
        assessment.runtime: assessment
        for assessment in assessments
        if assessment.eligible and not assessment.degraded
    }
    for runtime in policy.runtime_priority:
        if runtime in healthy_by_runtime:
            return RuntimeSelectionDecision(
                selected_runtime=runtime,
                policy_version=policy.policy_version,
                decision_code=RuntimeSelectionDecisionCode.SELECTED_HEALTHY,
                used_degraded_fallback=False,
                reasons=[f"selected highest-priority healthy eligible runtime: {runtime}"],
            )

    degraded_by_runtime = {
        assessment.runtime: assessment
        for assessment in assessments
        if assessment.eligible and assessment.degraded
    }
    if degraded_by_runtime and not policy.allow_degraded_fallback:
        return RuntimeSelectionDecision(
            selected_runtime=None,
            policy_version=policy.policy_version,
            decision_code=RuntimeSelectionDecisionCode.DEGRADED_FALLBACK_DISABLED,
            used_degraded_fallback=False,
            reasons=["no healthy eligible candidate remained", "degraded fallback is disabled"],
        )
    if policy.allow_degraded_fallback:
        for runtime in policy.runtime_priority:
            if runtime in degraded_by_runtime:
                return RuntimeSelectionDecision(
                    selected_runtime=runtime,
                    policy_version=policy.policy_version,
                    decision_code=RuntimeSelectionDecisionCode.SELECTED_DEGRADED_FALLBACK,
                    used_degraded_fallback=True,
                    reasons=[
                        "no healthy eligible candidate remained",
                        f"selected highest-priority degraded eligible runtime as fallback: {runtime}",
                    ],
                )

    if healthy_by_runtime or degraded_by_runtime:
        return RuntimeSelectionDecision(
            selected_runtime=None,
            policy_version=policy.policy_version,
            decision_code=RuntimeSelectionDecisionCode.NO_POLICY_MATCH,
            reasons=["eligible runtime candidates were not authorized by policy"],
        )

    return RuntimeSelectionDecision(
        selected_runtime=None,
        policy_version=policy.policy_version,
        decision_code=RuntimeSelectionDecisionCode.NO_ELIGIBLE_CANDIDATE,
        reasons=["no eligible runtime candidates were available"],
    )

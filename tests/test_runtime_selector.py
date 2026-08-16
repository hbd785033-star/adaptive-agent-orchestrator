"""Tests for deterministic runtime selection from candidate assessments."""
from __future__ import annotations

import pytest

from contracts.runtime_health import HealthStatus
from contracts.runtime_selection import RuntimeSelectionPolicy
from orchestrator.candidate_filter import CandidateAssessment
from orchestrator.runtime_selector import select_runtime


def assessment(runtime: str, *, eligible: bool = True, degraded: bool = False) -> CandidateAssessment:
    return CandidateAssessment(
        runtime=runtime,
        eligible=eligible,
        degraded=degraded,
        health_status=HealthStatus.DEGRADED if degraded else HealthStatus.AVAILABLE,
    )


def test_selects_highest_priority_healthy_candidate_not_input_order() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code", "hermes"),
    )
    assessments = [
        assessment("hermes"),
        assessment("claude_code"),
        assessment("codex"),
    ]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime == "codex"
    assert decision.policy_version == "runtime-selection-v1"
    assert decision.decision_code == "selected_healthy"
    assert decision.used_degraded_fallback is False
    assert decision.reasons == ["selected highest-priority healthy eligible runtime: codex"]


def test_ineligible_high_priority_candidate_is_never_revived() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code"),
    )
    assessments = [assessment("codex", eligible=False), assessment("claude_code")]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime == "claude_code"
    assert decision.decision_code == "selected_healthy"


def test_healthy_candidate_beats_higher_priority_degraded_candidate() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code"),
    )
    assessments = [assessment("codex", degraded=True), assessment("claude_code")]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime == "claude_code"
    assert decision.used_degraded_fallback is False
    assert decision.decision_code == "selected_healthy"


def test_selects_degraded_candidate_as_explicit_fallback() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code"),
        allow_degraded_fallback=True,
    )
    assessments = [assessment("claude_code", degraded=True), assessment("codex", degraded=True)]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime == "codex"
    assert decision.used_degraded_fallback is True
    assert decision.decision_code == "selected_degraded_fallback"
    assert decision.reasons == [
        "no healthy eligible candidate remained",
        "selected highest-priority degraded eligible runtime as fallback: codex",
    ]


def test_disabled_degraded_fallback_returns_no_selection() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code"),
        allow_degraded_fallback=False,
    )
    assessments = [assessment("codex", degraded=True)]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime is None
    assert decision.used_degraded_fallback is False
    assert decision.decision_code == "degraded_fallback_disabled"
    assert decision.reasons == ["no healthy eligible candidate remained", "degraded fallback is disabled"]


def test_all_ineligible_candidates_return_no_eligible_candidate() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code", "hermes"),
    )
    assessments = [
        assessment("codex", eligible=False),
        assessment("claude_code", eligible=False),
        assessment("hermes", eligible=False),
    ]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime is None
    assert decision.decision_code == "no_eligible_candidate"
    assert decision.reasons == ["no eligible runtime candidates were available"]


def test_empty_assessments_return_no_eligible_candidate() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code"),
    )

    decision = select_runtime([], policy)

    assert decision.selected_runtime is None
    assert decision.decision_code == "no_eligible_candidate"


def test_eligible_runtime_absent_from_policy_is_not_selected() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code"),
    )
    assessments = [assessment("new_runtime")]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime is None
    assert decision.decision_code == "no_policy_match"
    assert decision.reasons == ["eligible runtime candidates were not authorized by policy"]


def test_policy_entry_absent_from_candidate_set_is_not_fatal() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex", "claude_code", "hermes"),
    )
    assessments = [assessment("claude_code"), assessment("hermes")]

    decision = select_runtime(assessments, policy)

    assert decision.selected_runtime == "claude_code"
    assert decision.decision_code == "selected_healthy"


def test_duplicate_candidate_identity_fails_closed() -> None:
    policy = RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex",),
    )

    try:
        select_runtime([assessment("codex"), assessment("codex")], policy)
    except ValueError as exc:
        assert "duplicate candidate runtime" in str(exc)
    else:
        raise AssertionError("duplicate candidate identity should fail closed")


def test_duplicate_policy_identity_fails_closed() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RuntimeSelectionPolicy(
            policy_version="runtime-selection-v1",
            runtime_priority=("codex", "codex"),
        )


def test_policy_rejects_blank_runtime_identity_and_policy_version() -> None:
    from pydantic import ValidationError

    RuntimeSelectionPolicy(
        policy_version="runtime-selection-v1",
        runtime_priority=("codex",),
    )
    with pytest.raises(ValidationError):
        RuntimeSelectionPolicy(policy_version="", runtime_priority=("codex",))
    with pytest.raises(ValidationError):
        RuntimeSelectionPolicy(policy_version="runtime-selection-v1", runtime_priority=("",))
    with pytest.raises(ValidationError):
        RuntimeSelectionPolicy(policy_version="runtime-selection-v1", runtime_priority=("   ",))

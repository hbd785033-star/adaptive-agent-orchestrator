"""Tests for deterministic cross-runtime candidate assessment."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from adapters.runtime import RuntimeCapabilities
from contracts.requirements import TaskRequirements
from contracts.runtime_health import RuntimeHealth
from orchestrator.candidate_filter import RuntimeCandidate, assess_candidate, assess_candidates

CHECKED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def test_available_candidate_with_required_capabilities_is_eligible() -> None:
    requirements = TaskRequirements(filesystem_write=True, tests=True)
    candidate = RuntimeCandidate(
        runtime="codex",
        capabilities=RuntimeCapabilities(filesystem_write=True, tests=True),
        health=RuntimeHealth(runtime="codex", status="available", checked_at=CHECKED_AT),
    )

    assessment = assess_candidate(requirements, candidate)

    assert assessment.runtime == "codex"
    assert assessment.eligible is True
    assert assessment.degraded is False
    assert assessment.health_status == "available"
    assert assessment.missing_capabilities == []
    assert assessment.rejection_codes == []


def test_missing_capabilities_are_all_reported_in_requirement_order() -> None:
    requirements = TaskRequirements(filesystem_write=True, shell=True, tests=True)
    candidate = RuntimeCandidate(
        runtime="codex",
        capabilities=RuntimeCapabilities(),
        health=RuntimeHealth(runtime="codex", status="available", checked_at=CHECKED_AT),
    )

    assessment = assess_candidate(requirements, candidate)

    assert assessment.eligible is False
    assert assessment.missing_capabilities == ["filesystem_write", "shell", "tests"]
    assert assessment.rejection_codes == ["missing_capability"]
    assert assessment.reasons == [
        "missing required capability: filesystem_write",
        "missing required capability: shell",
        "missing required capability: tests",
    ]


def test_false_requirement_does_not_reject_when_capability_is_false() -> None:
    requirements = TaskRequirements(web=False)
    candidate = RuntimeCandidate(
        runtime="codex",
        capabilities=RuntimeCapabilities(web=False),
        health=RuntimeHealth(runtime="codex", status="available", checked_at=CHECKED_AT),
    )

    assessment = assess_candidate(requirements, candidate)

    assert assessment.eligible is True
    assert assessment.missing_capabilities == []
    assert assessment.rejection_codes == []


def test_unknown_health_fails_closed_even_when_capabilities_match() -> None:
    requirements = TaskRequirements(filesystem_write=True)
    candidate = RuntimeCandidate(
        runtime="codex",
        capabilities=RuntimeCapabilities(filesystem_write=True),
        health=RuntimeHealth(runtime="codex", status="unknown"),
    )

    assessment = assess_candidate(requirements, candidate)

    assert assessment.eligible is False
    assert assessment.degraded is False
    assert assessment.health_status == "unknown"
    assert assessment.missing_capabilities == []
    assert assessment.rejection_codes == ["health_unknown"]
    assert assessment.reasons == ["runtime health is unknown"]


def test_unavailable_health_fails_closed_and_preserves_reasons() -> None:
    requirements = TaskRequirements(filesystem_write=True)
    candidate = RuntimeCandidate(
        runtime="codex",
        capabilities=RuntimeCapabilities(filesystem_write=True),
        health=RuntimeHealth(
            runtime="codex",
            status="unavailable",
            checked_at=CHECKED_AT,
            reasons=["CLI executable missing"],
        ),
    )

    assessment = assess_candidate(requirements, candidate)

    assert assessment.eligible is False
    assert assessment.degraded is False
    assert assessment.health_status == "unavailable"
    assert assessment.rejection_codes == ["health_unavailable"]
    assert assessment.reasons == ["CLI executable missing"]


def test_degraded_health_remains_eligible_and_visible_when_capabilities_match() -> None:
    requirements = TaskRequirements(filesystem_write=True)
    candidate = RuntimeCandidate(
        runtime="claude_code",
        capabilities=RuntimeCapabilities(filesystem_write=True),
        health=RuntimeHealth(
            runtime="claude_code",
            status="degraded",
            checked_at=CHECKED_AT,
            reasons=["provider rate limited"],
        ),
    )

    assessment = assess_candidate(requirements, candidate)

    assert assessment.eligible is True
    assert assessment.degraded is True
    assert assessment.health_status == "degraded"
    assert assessment.rejection_codes == []
    assert assessment.reasons == ["provider rate limited"]


def test_health_and_capability_rejections_both_remain_visible() -> None:
    requirements = TaskRequirements(filesystem_write=True)
    candidate = RuntimeCandidate(
        runtime="codex",
        capabilities=RuntimeCapabilities(filesystem_write=False),
        health=RuntimeHealth(
            runtime="codex",
            status="unavailable",
            checked_at=CHECKED_AT,
            reasons=["CLI executable missing"],
        ),
    )

    assessment = assess_candidate(requirements, candidate)

    assert assessment.eligible is False
    assert assessment.missing_capabilities == ["filesystem_write"]
    assert assessment.rejection_codes == ["missing_capability", "health_unavailable"]
    assert assessment.reasons == [
        "missing required capability: filesystem_write",
        "CLI executable missing",
    ]


def test_task_requirements_fields_have_runtime_capability_schema_parity() -> None:
    assert set(TaskRequirements.model_fields) <= set(RuntimeCapabilities.__dataclass_fields__)


def test_runtime_candidate_rejects_health_identity_mismatch() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RuntimeCandidate(
            runtime="codex",
            capabilities=RuntimeCapabilities(),
            health=RuntimeHealth(
                runtime="claude_code",
                status="unavailable",
                checked_at=CHECKED_AT,
            ),
        )


def test_no_requirements_with_available_health_is_eligible() -> None:
    candidate = RuntimeCandidate(
        runtime="hermes",
        capabilities=RuntimeCapabilities(),
        health=RuntimeHealth(runtime="hermes", status="available", checked_at=CHECKED_AT),
    )

    assessment = assess_candidate(TaskRequirements(), candidate)

    assert assessment.eligible is True
    assert assessment.missing_capabilities == []
    assert assessment.rejection_codes == []


def test_batch_assessment_preserves_input_order_without_selection() -> None:
    requirements = TaskRequirements()
    candidates = [
        RuntimeCandidate(
            runtime="hermes",
            capabilities=RuntimeCapabilities(),
            health=RuntimeHealth(
                runtime="hermes", status="unavailable", checked_at=CHECKED_AT
            ),
        ),
        RuntimeCandidate(
            runtime="codex",
            capabilities=RuntimeCapabilities(),
            health=RuntimeHealth(runtime="codex", status="available", checked_at=CHECKED_AT),
        ),
        RuntimeCandidate(
            runtime="claude_code",
            capabilities=RuntimeCapabilities(),
            health=RuntimeHealth(
                runtime="claude_code", status="degraded", checked_at=CHECKED_AT
            ),
        ),
    ]

    assessments = assess_candidates(requirements, candidates)

    assert [assessment.runtime for assessment in assessments] == [
        "hermes",
        "codex",
        "claude_code",
    ]
    assert [assessment.eligible for assessment in assessments] == [False, True, True]

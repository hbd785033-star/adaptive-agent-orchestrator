"""Deterministic runtime candidate capability and health assessment."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adapters.runtime import RuntimeCapabilities
from contracts.requirements import TaskRequirements
from contracts.runtime_health import HealthStatus, RuntimeHealth


class RejectionCode(StrEnum):
    MISSING_CAPABILITY = "missing_capability"
    HEALTH_UNKNOWN = "health_unknown"
    HEALTH_UNAVAILABLE = "health_unavailable"
    CONTRACT_MISMATCH = "contract_mismatch"


class RuntimeCandidate(BaseModel):
    """Explicitly supplied runtime data for one assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: str
    capabilities: RuntimeCapabilities
    health: RuntimeHealth

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> RuntimeCandidate:
        if not self.runtime.strip():
            raise ValueError("runtime must not be blank")
        if self.health.runtime != self.runtime:
            raise ValueError("candidate runtime must match health runtime")
        return self


class CandidateAssessment(BaseModel):
    """Explainable eligibility result without selection or ranking semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: str
    eligible: bool
    degraded: bool
    health_status: HealthStatus
    missing_capabilities: list[str] = Field(default_factory=list)
    rejection_codes: list[RejectionCode] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def assess_candidate(
    requirements: TaskRequirements,
    candidate: RuntimeCandidate,
) -> CandidateAssessment:
    """Assess one explicitly supplied candidate without side effects or ranking."""
    missing_capabilities = [
        name
        for name, required in requirements.model_dump().items()
        if required and not getattr(candidate.capabilities, name)
    ]
    rejection_codes: list[RejectionCode] = []
    reasons: list[str] = []

    if missing_capabilities:
        rejection_codes.append(RejectionCode.MISSING_CAPABILITY)
        reasons.extend(f"missing required capability: {name}" for name in missing_capabilities)

    health_status = candidate.health.status
    if health_status == HealthStatus.UNKNOWN:
        rejection_codes.append(RejectionCode.HEALTH_UNKNOWN)
        reasons.append("runtime health is unknown")
    elif health_status == HealthStatus.UNAVAILABLE:
        rejection_codes.append(RejectionCode.HEALTH_UNAVAILABLE)
        reasons.extend(
            candidate.health.reasons or ["runtime is unavailable"]
        )
    elif health_status == HealthStatus.DEGRADED:
        reasons.extend(candidate.health.reasons)

    eligible = not missing_capabilities and health_status in {
        HealthStatus.AVAILABLE,
        HealthStatus.DEGRADED,
    }
    return CandidateAssessment(
        runtime=candidate.runtime,
        eligible=eligible,
        degraded=health_status == HealthStatus.DEGRADED,
        health_status=health_status,
        missing_capabilities=missing_capabilities,
        rejection_codes=rejection_codes,
        reasons=reasons,
    )


def assess_candidates(
    requirements: TaskRequirements,
    candidates: list[RuntimeCandidate],
) -> list[CandidateAssessment]:
    """Assess candidates in caller-provided order; never sort or select."""
    return [assess_candidate(requirements, candidate) for candidate in candidates]

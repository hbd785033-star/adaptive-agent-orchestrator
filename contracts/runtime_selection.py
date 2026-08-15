"""Runtime selection policy and decision contracts."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RuntimeSelectionDecisionCode(StrEnum):
    SELECTED_HEALTHY = "selected_healthy"
    SELECTED_DEGRADED_FALLBACK = "selected_degraded_fallback"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    NO_POLICY_MATCH = "no_policy_match"
    DEGRADED_FALLBACK_DISABLED = "degraded_fallback_disabled"


class RuntimeSelectionPolicy(BaseModel):
    """Versioned deterministic executor preference order supplied by the caller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    runtime_priority: tuple[str, ...]
    allow_degraded_fallback: bool = True

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy_version must not be blank")
        return value

    @model_validator(mode="after")
    def validate_runtime_priority(self) -> RuntimeSelectionPolicy:
        seen: set[str] = set()
        for runtime in self.runtime_priority:
            if not runtime.strip():
                raise ValueError("runtime_priority entries must not be blank")
            if runtime in seen:
                raise ValueError("runtime_priority entries must be unique")
            seen.add(runtime)
        return self


class RuntimeSelectionDecision(BaseModel):
    """Selection outcome without execution mode or RuntimePlan semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_runtime: str | None
    policy_version: str
    decision_code: RuntimeSelectionDecisionCode
    used_degraded_fallback: bool = False
    reasons: list[str] = Field(default_factory=list)

    @field_validator("selected_runtime")
    @classmethod
    def validate_selected_runtime(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("selected_runtime must not be blank")
        return value

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> RuntimeSelectionDecision:
        selected_codes = {
            RuntimeSelectionDecisionCode.SELECTED_HEALTHY,
            RuntimeSelectionDecisionCode.SELECTED_DEGRADED_FALLBACK,
        }
        empty_codes = {
            RuntimeSelectionDecisionCode.NO_ELIGIBLE_CANDIDATE,
            RuntimeSelectionDecisionCode.NO_POLICY_MATCH,
            RuntimeSelectionDecisionCode.DEGRADED_FALLBACK_DISABLED,
        }
        if self.decision_code in selected_codes and self.selected_runtime is None:
            raise ValueError("selected decision requires selected_runtime")
        if self.decision_code in empty_codes and self.selected_runtime is not None:
            raise ValueError("non-selected decision requires selected_runtime=None")
        expected_degraded = (
            self.decision_code == RuntimeSelectionDecisionCode.SELECTED_DEGRADED_FALLBACK
        )
        if self.used_degraded_fallback != expected_degraded:
            raise ValueError("used_degraded_fallback contradicts decision_code")
        return self

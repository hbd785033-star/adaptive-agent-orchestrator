"""Execution mode policy and decision contracts."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExecutionMode(StrEnum):
    NATIVE = "native"
    DIRECT = "direct"
    DELEGATE = "delegate"
    KANBAN = "kanban"


class ExecutionModeDecisionCode(StrEnum):
    SELECTED_NATIVE = "selected_native"
    SELECTED_DIRECT = "selected_direct"
    SELECTED_DELEGATE = "selected_delegate"
    SELECTED_KANBAN = "selected_kanban"
    NO_SELECTED_RUNTIME = "no_selected_runtime"
    UNSUPPORTED_MODE = "unsupported_mode"


class ExecutionModePolicy(BaseModel):
    """Versioned deterministic policy for selected-runtime execution mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str
    allow_delegate: bool = True
    allow_kanban: bool = True

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy_version must not be blank")
        return value


class ExecutionModeDecision(BaseModel):
    """Execution mode decision without RuntimePlan or execution semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: str | None
    mode: ExecutionMode | None
    policy_version: str
    decision_code: ExecutionModeDecisionCode
    reasons: list[str] = Field(default_factory=list)

    @field_validator("runtime")
    @classmethod
    def validate_runtime(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("runtime must not be blank")
        return value

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> ExecutionModeDecision:
        selected_modes = {
            ExecutionModeDecisionCode.SELECTED_NATIVE: ExecutionMode.NATIVE,
            ExecutionModeDecisionCode.SELECTED_DIRECT: ExecutionMode.DIRECT,
            ExecutionModeDecisionCode.SELECTED_DELEGATE: ExecutionMode.DELEGATE,
            ExecutionModeDecisionCode.SELECTED_KANBAN: ExecutionMode.KANBAN,
        }
        if self.decision_code in selected_modes:
            if self.runtime is None:
                raise ValueError("selected execution mode requires runtime")
            if self.mode != selected_modes[self.decision_code]:
                raise ValueError("execution mode contradicts decision_code")
        elif self.decision_code == ExecutionModeDecisionCode.NO_SELECTED_RUNTIME:
            if self.runtime is not None or self.mode is not None:
                raise ValueError("no_selected_runtime requires runtime=None and mode=None")
        elif self.decision_code == ExecutionModeDecisionCode.UNSUPPORTED_MODE:
            if self.runtime is None or self.mode is not None:
                raise ValueError("unsupported_mode requires runtime and mode=None")
        return self

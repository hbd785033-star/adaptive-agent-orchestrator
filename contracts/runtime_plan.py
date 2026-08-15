"""Cross-runtime planning and execution arrangement contract."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExecutionMode(StrEnum):
    NATIVE = "native"
    DIRECT = "direct"
    DELEGATE = "delegate"
    KANBAN = "kanban"


class RuntimePlan(BaseModel):
    """
    AAO's selected planning, execution, and review arrangement for one task.

    This is not the legacy delegation ExecutionPlan. It records which runtime
    owns execution orchestration, so AAO does not also fan out a task that a
    native runtime or Hermes Kanban already owns internally.
    """

    model_config = ConfigDict(extra="forbid")

    planner: str | None = None
    executor: str
    execution_mode: ExecutionMode
    reviewer: str | None = None
    fallback: str | None = None

    policy_version: str = "runtime-plan-v1.0"
    selection_policy_version: str | None = None
    execution_mode_policy_version: str | None = None
    reasons: list[str] = Field(default_factory=list)
    approval_required: bool = False

    @field_validator("executor", "policy_version")
    @classmethod
    def validate_required_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity field must not be blank")
        return value

    @field_validator("planner", "reviewer", "fallback", "selection_policy_version", "execution_mode_policy_version")
    @classmethod
    def validate_optional_identity(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("identity field must not be blank")
        return value

    @model_validator(mode="after")
    def validate_known_v1_arrangements(self) -> RuntimePlan:
        """
        Fail closed for V1 runtime identities with known execution modes.

        V1 compatibility guard only. Runtime/mode compatibility belongs to
        RuntimeCapabilities plus capability/policy filtering. Do not expand this
        validator as new runtimes or modes are added.
        """
        if self.reviewer is not None and self.reviewer == self.executor:
            raise ValueError("reviewer must be independent from executor")

        if self.executor == "hermes":
            allowed = {ExecutionMode.DIRECT, ExecutionMode.DELEGATE, ExecutionMode.KANBAN}
        elif self.executor in {"codex", "claude_code"}:
            allowed = {ExecutionMode.NATIVE}
        else:
            return self

        if self.execution_mode not in allowed:
            raise ValueError(
                f"executor {self.executor!r} does not support execution_mode "
                f"{self.execution_mode!r} in runtime-plan-v1.0"
            )
        return self

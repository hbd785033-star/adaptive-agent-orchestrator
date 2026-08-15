"""Runtime operational health contract."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HealthStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class RuntimeHealth(BaseModel):
    """
    Current operational availability for a runtime.

    Capability support belongs in RuntimeCapabilities; this model only reports
    whether a runtime is currently usable, degraded, unavailable, or unknown.
    """

    model_config = ConfigDict(extra="forbid")

    runtime: str
    status: HealthStatus = HealthStatus.UNKNOWN
    reasons: list[str] = Field(default_factory=list)
    checked_at: datetime | None = None

    @field_validator("runtime")
    @classmethod
    def validate_runtime_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("runtime must not be blank")
        return value

    @model_validator(mode="after")
    def validate_observed_status_has_timestamp(self) -> RuntimeHealth:
        if self.status != HealthStatus.UNKNOWN and self.checked_at is None:
            raise ValueError("observed runtime health requires checked_at")
        return self

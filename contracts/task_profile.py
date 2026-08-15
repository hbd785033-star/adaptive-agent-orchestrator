"""Deterministic task-shape analysis for cross-runtime routing."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ComplexityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskProfile(BaseModel):
    """
    Router-facing description of what shape a task has.

    This is intentionally separate from runtime capability requirements and
    does not duplicate TaskContract caller intent such as risk or scope.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning_complexity: ComplexityLevel = ComplexityLevel.LOW
    execution_complexity: ComplexityLevel = ComplexityLevel.LOW

    persistent_execution: bool = False
    long_running: bool = False
    parallelizable: bool = False
    cross_role_dependencies: bool = False
    human_in_loop: bool = False
    decision_diversity: bool = False
    external_effects: bool = False

"""Task Contract — the single source of truth passed to every agent."""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator
import uuid


class TaskType(str, Enum):
    CODE_FIX = "code_fix"
    MULTI_FILE_REFACTOR = "multi_file_refactor"
    PARALLEL_RESEARCH = "parallel_research"
    TEST_AND_IMPLEMENT = "test_and_implement"
    CODE_REVIEW = "code_review"
    GENERAL = "general"


class RiskLevel(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class WorkspaceSpec(BaseModel):
    """Injected by WorkspaceManager before execution starts."""
    repo_path: str
    worktree_path: str | None = None   # None → read-only / single-agent tasks
    branch: str | None = None
    child_id: str | None = None


class TaskContract(BaseModel):
    """
    Immutable specification for one unit of work.

    The orchestrator creates this; all downstream components (router, budget,
    workspace manager, adapter) read from it but never mutate it.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_type: TaskType = TaskType.GENERAL
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)

    # --- scope constraints (encoded into agent prompt + verified post-exec) ---
    allowed_paths: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)

    # --- output expectations ---
    output_schema: list[str] = Field(
        default_factory=lambda: ["summary", "files_changed", "tests_run", "unresolved_risks"]
    )

    # --- risk / priority ---
    risk: RiskLevel = RiskLevel.LOW
    complexity: int = Field(default=1, ge=1, le=5)

    # --- workspace (filled by WorkspaceManager, not the caller) ---
    workspace: WorkspaceSpec | None = None

    # --- metadata ---
    parent_task_id: str | None = None

    @model_validator(mode="after")
    def validate_high_risk_has_criteria(self) -> "TaskContract":
        if self.risk >= RiskLevel.HIGH and not self.success_criteria:
            raise ValueError("High-risk tasks must define success_criteria")
        return self

    def prompt_preamble(self) -> str:
        """
        Build the constraint block injected into the agent prompt.
        This is the Prompt Guard (v1 enforcement mechanism).
        """
        lines = [
            f"# Task: {self.goal}",
            "",
            "## Constraints (MANDATORY — do not deviate)",
        ]
        if self.allowed_paths:
            lines.append(f"- Only modify files under: {', '.join(self.allowed_paths)}")
        if self.forbidden_actions:
            lines.append("- Forbidden actions:")
            for a in self.forbidden_actions:
                lines.append(f"  - {a}")
        if self.success_criteria:
            lines.append("- Success criteria:")
            for c in self.success_criteria:
                lines.append(f"  - {c}")
        if self.workspace and self.workspace.worktree_path:
            lines.append(f"- Working directory: {self.workspace.worktree_path}")
            lines.append(f"- Branch: {self.workspace.branch}")
        lines.append("")
        lines.append("## Additional context")
        for k, v in self.context.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

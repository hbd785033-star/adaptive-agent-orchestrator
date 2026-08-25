"""Typed AAO experiment/profile/budget evidence transported in ExecutionRecord 0.1 metadata."""
from __future__ import annotations

import hashlib
import json
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

if TYPE_CHECKING:
    from contracts.task import TaskContract
    from orchestrator.budget import BudgetConfig, BudgetState

CONTRACT_VERSION = "1.0"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Id = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonKind(StrEnum):
    RUNTIME = "runtime_comparison"
    HARNESS = "harness_comparison"
    MODEL = "model_comparison"
    PROVIDER = "provider_comparison"
    BUDGET_POLICY = "budget_policy_comparison"


class ArmId(StrEnum):
    A = "A"
    B = "B"


class ProfileCompleteness(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_sha256(projection: Any) -> str:
    """Hash an explicit semantic projection using deterministic canonical JSON."""
    encoded = json.dumps(
        _jsonable(projection),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class ExperimentBindingV1(_FrozenModel):
    experiment_id: NonEmptyStr
    experiment_definition_revision: NonEmptyStr
    experiment_definition_sha256: Sha256Id
    comparison_kind: ComparisonKind
    pair_id: NonEmptyStr
    trial_id: int = Field(gt=0)
    arm_id: ArmId


class TaskIdentityV1(_FrozenModel):
    task_definition_id: NonEmptyStr
    task_definition_revision: NonEmptyStr
    task_contract_sha256: Sha256Id
    prompt_sha256: Sha256Id
    success_criteria_sha256: Sha256Id
    dataset_or_fixture_revision: NonEmptyStr


class WorkspaceContractV1(_FrozenModel):
    starting_revision: NonEmptyStr
    isolation_mode: NonEmptyStr
    fixture_revision: NonEmptyStr


class ConfiguredBudgetV1(_FrozenModel):
    max_children: int = Field(ge=0)
    max_depth: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    max_total_calls: int = Field(ge=0)
    require_approval_above_calls: int = Field(ge=0)
    budget_id: Sha256Id

    def semantic_projection(self) -> dict[str, int]:
        return {
            "max_children": self.max_children,
            "max_depth": self.max_depth,
            "max_retries": self.max_retries,
            "max_total_calls": self.max_total_calls,
            "require_approval_above_calls": self.require_approval_above_calls,
        }

    @model_validator(mode="after")
    def validate_identity(self) -> ConfiguredBudgetV1:
        if self.budget_id != canonical_sha256(self.semantic_projection()):
            raise ValueError("budget_id does not match configured budget semantic projection")
        return self


class EnforcedBudgetV1(_FrozenModel):
    calls_used: int = Field(ge=0)
    calls_reserved: int = Field(ge=0)
    retries_used: int = Field(ge=0)
    children_used: int = Field(ge=0)
    depth_used: int = Field(ge=0)
    submission_prevented: bool | None = None
    retry_prevented: bool | None = None
    approval_required: bool | None = None
    approval_granted: bool | None = None

    @classmethod
    def from_state(
        cls,
        state: BudgetState,
        *,
        submission_prevented: bool | None = None,
        retry_prevented: bool | None = None,
        approval_required: bool | None = None,
        approval_granted: bool | None = None,
    ) -> EnforcedBudgetV1:
        return cls(
            calls_used=state.calls_used,
            calls_reserved=state.calls_reserved,
            retries_used=state.retries_used,
            children_used=state.children_used,
            depth_used=state.depth_used,
            submission_prevented=submission_prevented,
            retry_prevented=retry_prevented,
            approval_required=approval_required,
            approval_granted=approval_granted,
        )


class ConfiguredProfileV1(_FrozenModel):
    runtime: NonEmptyStr | None = None
    harness: NonEmptyStr | None = None
    runtime_version: NonEmptyStr | None = None
    model: NonEmptyStr | None = None
    provider: NonEmptyStr | None = None
    execution_mode: NonEmptyStr | None = None
    tools_config_sha256: Sha256Id | None = None
    policy_config_sha256: Sha256Id | None = None
    reasoning_config_sha256: Sha256Id | None = None
    environment_config_sha256: Sha256Id | None = None
    workspace_contract: WorkspaceContractV1 | None = None
    network_policy_identity: NonEmptyStr | None = None
    sandbox_policy_identity: NonEmptyStr | None = None
    approval_policy_identity: NonEmptyStr | None = None
    budget_id: Sha256Id | None = None


class ObservedProfileV1(_FrozenModel):
    runtime: NonEmptyStr | None = None
    runtime_version: NonEmptyStr | None = None
    model: NonEmptyStr | None = None
    provider: NonEmptyStr | None = None
    effective_workspace_revision: NonEmptyStr | None = None
    effective_workspace_root: NonEmptyStr | None = None
    observed_isolation_level: NonEmptyStr | None = None
    observed_network_mode: NonEmptyStr | None = None
    observed_sandbox_mode: NonEmptyStr | None = None
    observed_approval_behavior: NonEmptyStr | None = None
    tool_evidence_completeness: Literal["complete", "partial", "unknown"] | None = None
    file_evidence_completeness: Literal["complete", "partial", "unknown"] | None = None


def configured_profile_semantic_projection(profile: ConfiguredProfileV1) -> dict[str, Any]:
    """Return only configured facts that materially define the profile."""
    return {
        "runtime": profile.runtime,
        "harness": profile.harness,
        "runtime_version": profile.runtime_version,
        "model": profile.model,
        "provider": profile.provider,
        "execution_mode": profile.execution_mode,
        "tools_config_sha256": profile.tools_config_sha256,
        "policy_config_sha256": profile.policy_config_sha256,
        "reasoning_config_sha256": profile.reasoning_config_sha256,
        "environment_config_sha256": profile.environment_config_sha256,
        "workspace_contract": profile.workspace_contract,
        "network_policy_identity": profile.network_policy_identity,
        "sandbox_policy_identity": profile.sandbox_policy_identity,
        "approval_policy_identity": profile.approval_policy_identity,
        "budget_id": profile.budget_id,
    }


def effective_profile_semantic_projection(profile: ObservedProfileV1) -> dict[str, Any]:
    """Return observed material facts, excluding machine-local workspace paths."""
    return {
        "runtime": profile.runtime,
        "runtime_version": profile.runtime_version,
        "model": profile.model,
        "provider": profile.provider,
        "effective_workspace_revision": profile.effective_workspace_revision,
        "observed_isolation_level": profile.observed_isolation_level,
        "observed_network_mode": profile.observed_network_mode,
        "observed_sandbox_mode": profile.observed_sandbox_mode,
        "observed_approval_behavior": profile.observed_approval_behavior,
        "tool_evidence_completeness": profile.tool_evidence_completeness,
        "file_evidence_completeness": profile.file_evidence_completeness,
    }


class ProfileIdentityV1(_FrozenModel):
    configured_profile_id: Sha256Id | None = None
    effective_profile_id: Sha256Id | None = None
    completeness: ProfileCompleteness
    incompleteness_reasons: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_completeness(self) -> ProfileIdentityV1:
        if self.completeness == ProfileCompleteness.COMPLETE:
            if self.configured_profile_id is None or self.effective_profile_id is None:
                raise ValueError("complete profile requires configured and effective profile identities")
            if self.incompleteness_reasons:
                raise ValueError("complete profile cannot have incompleteness reasons")
        elif not self.incompleteness_reasons:
            raise ValueError("incomplete profile requires reasons")
        return self


class ObservedUsageV1(_FrozenModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    tool_calls: list[dict[str, Any]] | None = None
    files_changed: list[str] | None = None
    latency_seconds: float | None = Field(default=None, ge=0)


class EstimatedCostV1(_FrozenModel):
    amount_usd: float | None = Field(default=None, ge=0)
    provenance: NonEmptyStr | None = None
    price_model_identity: NonEmptyStr | None = None


class BilledCostV1(_FrozenModel):
    amount_usd: float | None = Field(default=None, ge=0)
    provenance: NonEmptyStr | None = None


class BudgetEvidenceV1(_FrozenModel):
    configured_budget: ConfiguredBudgetV1
    enforced_budget: EnforcedBudgetV1
    observed_usage: ObservedUsageV1 | None = None
    estimated_cost: EstimatedCostV1
    billed_cost: BilledCostV1


class ExecutionLifecycleV1(_FrozenModel):
    submission_attempted: bool
    runtime_adapter_invoked: bool
    runtime_run_id: NonEmptyStr | None = None
    terminal_status: Literal["completed", "failed", "cancelled", "timeout"] | None = None
    failure_phase: NonEmptyStr | None = None
    failure_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ExecutionLifecycleV1:
        if not self.submission_attempted:
            if self.runtime_adapter_invoked or self.runtime_run_id is not None:
                raise ValueError("pre-submission failure cannot claim runtime invocation or run ID")
            if self.terminal_status is not None:
                raise ValueError("pre-submission failure cannot claim a runtime terminal status")
            if self.failure_phase is None or self.failure_reason is None:
                raise ValueError("pre-submission failure requires phase and reason")
        if not self.runtime_adapter_invoked and self.runtime_run_id is not None:
            raise ValueError("runtime_run_id requires runtime adapter invocation")
        return self


class AAOExperimentV1(_FrozenModel):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    experiment: ExperimentBindingV1
    task: TaskIdentityV1
    configured_profile: ConfiguredProfileV1
    observed_profile: ObservedProfileV1
    profile_identity: ProfileIdentityV1
    budget: BudgetEvidenceV1
    execution_lifecycle: ExecutionLifecycleV1

    @model_validator(mode="after")
    def validate_cross_field_identities(self) -> AAOExperimentV1:
        if self.configured_profile.budget_id != self.budget.configured_budget.budget_id:
            raise ValueError("configured profile budget_id must match configured budget")
        run_id = self.execution_lifecycle.runtime_run_id
        if run_id is not None and run_id in {
            self.experiment.experiment_id,
            self.experiment.pair_id,
        }:
            raise ValueError("runtime_run_id must remain separate from experiment and pair IDs")
        expected = _profile_identity(
            self.experiment.comparison_kind,
            self.configured_profile,
            self.observed_profile,
        )
        if self.profile_identity != expected:
            raise ValueError("profile_identity does not match canonical profile projections")
        return self


def configured_budget_from_config(config: BudgetConfig) -> ConfiguredBudgetV1:
    projection = {
        "max_children": config.max_children,
        "max_depth": config.max_depth,
        "max_retries": config.max_retries,
        "max_total_calls": config.max_total_calls,
        "require_approval_above_calls": config.require_approval_above_calls,
    }
    return ConfiguredBudgetV1(**projection, budget_id=canonical_sha256(projection))


def _criterion_projection(criterion: Any) -> Any:
    return _jsonable(criterion)


def build_task_identity_v1(
    task: TaskContract,
    *,
    task_definition_id: str,
    task_definition_revision: str,
    prompt: str,
    dataset_or_fixture_revision: str,
    starting_revision: str,
) -> TaskIdentityV1:
    """Bind task equality to explicit semantics, never TaskContract.id or workspace paths."""
    criteria = [_criterion_projection(item) for item in task.success_criteria]
    subtasks = [
        {
            "id": item.id,
            "goal": item.goal,
            "allowed_paths": list(item.allowed_paths),
            "dependencies": list(item.dependencies),
            "expected_output": item.expected_output,
        }
        for item in task.subtasks
    ]
    projection = {
        "task_definition_id": task_definition_id,
        "task_definition_revision": task_definition_revision,
        "goal": task.goal,
        "prompt": prompt,
        "task_type": task.task_type.value,
        "allowed_paths": list(task.allowed_paths),
        "forbidden_actions": list(task.forbidden_actions),
        "success_criteria": criteria,
        "output_schema": list(task.output_schema),
        "risk": int(task.risk),
        "complexity": task.complexity,
        "subtasks": subtasks,
        "starting_revision": starting_revision,
        "dataset_or_fixture_revision": dataset_or_fixture_revision,
    }
    return TaskIdentityV1(
        task_definition_id=task_definition_id,
        task_definition_revision=task_definition_revision,
        task_contract_sha256=canonical_sha256(projection),
        prompt_sha256=canonical_sha256({"prompt": prompt}),
        success_criteria_sha256=canonical_sha256({"success_criteria": criteria}),
        dataset_or_fixture_revision=dataset_or_fixture_revision,
    )


def _missing_fields(model: BaseModel, prefix: str, fields: tuple[str, ...]) -> list[str]:
    return [f"{prefix}.{field}" for field in fields if getattr(model, field) is None]


def _profile_identity(
    comparison_kind: ComparisonKind,
    configured: ConfiguredProfileV1,
    observed: ObservedProfileV1,
) -> ProfileIdentityV1:
    configured_required = (
        "runtime",
        "harness",
        "execution_mode",
        "tools_config_sha256",
        "policy_config_sha256",
        "environment_config_sha256",
        "workspace_contract",
        "approval_policy_identity",
        "budget_id",
    )
    observed_required = (
        "runtime",
        "effective_workspace_revision",
        "observed_isolation_level",
        "tool_evidence_completeness",
        "file_evidence_completeness",
    )
    configured_extra: tuple[str, ...] = ()
    observed_extra: tuple[str, ...] = ()
    if comparison_kind == ComparisonKind.HARNESS:
        configured_extra = ("model", "provider", "reasoning_config_sha256")
        observed_extra = ("model", "provider")
    elif comparison_kind in {ComparisonKind.MODEL, ComparisonKind.PROVIDER}:
        configured_extra = ("model", "provider")
        observed_extra = ("model", "provider")
    elif comparison_kind == ComparisonKind.BUDGET_POLICY:
        configured_extra = ("model", "provider", "reasoning_config_sha256")
        observed_extra = ("model", "provider")

    reasons = _missing_fields(
        configured, "configured_profile", configured_required + configured_extra
    )
    reasons.extend(_missing_fields(observed, "observed_profile", observed_required + observed_extra))

    configured_id = None
    if not _missing_fields(configured, "configured_profile", configured_required + configured_extra):
        configured_id = canonical_sha256(configured_profile_semantic_projection(configured))

    effective_id = None
    if not _missing_fields(observed, "observed_profile", observed_required + observed_extra):
        effective_id = canonical_sha256(effective_profile_semantic_projection(observed))

    if reasons:
        return ProfileIdentityV1(
            configured_profile_id=configured_id,
            effective_profile_id=effective_id,
            completeness=ProfileCompleteness.INCOMPLETE,
            incompleteness_reasons=reasons,
        )
    return ProfileIdentityV1(
        configured_profile_id=configured_id,
        effective_profile_id=effective_id,
        completeness=ProfileCompleteness.COMPLETE,
        incompleteness_reasons=[],
    )


def build_aao_experiment_v1(
    *,
    binding: ExperimentBindingV1,
    task: TaskIdentityV1,
    configured_profile: ConfiguredProfileV1,
    observed_profile: ObservedProfileV1,
    configured_budget: ConfiguredBudgetV1,
    enforced_budget: EnforcedBudgetV1,
    observed_usage: ObservedUsageV1 | None,
    estimated_cost: EstimatedCostV1,
    billed_cost: BilledCostV1,
    lifecycle: ExecutionLifecycleV1,
) -> AAOExperimentV1:
    return AAOExperimentV1(
        experiment=binding,
        task=task,
        configured_profile=configured_profile,
        observed_profile=observed_profile,
        profile_identity=_profile_identity(
            binding.comparison_kind,
            configured_profile,
            observed_profile,
        ),
        budget=BudgetEvidenceV1(
            configured_budget=configured_budget,
            enforced_budget=enforced_budget,
            observed_usage=observed_usage,
            estimated_cost=estimated_cost,
            billed_cost=billed_cost,
        ),
        execution_lifecycle=lifecycle,
    )

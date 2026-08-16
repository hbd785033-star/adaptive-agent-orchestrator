"""Cross-runtime control-plane domain contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from contracts.task_profile import ComplexityLevel, TaskProfile


def test_task_profile_reasoning_and_execution_complexity_are_independent() -> None:
    profile = TaskProfile(
        reasoning_complexity=ComplexityLevel.HIGH,
        execution_complexity=ComplexityLevel.LOW,
    )

    assert profile.reasoning_complexity == ComplexityLevel.HIGH
    assert profile.execution_complexity == ComplexityLevel.LOW


def test_task_requirements_are_vendor_neutral_and_do_not_duplicate_risk() -> None:
    from contracts.requirements import TaskRequirements

    requirements = TaskRequirements(filesystem_write=True, shell=True)

    assert requirements.filesystem_write is True
    assert not {"codex", "claude", "hermes", "provider", "model"} & set(
        type(requirements).model_fields
    )
    assert "risk" not in type(requirements).model_fields
    assert "high_risk" not in type(requirements).model_fields

    with pytest.raises(ValidationError):
        TaskRequirements(codex=True)
    with pytest.raises(ValidationError):
        TaskRequirements(risk="high")


def test_requirement_capability_direct_match_names_are_canonical() -> None:
    from adapters.runtime import RuntimeCapabilities
    from contracts.requirements import TaskRequirements

    requirement_fields = set(TaskRequirements.model_fields)
    capability_fields = set(RuntimeCapabilities.__dataclass_fields__)
    direct_match_fields = {
        "filesystem_read",
        "filesystem_write",
        "shell",
        "tests",
        "web",
        "background_execution",
        "persistent_tasks",
        "human_in_loop",
    }

    assert direct_match_fields <= requirement_fields
    assert direct_match_fields <= capability_fields
    assert "persistent_task_support" not in requirement_fields
    assert "human_in_loop_support" not in requirement_fields

    with pytest.raises(ValidationError):
        TaskRequirements(persistent_task_support=True)
    with pytest.raises(ValidationError):
        TaskRequirements(human_in_loop_support=True)


def test_new_contracts_leave_risk_source_on_task_contract() -> None:
    from contracts.requirements import TaskRequirements
    from contracts.runtime_health import RuntimeHealth
    from contracts.runtime_plan import RuntimePlan
    from contracts.task import RiskLevel, TaskContract

    task = TaskContract(goal="review", risk=RiskLevel.MEDIUM)
    contract_fields = {
        TaskProfile: set(TaskProfile.model_fields),
        TaskRequirements: set(TaskRequirements.model_fields),
        RuntimeHealth: set(RuntimeHealth.model_fields),
        RuntimePlan: set(RuntimePlan.model_fields),
    }

    assert task.risk == RiskLevel.MEDIUM
    for fields in contract_fields.values():
        assert "risk" not in fields
        assert "high_risk" not in fields


def test_runtime_plan_does_not_duplicate_governance_or_evaluation_planes() -> None:
    from contracts.runtime_plan import RuntimePlan

    plan = RuntimePlan(planner="model_council", executor="hermes", execution_mode="direct")
    fields = set(type(plan).model_fields)

    assert plan.planner == "model_council"
    assert not {"budget", "allowed_paths", "forbidden_actions", "evaluator"} & fields


def test_runtime_health_accepts_only_operational_statuses() -> None:
    from contracts.runtime_health import RuntimeHealth

    checked_at = datetime.now(UTC)
    for status in ("available", "degraded", "unavailable", "unknown"):
        health = RuntimeHealth(runtime="example", status=status, checked_at=checked_at)
        assert health.status == status

    with pytest.raises(ValidationError):
        RuntimeHealth(runtime="example", status="supported")


def test_runtime_health_defaults_to_unknown_without_probe() -> None:
    from contracts.runtime_health import HealthStatus, RuntimeHealth

    health = RuntimeHealth(runtime="example")

    assert health.status == HealthStatus.UNKNOWN
    assert health.checked_at is None


def test_runtime_health_requires_timestamp_for_observed_statuses() -> None:
    from contracts.runtime_health import RuntimeHealth

    checked_at = datetime.now(UTC)
    for status in ("available", "degraded", "unavailable"):
        assert RuntimeHealth(runtime="example", status=status, checked_at=checked_at)
        with pytest.raises(ValidationError):
            RuntimeHealth(runtime="example", status=status)

    assert RuntimeHealth(runtime="example", status="unknown", checked_at=None)


def test_runtime_health_runtime_identity_is_not_blank() -> None:
    from contracts.runtime_health import RuntimeHealth

    with pytest.raises(ValidationError):
        RuntimeHealth(runtime="")
    with pytest.raises(ValidationError):
        RuntimeHealth(runtime="   ")


def test_new_domain_models_reject_unknown_fields() -> None:
    from contracts.requirements import TaskRequirements
    from contracts.runtime_health import RuntimeHealth
    from contracts.runtime_plan import RuntimePlan

    with pytest.raises(ValidationError):
        TaskProfile(filesystem_wirte=True)
    with pytest.raises(ValidationError):
        TaskRequirements(filesystem_wirte=True)
    with pytest.raises(ValidationError):
        RuntimeHealth(runtime="example", healthy_enough=True)
    with pytest.raises(ValidationError):
        RuntimePlan(executor="hermes", executon_mode="direct")


def test_runtime_capabilities_include_minimal_cross_runtime_filtering_fields() -> None:
    from adapters.runtime import RuntimeCapabilities

    capabilities = RuntimeCapabilities()

    expected_fields = {
        "filesystem_read",
        "filesystem_write",
        "shell",
        "tests",
        "web",
        "background_execution",
        "persistent_tasks",
        "human_in_loop",
        "native_kanban",
        "structured_output",
        "usage_observable",
        "cost_observable",
    }

    assert expected_fields <= set(capabilities.__dataclass_fields__)


def test_new_routing_capabilities_default_false_without_changing_legacy_defaults() -> None:
    from adapters.runtime import RuntimeCapabilities

    caps = RuntimeCapabilities()
    new_capability_fields = {
        "filesystem_read",
        "filesystem_write",
        "shell",
        "tests",
        "web",
        "background_execution",
        "persistent_tasks",
        "human_in_loop",
        "native_kanban",
        "structured_output",
        "usage_observable",
        "cost_observable",
    }

    assert {field: getattr(caps, field) for field in new_capability_fields} == {
        field: False for field in new_capability_fields
    }
    assert caps.streaming_events is True
    assert caps.mid_run_steer is False
    assert caps.native_delegation is False
    assert caps.cancellation is True
    assert caps.session_resume is False
    assert caps.max_concurrent_runs == 8


def test_runtime_plan_runtime_identity_remains_extensible() -> None:
    from contracts.runtime_plan import RuntimePlan

    # Runtime identity is open; capability and policy filtering establish support.
    plan = RuntimePlan(executor="future_runtime", execution_mode="native")

    assert plan.executor == "future_runtime"


def test_runtime_plan_accepts_v1_runtime_execution_arrangements() -> None:
    from contracts.runtime_plan import RuntimePlan

    valid_pairs = [
        ("hermes", "direct"),
        ("hermes", "delegate"),
        ("hermes", "kanban"),
        ("codex", "native"),
        ("claude_code", "native"),
    ]

    for executor, mode in valid_pairs:
        plan = RuntimePlan(executor=executor, execution_mode=mode)
        assert plan.executor == executor
        assert plan.execution_mode == mode


def test_runtime_plan_rejects_known_invalid_v1_arrangements() -> None:
    from contracts.runtime_plan import RuntimePlan

    with pytest.raises(ValidationError):
        RuntimePlan(executor="codex", execution_mode="kanban")

    with pytest.raises(ValidationError):
        RuntimePlan(executor="claude_code", execution_mode="delegate")


def test_runtime_plan_identity_fields_are_not_blank() -> None:
    from contracts.runtime_plan import RuntimePlan

    with pytest.raises(ValidationError):
        RuntimePlan(executor="", execution_mode="direct")
    with pytest.raises(ValidationError):
        RuntimePlan(executor="   ", execution_mode="direct")
    with pytest.raises(ValidationError):
        RuntimePlan(planner="   ", executor="hermes", execution_mode="direct")
    with pytest.raises(ValidationError):
        RuntimePlan(executor="hermes", execution_mode="direct", reviewer="")
    with pytest.raises(ValidationError):
        RuntimePlan(executor="hermes", execution_mode="direct", fallback="   ")


def test_runtime_plan_policy_version_is_not_blank() -> None:
    from contracts.runtime_plan import RuntimePlan

    plan = RuntimePlan(executor="hermes", execution_mode="direct", policy_version="runtime-plan-v1.0")

    assert plan.policy_version == "runtime-plan-v1.0"
    with pytest.raises(ValidationError):
        RuntimePlan(executor="hermes", execution_mode="direct", policy_version="")
    with pytest.raises(ValidationError):
        RuntimePlan(executor="hermes", execution_mode="direct", policy_version="   ")


def test_runtime_plan_has_one_internal_orchestration_owner() -> None:
    from contracts.runtime_plan import RuntimePlan

    plan = RuntimePlan(executor="hermes", execution_mode="kanban")

    assert "route" not in type(plan).model_fields
    assert "children" not in type(plan).model_fields
    with pytest.raises(ValidationError):
        RuntimePlan(executor="hermes", execution_mode="kanban", route="delegation")

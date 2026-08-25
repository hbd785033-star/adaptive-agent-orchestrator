"""Phase 4B experiment profile, task identity, and budget truth contracts."""
from __future__ import annotations

from copy import deepcopy

import pytest

from contracts.experiment_profile import (
    BilledCostV1,
    ConfiguredProfileV1,
    EnforcedBudgetV1,
    EstimatedCostV1,
    ExecutionLifecycleV1,
    ExperimentBindingV1,
    ObservedProfileV1,
    ObservedUsageV1,
    WorkspaceContractV1,
    build_aao_experiment_v1,
    build_task_identity_v1,
    canonical_sha256,
    configured_budget_from_config,
    configured_profile_semantic_projection,
    effective_profile_semantic_projection,
)
from contracts.task import TaskContract
from orchestrator.budget import BudgetConfig, BudgetState


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _binding(*, comparison_kind: str = "runtime_comparison", suffix: str = "1"):
    return ExperimentBindingV1(
        experiment_id=f"experiment-{suffix}",
        experiment_definition_revision="revision-1",
        experiment_definition_sha256=_sha("experiment-definition"),
        comparison_kind=comparison_kind,
        pair_id=f"pair-{suffix}",
        trial_id=1,
        arm_id="A",
    )


def _task_identity(*, starting_revision: str = "a" * 40):
    task = TaskContract(
        id="execution-local-id",
        goal="Implement the frozen contract",
        allowed_paths=["contracts/**"],
        forbidden_actions=["push"],
        success_criteria=["focused tests pass"],
    )
    return build_task_identity_v1(
        task,
        task_definition_id="task-definition-1",
        task_definition_revision="task-revision-1",
        prompt="Implement the frozen contract exactly.",
        dataset_or_fixture_revision="fixture-1",
        starting_revision=starting_revision,
    )


def _budget(*, max_retries: int = 1):
    config = BudgetConfig(
        max_children=2,
        max_depth=1,
        max_retries=max_retries,
        max_total_calls=8,
        require_approval_above_calls=5,
    )
    return configured_budget_from_config(config)


def _configured_profile(*, budget_id: str, **changes):
    values = {
        "runtime": "runtime-a",
        "harness": "adaptive-agent-orchestrator",
        "runtime_version": None,
        "model": None,
        "provider": None,
        "execution_mode": "direct",
        "tools_config_sha256": _sha("tools"),
        "policy_config_sha256": _sha("policy"),
        "reasoning_config_sha256": None,
        "environment_config_sha256": _sha("environment"),
        "workspace_contract": WorkspaceContractV1(
            starting_revision="a" * 40,
            isolation_mode="workspace",
            fixture_revision="fixture-1",
        ),
        "network_policy_identity": None,
        "sandbox_policy_identity": None,
        "approval_policy_identity": "approval-policy-v1",
        "budget_id": budget_id,
    }
    values.update(changes)
    return ConfiguredProfileV1(**values)


def _observed_profile(**changes):
    values = {
        "runtime": "runtime-a",
        "runtime_version": None,
        "model": None,
        "provider": None,
        "effective_workspace_revision": "a" * 40,
        "effective_workspace_root": "C:/Temp/aao-worktree-1",
        "observed_isolation_level": "workspace",
        "observed_network_mode": None,
        "observed_sandbox_mode": None,
        "observed_approval_behavior": None,
        "tool_evidence_completeness": "complete",
        "file_evidence_completeness": "complete",
    }
    values.update(changes)
    return ObservedProfileV1(**values)


def _contract(
    *,
    binding=None,
    task=None,
    configured_profile=None,
    observed_profile=None,
    configured_budget=None,
    enforced_budget=None,
    observed_usage=None,
    estimated_cost=None,
    billed_cost=None,
    lifecycle=None,
):
    configured_budget = configured_budget or _budget()
    configured_profile = configured_profile or _configured_profile(
        budget_id=configured_budget.budget_id
    )
    return build_aao_experiment_v1(
        binding=binding or _binding(),
        task=task or _task_identity(),
        configured_profile=configured_profile,
        observed_profile=observed_profile or _observed_profile(),
        configured_budget=configured_budget,
        enforced_budget=enforced_budget
        or EnforcedBudgetV1(
            calls_used=1,
            calls_reserved=0,
            retries_used=0,
            children_used=0,
            depth_used=0,
            submission_prevented=False,
            retry_prevented=False,
            approval_required=False,
            approval_granted=None,
        ),
        observed_usage=observed_usage,
        estimated_cost=estimated_cost or EstimatedCostV1(),
        billed_cost=billed_cost or BilledCostV1(),
        lifecycle=lifecycle
        or ExecutionLifecycleV1(
            submission_attempted=True,
            runtime_adapter_invoked=True,
            runtime_run_id="observed-run-1",
            terminal_status="completed",
            failure_phase=None,
            failure_reason=None,
        ),
    )


def test_incomplete_material_observed_profile_has_no_effective_identity():
    contract = _contract(
        binding=_binding(comparison_kind="harness_comparison"),
        configured_profile=_configured_profile(
            budget_id=_budget().budget_id,
            model="gpt-example",
            provider="provider-example",
            reasoning_config_sha256=_sha("reasoning"),
        ),
        observed_profile=_observed_profile(model=None, provider="provider-example"),
    )

    identity = contract.profile_identity
    assert identity.effective_profile_id is None
    assert identity.completeness == "incomplete"
    assert "observed_profile.model" in identity.incompleteness_reasons


def test_configured_identity_excludes_execution_ephemera():
    first = _contract()
    second = _contract(
        binding=_binding(suffix="2"),
        observed_profile=_observed_profile(
            effective_workspace_root="D:/Temp/different-worktree-nonce"
        ),
        lifecycle=ExecutionLifecycleV1(
            submission_attempted=True,
            runtime_adapter_invoked=True,
            runtime_run_id="different-run-and-session",
            terminal_status="completed",
            failure_phase=None,
            failure_reason=None,
        ),
    )

    assert first.profile_identity.configured_profile_id == (
        second.profile_identity.configured_profile_id
    )


def test_profile_hash_inputs_are_explicit_semantic_projections():
    configured = _configured_profile(budget_id=_budget().budget_id)
    observed = _observed_profile()

    assert set(configured_profile_semantic_projection(configured)) == {
        "runtime",
        "harness",
        "runtime_version",
        "model",
        "provider",
        "execution_mode",
        "tools_config_sha256",
        "policy_config_sha256",
        "reasoning_config_sha256",
        "environment_config_sha256",
        "workspace_contract",
        "network_policy_identity",
        "sandbox_policy_identity",
        "approval_policy_identity",
        "budget_id",
    }
    assert set(effective_profile_semantic_projection(observed)) == {
        "runtime",
        "runtime_version",
        "model",
        "provider",
        "effective_workspace_revision",
        "observed_isolation_level",
        "observed_network_mode",
        "observed_sandbox_mode",
        "observed_approval_behavior",
        "tool_evidence_completeness",
        "file_evidence_completeness",
    }
    assert "effective_workspace_root" not in effective_profile_semantic_projection(observed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "different-model"),
        ("provider", "different-provider"),
        ("policy_config_sha256", _sha("different-policy")),
    ],
)
def test_material_configured_profile_difference_changes_identity(field, value):
    baseline = _contract()
    changed_profile = _configured_profile(budget_id=_budget().budget_id, **{field: value})
    changed = _contract(configured_profile=changed_profile)

    assert baseline.profile_identity.configured_profile_id != (
        changed.profile_identity.configured_profile_id
    )


def test_retry_budget_changes_budget_and_configured_profile_identity():
    first_budget = _budget(max_retries=1)
    second_budget = _budget(max_retries=2)
    first = _contract(
        configured_budget=first_budget,
        configured_profile=_configured_profile(budget_id=first_budget.budget_id),
    )
    second = _contract(
        configured_budget=second_budget,
        configured_profile=_configured_profile(budget_id=second_budget.budget_id),
    )

    assert first_budget.budget_id != second_budget.budget_id
    assert first.profile_identity.configured_profile_id != (
        second.profile_identity.configured_profile_id
    )


def test_workspace_starting_revision_changes_task_fairness_identity():
    first = _task_identity(starting_revision="a" * 40)
    second = _task_identity(starting_revision="b" * 40)

    assert first.task_contract_sha256 != second.task_contract_sha256


def test_budget_state_exports_only_truthful_current_enforcement_snapshot():
    state = BudgetState(task_id="task", config=BudgetConfig())
    assert state.reserve_calls() is None
    state.commit_reserved_call()
    state.retries_used = 1
    state.children_used = 1
    state.depth_used = 1

    evidence = EnforcedBudgetV1.from_state(
        state,
        submission_prevented=False,
        retry_prevented=True,
        approval_required=True,
        approval_granted=None,
    )

    assert evidence.calls_used == 1
    assert evidence.calls_reserved == 0
    assert evidence.retries_used == 1
    assert evidence.children_used == 1
    assert evidence.depth_used == 1
    assert evidence.retry_prevented is True
    assert evidence.approval_granted is None


def test_pre_submission_failure_preserves_unknown_runtime_usage_and_cost():
    contract = _contract(
        observed_profile=ObservedProfileV1(),
        enforced_budget=EnforcedBudgetV1(
            calls_used=0,
            calls_reserved=0,
            retries_used=0,
            children_used=0,
            depth_used=0,
            submission_prevented=True,
            retry_prevented=False,
            approval_required=True,
            approval_granted=False,
        ),
        observed_usage=None,
        estimated_cost=EstimatedCostV1(amount_usd=None, provenance=None),
        billed_cost=BilledCostV1(amount_usd=None, provenance=None),
        lifecycle=ExecutionLifecycleV1(
            submission_attempted=False,
            runtime_adapter_invoked=False,
            runtime_run_id=None,
            terminal_status=None,
            failure_phase="pre_submission",
            failure_reason="approval denied",
        ),
    )

    payload = contract.model_dump(mode="json")
    assert payload["execution_lifecycle"]["runtime_run_id"] is None
    assert payload["budget"]["observed_usage"] is None
    assert payload["budget"]["estimated_cost"]["amount_usd"] is None
    assert payload["budget"]["billed_cost"]["amount_usd"] is None
    assert contract.profile_identity.effective_profile_id is None


def test_unknown_values_survive_copy_and_serialization_without_empty_defaults():
    payload = _contract(
        observed_profile=ObservedProfileV1(),
        observed_usage=ObservedUsageV1(
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            total_tokens=None,
            tool_calls=None,
            files_changed=None,
            latency_seconds=None,
        ),
    ).model_dump(mode="json")
    copied = deepcopy(payload)

    assert copied["observed_profile"]["model"] is None
    assert copied["observed_profile"]["provider"] is None
    assert copied["budget"]["observed_usage"]["input_tokens"] is None
    assert copied["budget"]["observed_usage"]["tool_calls"] is None
    assert copied["budget"]["observed_usage"]["files_changed"] is None

"""ExecutionRecord 0.1 transport tests for the frozen Phase 4B namespace."""
from __future__ import annotations

from aao_cli.main import _build_execution_record
from contracts.experiment_profile import (
    BilledCostV1,
    ConfiguredProfileV1,
    EnforcedBudgetV1,
    EstimatedCostV1,
    ExecutionLifecycleV1,
    ExperimentBindingV1,
    ObservedProfileV1,
    WorkspaceContractV1,
    build_aao_experiment_v1,
    build_task_identity_v1,
    canonical_sha256,
    configured_budget_from_config,
)
from contracts.task import TaskContract
from orchestrator.budget import BudgetConfig


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _experiment():
    budget = configured_budget_from_config(BudgetConfig())
    task = TaskContract(
        id="execution-provenance-only",
        goal="Produce a deterministic record",
        success_criteria=["record validates"],
    )
    return build_aao_experiment_v1(
        binding=ExperimentBindingV1(
            experiment_id="experiment-1",
            experiment_definition_revision="revision-1",
            experiment_definition_sha256=_sha("experiment"),
            comparison_kind="runtime_comparison",
            pair_id="pair-1",
            trial_id=1,
            arm_id="A",
        ),
        task=build_task_identity_v1(
            task,
            task_definition_id="task-definition-1",
            task_definition_revision="task-revision-1",
            prompt="Produce the deterministic record.",
            dataset_or_fixture_revision="fixture-1",
            starting_revision="a" * 40,
        ),
        configured_profile=ConfiguredProfileV1(
            runtime="runtime-a",
            harness="adaptive-agent-orchestrator",
            runtime_version=None,
            model=None,
            provider=None,
            execution_mode="direct",
            tools_config_sha256=_sha("tools"),
            policy_config_sha256=_sha("policy"),
            reasoning_config_sha256=None,
            environment_config_sha256=_sha("environment"),
            workspace_contract=WorkspaceContractV1(
                starting_revision="a" * 40,
                isolation_mode="workspace",
                fixture_revision="fixture-1",
            ),
            network_policy_identity=None,
            sandbox_policy_identity=None,
            approval_policy_identity="approval-policy-v1",
            budget_id=budget.budget_id,
        ),
        observed_profile=ObservedProfileV1(
            runtime="runtime-a",
            runtime_version=None,
            model=None,
            provider=None,
            effective_workspace_revision="a" * 40,
            effective_workspace_root="C:/Temp/provenance-only",
            observed_isolation_level="workspace",
            observed_network_mode=None,
            observed_sandbox_mode=None,
            observed_approval_behavior=None,
            tool_evidence_completeness="complete",
            file_evidence_completeness="complete",
        ),
        configured_budget=budget,
        enforced_budget=EnforcedBudgetV1(
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
        observed_usage=None,
        estimated_cost=EstimatedCostV1(amount_usd=None, provenance=None),
        billed_cost=BilledCostV1(amount_usd=None, provenance=None),
        lifecycle=ExecutionLifecycleV1(
            submission_attempted=True,
            runtime_adapter_invoked=True,
            runtime_run_id="runtime-run-1",
            terminal_status="completed",
            failure_phase=None,
            failure_reason=None,
        ),
    )


def test_execution_record_01_exports_typed_experiment_namespace():
    experiment = _experiment()
    record = _build_execution_record(
        task_id="execution-provenance-only",
        result={
            "outcome": "completed",
            "run_id": "runtime-run-1",
            "observed": {
                "runtime_adapter": "runtime-a",
                "runtime_adapter_invoked": True,
                "output": None,
            },
        },
        mock=False,
        started_at="2026-08-25T00:00:00Z",
        finished_at="2026-08-25T00:00:01Z",
        experiment=experiment,
    )

    payload = record.model_dump(mode="json")
    namespace = payload["metadata"]["aao_experiment_v1"]
    assert payload["schema_version"] == "0.1"
    assert namespace["contract_version"] == "1.0"
    assert namespace == experiment.model_dump(mode="json")
    assert namespace["experiment"]["experiment_id"] != payload["run_id"]
    assert namespace["experiment"]["pair_id"] != payload["run_id"]


def test_execution_record_experiment_transport_preserves_unknowns():
    record = _build_execution_record(
        task_id="execution-provenance-only",
        result={"outcome": "completed", "run_id": "runtime-run-1"},
        mock=False,
        started_at="2026-08-25T00:00:00Z",
        finished_at="2026-08-25T00:00:01Z",
        experiment=_experiment(),
    )

    namespace = record.metadata["aao_experiment_v1"]
    assert namespace["configured_profile"]["model"] is None
    assert namespace["configured_profile"]["provider"] is None
    assert namespace["budget"]["observed_usage"] is None
    assert namespace["budget"]["estimated_cost"]["amount_usd"] is None
    assert namespace["budget"]["billed_cost"]["amount_usd"] is None


def test_execution_record_without_experiment_remains_backward_compatible():
    record = _build_execution_record(
        task_id="legacy-task",
        result={"outcome": "failed"},
        mock=False,
        started_at="2026-08-25T00:00:00Z",
        finished_at="2026-08-25T00:00:01Z",
    )

    assert record.schema_version == "0.1"
    assert "aao_experiment_v1" not in record.metadata
    assert record.run_id is None
    assert record.input_tokens is None
    assert record.tool_calls is None
    assert record.files_changed is None

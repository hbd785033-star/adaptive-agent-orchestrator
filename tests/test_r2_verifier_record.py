"""R2 verifier and execution-record acceptance tests."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from contracts.delegation import ChildExecution, DelegationResult
from contracts.evaluation import EvalCheck, EvalResult, EvalStatus
from contracts.execution import ExecutionRecord, SuccessCriterion, TaskOutcome
from contracts.result import AgentResult, RunStatus, Usage
from contracts.task import TaskContract
from evals.gate import check_success_criteria
from evals.verifier import CriterionVerifier
from orchestrator.delegation_executor import DelegationExecutor
from orchestrator.engine import Orchestrator


def test_completed_is_not_passed_until_all_criteria_verify(tmp_path):
    outcome = TaskOutcome(completed=True, criteria=[])
    assert outcome.completed
    assert not outcome.passed

    criteria = [SuccessCriterion(type="file_exists", target="missing.txt")]
    verified = CriterionVerifier(tmp_path).verify(criteria, completed=True)
    assert verified.completed
    assert not verified.passed
    assert verified.criteria[0].passed is False


def test_plain_string_success_criterion_is_unverifiable_not_pass(tmp_path):
    task = TaskContract(goal="verify truth", success_criteria=["tests must pass"])

    check = check_success_criteria(tmp_path, task, completed=True)

    assert check.status.value == "fail"
    assert check.blocker
    assert "unverifiable" in check.detail


def test_structured_success_criterion_can_pass(tmp_path):
    (tmp_path / "proof.txt").write_text("verified\n")
    task = TaskContract(
        goal="verify truth",
        success_criteria=[SuccessCriterion(type="file_exists", target="proof.txt")],
    )

    check = check_success_criteria(tmp_path, task, completed=True)

    assert check.status.value == "pass"


def test_structured_success_criterion_failure_remains_authoritative(tmp_path):
    task = TaskContract(
        goal="verify truth",
        success_criteria=[SuccessCriterion(type="file_exists", target="missing.txt")],
    )

    check = check_success_criteria(tmp_path, task, completed=True)

    assert check.status.value == "fail"
    assert check.blocker


def test_mixed_criteria_cannot_pass_with_unverifiable_string(tmp_path):
    (tmp_path / "proof.txt").write_text("verified\n")
    task = TaskContract(
        goal="verify truth",
        success_criteria=[
            SuccessCriterion(type="file_exists", target="proof.txt"),
            "human says it looks good",
        ],
    )

    check = check_success_criteria(tmp_path, task, completed=True)

    assert check.status.value == "fail"
    assert "unverifiable" in check.detail


def test_verifier_v1_types(tmp_path):
    (tmp_path / "result.txt").write_text("hello verified world\n")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "result.txt").write_text("hello verified world\nchanged\n")

    verifier = CriterionVerifier(tmp_path, registered={"custom": lambda c, root: (True, "ok")})
    outcome = verifier.verify([
        SuccessCriterion(type="file_exists", target="result.txt"),
        SuccessCriterion(type="file_contains", target="result.txt", value="verified"),
        SuccessCriterion(type="command", command=["python", "-c", "print('ok')"]),
        SuccessCriterion(type="git_diff", target="result.txt"),
        SuccessCriterion(type="registered", name="custom"),
    ], completed=True)
    assert outcome.passed


def test_execution_record_01_exact_export_round_trip(tmp_path):
    record = ExecutionRecord(
        task_id="task-1",
        run_id="exec-1",
        model="model-a",
        provider="provider-a",
        harness="adaptive-agent-orchestrator",
        status="completed",
        started_at="2026-08-10T10:00:00Z",
        finished_at="2026-08-10T10:00:01Z",
        latency_seconds=1.0,
        input_tokens=10,
        output_tokens=5,
        cached_tokens=0,
        cost_usd=0.01,
        tool_calls=[],
        files_changed=["x"],
        output="done",
        workspace_root="/workspace",
        isolation_level="workspace",
        metadata={"route": "single", "verification_status": "passed", "trial": 1},
    )
    destination = tmp_path / "record.json"
    record.export(destination)
    raw = json.loads(destination.read_text())
    assert raw["schema_version"] == "0.1"
    assert set(raw) == {
        "schema_version", "task_id", "run_id", "model", "provider", "harness",
        "status", "started_at", "finished_at", "latency_seconds", "input_tokens",
        "output_tokens", "cached_tokens", "cost_usd", "tool_calls", "files_changed",
        "output", "workspace_root", "isolation_level", "metadata",
    }
    assert ExecutionRecord.model_validate_json(destination.read_text()) == record


def test_execution_record_distinguishes_unknown_from_observed_zero_and_empty():
    common = {
        "task_id": "task-truth",
        "status": "failed",
        "started_at": "2026-08-10T10:00:00Z",
        "finished_at": "2026-08-10T10:00:01Z",
        "latency_seconds": 1.0,
    }
    unknown = ExecutionRecord(
        **common,
        run_id=None,
        model=None,
        provider=None,
        input_tokens=None,
        output_tokens=None,
        cached_tokens=None,
        cost_usd=None,
        tool_calls=None,
        files_changed=None,
        isolation_level=None,
    )
    observed_empty = ExecutionRecord(
        **common,
        run_id="run-observed",
        model="model-observed",
        provider="provider-observed",
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        cost_usd=0.0,
        tool_calls=[],
        files_changed=[],
        isolation_level="workspace",
    )

    assert unknown.input_tokens is None
    assert unknown.tool_calls is None
    assert unknown.files_changed is None
    assert observed_empty.input_tokens == 0
    assert observed_empty.tool_calls == []
    assert observed_empty.files_changed == []


def test_cli_record_builder_never_fabricates_unobserved_evidence():
    from aao_cli.main import _build_execution_record

    unknown = _build_execution_record(
        task_id="task-cli",
        result={"outcome": "failed", "route": "delegation", "detail": "blocked"},
        mock=False,
        started_at="2026-08-10T10:00:00Z",
        finished_at="2026-08-10T10:00:01Z",
    )
    observed_zero = _build_execution_record(
        task_id="task-cli",
        result={
            "outcome": "completed",
            "run_id": "run-0",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
            "files_changed": [],
            "isolation_level": "workspace",
        },
        mock=True,
        started_at="2026-08-10T10:00:00Z",
        finished_at="2026-08-10T10:00:01Z",
    )

    assert unknown.run_id is None
    assert unknown.model is None
    assert unknown.provider is None
    assert unknown.input_tokens is None
    assert unknown.cost_usd is None
    assert unknown.tool_calls is None
    assert unknown.files_changed is None
    assert unknown.isolation_level is None
    assert observed_zero.run_id == "run-0"
    assert observed_zero.input_tokens == 0
    assert observed_zero.cost_usd == 0.0
    assert observed_zero.files_changed == []


def test_delegated_summary_preserves_child_evidence_without_parent_fabrication():
    evaluation = EvalResult.aggregate(
        "task-delegated",
        "run-child",
        [EvalCheck(name="proof", status=EvalStatus.PASS)],
    )
    observed = DelegationResult(
        parent_task_id="task-delegated",
        children=[
            ChildExecution(
                child_id="child-1",
                run_id="run-child",
                attempt_run_ids=["run-old", "run-child"],
                status="completed",
                result=AgentResult(
                    run_id="run-child",
                    task_id="task-delegated",
                    status=RunStatus.COMPLETED,
                ),
                eval_result=evaluation,
            )
        ],
    )
    record = SimpleNamespace(
        task=SimpleNamespace(id="task-delegated"), retry_count=0
    )

    summary = Orchestrator._summary_delegation(
        record,
        observed,
        files_changed=["observed.txt"],
        verification_status="pass",
    )

    assert summary["run_id"] is None
    assert summary["child_runs"] == [
        {"child_id": "child-1", "run_id": "run-old"},
        {"child_id": "child-1", "run_id": "run-child"},
    ]
    assert summary["files_changed"] == ["observed.txt"]
    assert summary["isolation_level"] == "workspace"
    assert summary["verification_status"] == "pass"


def test_child_execution_distinguishes_unknown_from_observed_zero_usage():
    unknown = ChildExecution(child_id="unknown", status="failed")
    observed_zero = ChildExecution(
        child_id="blocked",
        status="cancelled",
        run_id=None,
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
    )

    assert unknown.run_id is None
    assert unknown.attempt_run_ids == []
    assert unknown.input_tokens is None
    assert unknown.output_tokens is None
    assert unknown.estimated_cost_usd is None
    assert unknown.usage is None
    assert observed_zero.run_id is None
    assert observed_zero.usage == Usage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
    )


def test_delegation_totals_are_complete_or_unknown_per_dimension():
    partial = DelegationResult(
        parent_task_id="parent",
        children=[
            ChildExecution(
                child_id="observed",
                run_id="run-1",
                input_tokens=100,
                output_tokens=20,
                estimated_cost_usd=0.25,
            ),
            ChildExecution(
                child_id="unknown",
                run_id="run-2",
                input_tokens=None,
                output_tokens=0,
                estimated_cost_usd=None,
            ),
        ],
    )
    complete = DelegationResult(
        parent_task_id="parent",
        children=[
            ChildExecution(
                child_id="zero",
                run_id=None,
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
            ),
            ChildExecution(
                child_id="observed",
                run_id="run-3",
                input_tokens=200,
                output_tokens=30,
                estimated_cost_usd=0.5,
            ),
        ],
    )

    assert partial.total_input_tokens is None
    assert partial.total_output_tokens == 20
    assert partial.total_cost_usd is None
    assert complete.total_input_tokens == 200
    assert complete.total_output_tokens == 30
    assert complete.total_cost_usd == 0.5
    partial_usage = Usage(
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        estimated_cost_usd=0.25,
    ) + Usage(input_tokens=4, output_tokens=5, total_tokens=9)
    assert partial_usage.input_tokens == 5
    assert partial_usage.output_tokens == 7
    assert partial_usage.estimated_cost_usd is None
    unknown_usage = Usage(
        input_tokens=1,
        output_tokens=None,
        total_tokens=None,
        estimated_cost_usd=0.25,
    ) + Usage(
        input_tokens=4,
        output_tokens=5,
        total_tokens=9,
        estimated_cost_usd=0.5,
    )
    assert unknown_usage.input_tokens == 5
    assert unknown_usage.output_tokens is None
    assert unknown_usage.total_tokens is None
    assert unknown_usage.estimated_cost_usd == 0.75


def test_native_and_aggregate_results_preserve_run_identity_truth():
    native = AgentResult(
        run_id="native-run",
        task_id="native-task",
        status=RunStatus.COMPLETED,
    )
    native_eval = EvalResult.aggregate(
        native.task_id,
        native.run_id,
        [EvalCheck(name="native", status=EvalStatus.PASS)],
    )
    child_eval = EvalResult.aggregate(
        "parent",
        "child-run",
        [EvalCheck(name="child", status=EvalStatus.PASS)],
    )
    delegated = DelegationResult(
        parent_task_id="parent",
        children=[
            ChildExecution(
                child_id="child",
                run_id="child-run",
                status="completed",
                result=AgentResult(
                    run_id="child-run",
                    task_id="parent",
                    status=RunStatus.COMPLETED,
                ),
                eval_result=child_eval,
                input_tokens=10,
                output_tokens=5,
                estimated_cost_usd=None,
            )
        ],
    )

    aggregate = DelegationExecutor._aggregate(delegated)
    assert native.run_id == "native-run"
    assert native_eval.run_id == "native-run"
    assert aggregate is not None
    assert aggregate.run_id is None
    aggregate_eval = EvalResult.aggregate(
        aggregate.task_id,
        aggregate.run_id,
        [EvalCheck(name="aggregate", status=EvalStatus.PASS)],
    )
    assert aggregate_eval.run_id is None


def test_delegated_summary_keeps_unavailable_parent_and_child_identity_absent():
    record = SimpleNamespace(
        task=SimpleNamespace(id="task-delegated"), retry_count=0
    )
    unavailable = DelegationResult(
        parent_task_id="task-delegated",
        children=[
            ChildExecution(
                child_id="child-1",
                run_id=None,
                status="cancelled",
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
            )
        ],
    )
    unavailable_summary = Orchestrator._summary_delegation(record, unavailable)
    assert unavailable_summary["run_id"] is None
    assert unavailable_summary["child_runs"] == []
    assert unavailable_summary["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

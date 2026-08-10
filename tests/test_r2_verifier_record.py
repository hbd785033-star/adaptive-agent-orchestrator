"""R2 verifier and execution-record acceptance tests."""
from __future__ import annotations

import json
import subprocess

from contracts.execution import ExecutionRecord, SuccessCriterion, TaskOutcome
from evals.verifier import CriterionVerifier


def test_completed_is_not_passed_until_all_criteria_verify(tmp_path):
    outcome = TaskOutcome(completed=True, criteria=[])
    assert outcome.completed
    assert not outcome.passed

    criteria = [SuccessCriterion(type="file_exists", target="missing.txt")]
    verified = CriterionVerifier(tmp_path).verify(criteria, completed=True)
    assert verified.completed
    assert not verified.passed
    assert verified.criteria[0].passed is False


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

"""R2 shared policy and read-only enforcement tests."""

from __future__ import annotations

import subprocess

from contracts.result import AgentResult, RunStatus
from contracts.task import RiskLevel, TaskContract, TaskType
from evals.gate import DeterministicEvalGate
from orchestrator.budget import BudgetConfig, BudgetState
from orchestrator.execution_policy import ApprovalOutcome, ExecutionPolicy
from orchestrator.prompt_guard import inject_constraints


def _task(**updates):
    values = {
        "id": "policy-task",
        "goal": "Inspect the repository",
        "task_type": TaskType.CODE_REVIEW,
        "risk": RiskLevel.LOW,
    }
    values.update(updates)
    return TaskContract(**values)


def test_execution_policy_applies_same_decision_to_single_and_child():
    policy = ExecutionPolicy(
        always_require_actions={"delete_files"},
        require_approval_above_calls=2,
        max_total_calls=3,
        approval_timeout_s=5,
    )
    task = _task(task_type=TaskType.CODE_FIX)
    event = {"action": "delete_files", "reason": "cleanup"}

    single = policy.authorize_event(task, event, calls_used=1, approval=False)
    child = policy.authorize_event(task, event, calls_used=1, approval=False)

    assert single == child
    assert single.outcome == ApprovalOutcome.DENIED
    assert not single.allowed


def test_execution_policy_denies_budget_and_timeout():
    policy = ExecutionPolicy(
        require_approval_above_calls=2,
        max_total_calls=3,
        approval_timeout_s=5,
    )
    task = _task(task_type=TaskType.CODE_FIX)

    assert policy.authorize_event(task, {}, calls_used=3).outcome == ApprovalOutcome.BUDGET_EXCEEDED
    assert policy.authorize_event(task, {}, calls_used=2).outcome == ApprovalOutcome.TIMEOUT


async def test_readonly_prompt_and_trusted_diff_failure(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True
    )
    (tmp_path / "tracked.txt").write_text("before\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, check=True)

    task = _task(context={"_eval_base_sha": "HEAD"})
    guarded = inject_constraints(task)
    assert "READ-ONLY" in guarded.goal

    (tmp_path / "tracked.txt").write_text("after\n")
    result = AgentResult(
        run_id="run-1",
        task_id=task.id,
        status=RunStatus.COMPLETED,
        files_changed=[],
    )
    budget = BudgetState(task_id=task.id, config=BudgetConfig())

    evaluated = await DeterministicEvalGate(tmp_path).run(task, result, budget)

    readonly = next(check for check in evaluated.checks if check.name == "read_only")
    assert readonly.status.value == "fail"
    assert "tracked.txt" in readonly.detail

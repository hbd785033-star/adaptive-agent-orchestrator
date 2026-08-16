"""R2 shared policy and read-only enforcement tests."""

from __future__ import annotations

import subprocess

from contracts.result import AgentResult, RunStatus
from contracts.task import RiskLevel, TaskContract, TaskType
from evals.gate import DeterministicEvalGate
from orchestrator.budget import BudgetConfig, BudgetState
from orchestrator.execution_policy import ApprovalOutcome, ExecutionPolicy
from orchestrator.prompt_guard import inject_constraints
from orchestrator.workspace import RepositoryBaseline


def _task(**updates):
    values = {
        "id": "policy-task",
        "goal": "Inspect the repository",
        "task_type": TaskType.CODE_REVIEW,
        "risk": RiskLevel.LOW,
    }
    values.update(updates)
    return TaskContract(**values)


def test_runtime_approval_request_is_always_fail_closed():
    policy = ExecutionPolicy(
        always_require_actions={"delete_files"},
        require_approval_above_calls=2,
        max_total_calls=3,
        approval_timeout_s=5,
    )
    task = _task(task_type=TaskType.CODE_FIX)
    denied = policy.reject_runtime_approval_request(task, calls_used=1, approval=False)
    affirmative = policy.reject_runtime_approval_request(
        task, calls_used=1, approval=True
    )

    assert denied.outcome == ApprovalOutcome.DENIED
    assert affirmative.outcome == ApprovalOutcome.DENIED
    assert not denied.allowed
    assert not affirmative.allowed


def test_execution_policy_denies_budget_and_timeout():
    policy = ExecutionPolicy(
        require_approval_above_calls=2,
        max_total_calls=3,
        approval_timeout_s=5,
    )
    task = _task(task_type=TaskType.CODE_FIX)

    assert policy.authorize_submission(task, calls_used=3).outcome == ApprovalOutcome.BUDGET_EXCEEDED
    assert policy.authorize_submission(task, calls_used=2).outcome == ApprovalOutcome.TIMEOUT


def test_explicit_generic_approval_denial_is_authoritative():
    policy = ExecutionPolicy(
        always_require_actions={"delete_files"},
        require_approval_above_calls=5,
        max_total_calls=8,
    )
    task = _task(task_type=TaskType.CODE_FIX)

    denied = policy.authorize_submission(
        task, calls_used=0, planned_actions={"delete_files"}, approval=False
    )
    missing = policy.authorize_submission(
        task, calls_used=0, planned_actions={"delete_files"}, approval=None
    )

    assert denied.outcome == ApprovalOutcome.DENIED
    assert not denied.allowed
    assert missing.outcome == ApprovalOutcome.TIMEOUT
    assert not missing.allowed


def test_submission_threshold_requires_approval_before_next_call():
    policy = ExecutionPolicy(require_approval_above_calls=2, max_total_calls=8)
    task = _task(task_type=TaskType.CODE_FIX)

    below = policy.authorize_submission(task, calls_used=1)
    protected = policy.authorize_submission(task, calls_used=2)
    approved = policy.authorize_submission(task, calls_used=2, approval=True)

    assert below.outcome == ApprovalOutcome.ALLOWED
    assert protected.outcome == ApprovalOutcome.TIMEOUT
    assert not protected.allowed
    assert approved.outcome == ApprovalOutcome.APPROVED


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


async def test_readonly_detects_ignored_file_created_after_baseline(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("*.generated\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    baseline = RepositoryBaseline.capture(tmp_path)
    (tmp_path / "agent.generated").write_text("mutation\n")

    task = _task(context={"_eval_base_sha": base})
    result = AgentResult(run_id="run-ignored", task_id=task.id, status=RunStatus.COMPLETED)
    budget = BudgetState(task_id=task.id, config=BudgetConfig())

    evaluated = await DeterministicEvalGate(tmp_path).run(task, result, budget)

    readonly = next(check for check in evaluated.checks if check.name == "read_only")
    assert readonly.status.value == "pass"
    assert baseline.changed() == ["agent.generated"]


async def test_readonly_rescans_after_verifier_commands(tmp_path, monkeypatch):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "seed"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    async def mutating_check(repo_path, _changed_files):
        from contracts.evaluation import EvalCheck, EvalStatus

        (repo_path / "late.txt").write_text("created by verifier\n")
        return EvalCheck(name="tests", status=EvalStatus.PASS)

    monkeypatch.setattr("evals.gate.check_tests", mutating_check)
    task = _task(context={"_eval_base_sha": base})
    result = AgentResult(run_id="run-late", task_id=task.id, status=RunStatus.COMPLETED)
    budget = BudgetState(task_id=task.id, config=BudgetConfig())

    evaluated = await DeterministicEvalGate(tmp_path).run(task, result, budget)

    readonly = next(check for check in evaluated.checks if check.name == "read_only")
    assert readonly.status.value == "fail"
    assert "late.txt" in readonly.detail

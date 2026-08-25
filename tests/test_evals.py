"""Tests for DeterministicEvalGate — paths and budget checks (sync)."""
from __future__ import annotations

import subprocess

import pytest

from contracts.evaluation import EvalStatus
from contracts.execution import SuccessCriterion
from contracts.result import AgentResult, RunStatus, Usage
from contracts.task import TaskContract, WorkspaceSpec
from evals.gate import DeterministicEvalGate, check_budget, check_paths
from orchestrator.budget import BudgetConfig, BudgetState


def make_result(files_changed: list[str], *, summary: str | None = None) -> AgentResult:
    return AgentResult(
        run_id="run-1",
        task_id="task-1",
        status=RunStatus.COMPLETED,
        files_changed=files_changed,
        summary=summary,
        usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150),
    )


def make_task(allowed_paths: list[str]) -> TaskContract:
    return TaskContract(goal="fix bug", allowed_paths=allowed_paths)


def make_budget(calls: int = 2, max_calls: int = 8) -> BudgetState:
    cfg = BudgetConfig(max_total_calls=max_calls)
    b = BudgetState(task_id="task-1", config=cfg)
    b.calls_used = calls
    return b


class TestCheckPaths:
    def test_no_allowed_paths_skips(self):
        task = make_task([])
        result = make_result(["src/foo.py"])
        check = check_paths(result, task)
        assert check.status == EvalStatus.SKIP

    def test_files_within_allowed(self):
        task = make_task(["src/**", "tests/**"])
        result = make_result(["src/auth.py", "tests/test_auth.py"])
        check = check_paths(result, task)
        assert check.status == EvalStatus.PASS

    def test_file_outside_allowed_fails(self):
        task = make_task(["src/auth/**"])
        result = make_result(["src/auth/login.py", "src/payments/charge.py"])
        check = check_paths(result, task)
        assert check.status == EvalStatus.FAIL
        assert check.blocker is True
        assert "src/payments/charge.py" in check.detail

    def test_glob_wildcard(self):
        task = make_task(["src/auth/**"])
        result = make_result(["src/auth/login.py", "src/auth/models/user.py"])
        check = check_paths(result, task)
        assert check.status == EvalStatus.PASS

    def test_no_files_changed_passes(self):
        task = make_task(["src/**"])
        result = make_result([])
        check = check_paths(result, task)
        assert check.status == EvalStatus.PASS

    def test_traversal_and_absolute_paths_fail_closed(self, tmp_path):
        task = make_task(["src/**"])
        traversal = check_paths(
            make_result(["src/../secrets.txt"]), task, repo_path=tmp_path
        )
        absolute = check_paths(
            make_result([str(tmp_path / "src" / "auth.py")]), task, repo_path=tmp_path
        )
        assert traversal.status == EvalStatus.FAIL
        assert absolute.status == EvalStatus.FAIL

    @pytest.mark.asyncio
    async def test_gate_uses_workspace_git_state_not_runtime_report(self, tmp_path):
        root = tmp_path / "root"
        workspace = tmp_path / "workspace"
        root.mkdir()
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
        (root / "README.md").write_text("base")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "worktree", "add", str(workspace)],
            cwd=root,
            check=True,
            capture_output=True,
        )
        (workspace / "outside.txt").write_text("unreported")

        task = make_task(["src/**"])
        task.workspace = WorkspaceSpec(path=str(workspace))
        gate = DeterministicEvalGate(root)
        evaluated = await gate.run(task, make_result([]), make_budget())

        path_check = next(check for check in evaluated.checks if check.name == "paths")
        assert path_check.status == EvalStatus.FAIL
        assert "outside.txt" in path_check.detail


class TestCheckBudget:
    def test_within_budget_passes(self):
        check = check_budget(make_result([]), make_budget(calls=3, max_calls=8))
        assert check.status == EvalStatus.PASS

    def test_at_limit_passes_postflight(self):
        check = check_budget(make_result([]), make_budget(calls=8, max_calls=8))
        assert check.status == EvalStatus.PASS

    def test_over_limit_fails(self):
        check = check_budget(make_result([]), make_budget(calls=10, max_calls=8))
        assert check.status == EvalStatus.FAIL


class TestObservedOutputCriteria:
    @pytest.mark.asyncio
    async def test_gate_supplies_authoritative_runtime_output(self, tmp_path):
        task = TaskContract(
            goal="reply exactly",
            success_criteria=[SuccessCriterion(type="output_equals", value="OK")],
        )

        evaluated = await DeterministicEvalGate(tmp_path).run(
            task,
            make_result([], summary="OK"),
            make_budget(),
        )

        check = next(item for item in evaluated.checks if item.name == "success_criteria")
        assert check.status == EvalStatus.PASS

    @pytest.mark.asyncio
    async def test_gate_does_not_use_runtime_error_as_output(self, tmp_path):
        task = TaskContract(
            goal="reply exactly",
            context={"detail": "OK"},
            success_criteria=[SuccessCriterion(type="output_equals", value="OK")],
        )
        result = make_result([], summary=None)
        result.error = "OK"

        evaluated = await DeterministicEvalGate(tmp_path).run(task, result, make_budget())

        check = next(item for item in evaluated.checks if item.name == "success_criteria")
        assert check.status == EvalStatus.FAIL


class TestBudgetReservations:
    def test_reservation_prevents_parallel_oversubmission(self):
        budget = make_budget(calls=0, max_calls=1)
        assert budget.reserve_calls(1) is None
        violation = budget.reserve_calls(1)
        assert violation is not None
        assert budget.calls_reserved == 1
        budget.commit_reserved_call()
        assert budget.calls_used == 1
        assert budget.calls_reserved == 0

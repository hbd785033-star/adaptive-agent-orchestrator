"""Tests for DeterministicEvalGate — paths and budget checks (sync)."""
from __future__ import annotations

import pytest
from contracts.evaluation import EvalStatus
from contracts.result import AgentResult, RunStatus, Usage
from contracts.task import TaskContract, RiskLevel
from evals.gate import check_paths, check_budget
from orchestrator.budget import BudgetConfig, BudgetState


def make_result(files_changed: list[str]) -> AgentResult:
    return AgentResult(
        run_id="run-1",
        task_id="task-1",
        status=RunStatus.COMPLETED,
        files_changed=files_changed,
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


class TestCheckBudget:
    def test_within_budget_passes(self):
        check = check_budget(make_result([]), make_budget(calls=3, max_calls=8))
        assert check.status == EvalStatus.PASS

    def test_at_limit_fails(self):
        check = check_budget(make_result([]), make_budget(calls=8, max_calls=8))
        assert check.status == EvalStatus.FAIL
        assert check.blocker is True

    def test_over_limit_fails(self):
        check = check_budget(make_result([]), make_budget(calls=10, max_calls=8))
        assert check.status == EvalStatus.FAIL

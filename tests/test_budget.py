"""Tests for Budget Gate and Approval Gate."""
from __future__ import annotations

import pytest
from orchestrator.budget import BudgetConfig, BudgetState, BudgetViolation, ApprovalGate
from contracts.task import TaskContract, RiskLevel, TaskType


@pytest.fixture
def config():
    return BudgetConfig(max_children=2, max_depth=1, max_retries=1, max_total_calls=4,
                        require_approval_above_calls=3)


@pytest.fixture
def budget(config):
    return BudgetState(task_id="test-task", config=config)


class TestBudgetState:
    def test_no_violation_initially(self, budget):
        assert budget.check_children() is None
        assert budget.check_retries() is None
        assert budget.check_calls() is None

    def test_children_violation(self, budget):
        budget.children_used = 2
        v = budget.check_children()
        assert v is not None
        assert isinstance(v, BudgetViolation)
        assert v.field == "max_children"

    def test_retries_violation(self, budget):
        budget.retries_used = 1
        assert budget.check_retries() is not None

    def test_calls_violation(self, budget):
        budget.calls_used = 4
        assert budget.check_calls() is not None

    def test_needs_approval_threshold(self, budget):
        budget.calls_used = 2
        assert not budget.needs_approval_for_calls()
        budget.calls_used = 3
        assert budget.needs_approval_for_calls()


class TestApprovalGate:
    @pytest.fixture
    def gate(self, tmp_path):
        policy = tmp_path / "default.yaml"
        policy.write_text(
            """
approval:
  always_require:
    - delete_files
    - deploy
    - push_to_main
  require_for_risk_levels:
    - 3
    - 4
budget:
  max_children: 2
  max_depth: 1
  max_retries: 1
  max_total_calls: 8
  require_approval_above_calls: 5
routing:
  delegation: {}
  single: {}
  constraints: {}
worktree:
  base_path: ".worktrees"
  readonly_task_types: []
"""
        )
        return ApprovalGate(policy)

    def _task(self, risk: RiskLevel) -> TaskContract:
        return TaskContract(goal="test", risk=risk,
                            success_criteria=["done"] if risk >= RiskLevel.HIGH else [])

    def test_low_risk_no_approval(self, gate):
        needs, _ = gate.requires_approval(self._task(RiskLevel.LOW))
        assert not needs

    def test_high_risk_needs_approval(self, gate):
        needs, reason = gate.requires_approval(self._task(RiskLevel.HIGH))
        assert needs
        assert "HIGH" in reason

    def test_critical_needs_approval(self, gate):
        needs, _ = gate.requires_approval(self._task(RiskLevel.CRITICAL))
        assert needs

    def test_forbidden_action_triggers(self, gate):
        task = self._task(RiskLevel.LOW)
        needs, reason = gate.requires_approval(task, actions=["deploy"])
        assert needs
        assert "deploy" in reason

    def test_safe_action_no_approval(self, gate):
        task = self._task(RiskLevel.LOW)
        needs, _ = gate.requires_approval(task, actions=["read_file"])
        assert not needs

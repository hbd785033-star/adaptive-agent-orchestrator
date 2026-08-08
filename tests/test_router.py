"""Tests for RuleRouter."""
from __future__ import annotations

import pytest
from contracts.task import TaskContract, TaskType, RiskLevel
from orchestrator.router import RuleRouter, RoutingDecision


@pytest.fixture
def router(tmp_path):
    policy = tmp_path / "default.yaml"
    policy.write_text(
        """
policy_version: "routing-v1.0"
routing:
  delegation:
    min_independent_subtasks: 2
    min_estimated_input_tokens: 8000
    allowed_task_types:
      - multi_file_refactor
      - parallel_research
      - test_and_implement
  single:
    max_complexity: 2
    max_affected_modules: 1
  constraints:
    sequential_dependency_forces_single: true
budget:
  max_children: 2
  max_depth: 1
  max_retries: 1
  max_total_calls: 8
  require_approval_above_calls: 5
approval:
  always_require: []
  require_for_risk_levels: [3, 4]
worktree:
  base_path: ".worktrees"
  readonly_task_types:
    - parallel_research
    - code_review
"""
    )
    return RuleRouter(policy)


def make_task(**kwargs) -> TaskContract:
    defaults = dict(goal="fix the login bug", task_type=TaskType.GENERAL, complexity=1)
    defaults.update(kwargs)
    return TaskContract(**defaults)


class TestSequentialDependencyForcesSingle:
    def test_sequential_dep_always_single(self, router):
        task = make_task(goal="first fix auth then deploy", complexity=4,
                         task_type=TaskType.MULTI_FILE_REFACTOR)
        decision = router.route(task, independent_subtask_count=3, has_sequential_dependency=True)
        assert decision.route == "single"
        assert any("sequential_dependency" in r for r in decision.reasons)


class TestSimpleTaskGoesToSingle:
    def test_low_complexity_single_module(self, router):
        task = make_task(complexity=1, allowed_paths=["src/auth/**"])
        decision = router.route(task, affected_module_count=1)
        assert decision.route == "single"

    def test_medium_complexity_still_single_if_one_module(self, router):
        task = make_task(complexity=2, allowed_paths=["src/auth/**"])
        decision = router.route(task, affected_module_count=1)
        assert decision.route == "single"

    def test_complexity_3_single_module_is_still_single(self, router):
        # complexity>max but single module — default fallback
        task = make_task(complexity=3, allowed_paths=["src/auth/**"])
        decision = router.route(task, independent_subtask_count=1, affected_module_count=1)
        # Only one delegation signal needed for delegation; here none fire → single
        assert decision.route == "single"


class TestDelegationTriggers:
    def test_allowed_task_type_triggers_delegation(self, router):
        task = make_task(task_type=TaskType.MULTI_FILE_REFACTOR, complexity=3,
                         allowed_paths=["src/a/**", "src/b/**"])
        decision = router.route(task, independent_subtask_count=1, affected_module_count=2)
        assert decision.route == "delegation"
        assert any("task_type" in r for r in decision.reasons)

    def test_subtask_count_triggers_delegation(self, router):
        task = make_task(complexity=3, allowed_paths=["src/a/**", "src/b/**"])
        decision = router.route(task, independent_subtask_count=2, affected_module_count=2)
        assert decision.route == "delegation"
        assert any("independent_subtasks" in r for r in decision.reasons)

    def test_large_token_estimate_triggers_delegation(self, router):
        long_goal = "x" * 30_000  # ~8571 tokens
        task = make_task(goal=long_goal, complexity=3, allowed_paths=["src/a/**", "src/b/**"])
        decision = router.route(task, affected_module_count=2)
        assert decision.route == "delegation"
        assert any("estimated_input_tokens" in r for r in decision.reasons)


class TestPolicyVersion:
    def test_policy_version_recorded(self, router):
        task = make_task()
        decision = router.route(task)
        assert decision.policy_version == "routing-v1.0"

    def test_to_dict_has_policy_version(self, router):
        task = make_task()
        d = router.route(task).to_dict()
        assert "policy_version" in d
        assert "reasons" in d
        assert "route" in d

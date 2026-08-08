"""Tests for PromptGuard — constraint injection and delegation splitting."""
from __future__ import annotations

from pathlib import Path

import pytest

from contracts.task import RiskLevel, TaskContract, TaskType, WorkspaceSpec
from orchestrator.prompt_guard import inject_constraints, split_for_delegation


def make_task(**kwargs) -> TaskContract:
    defaults = dict(
        task_type=TaskType.GENERAL,
        goal="fix the login bug",
        allowed_paths=["src/auth/**"],
        forbidden_actions=["delete_database", "push_to_main"],
        success_criteria=["all tests pass", "no new warnings"],
        output_schema=["summary", "files_changed", "tests_run", "unresolved_risks"],
        risk=RiskLevel.LOW,
        complexity=2,
    )
    defaults.update(kwargs)
    return TaskContract(**defaults)


class TestInjectConstraints:
    def test_original_not_mutated(self):
        task = make_task()
        original_goal = task.goal
        _ = inject_constraints(task)
        assert task.goal == original_goal

    def test_constraints_appended_to_goal(self):
        task = make_task()
        guarded = inject_constraints(task)
        assert "ORCHESTRATOR CONSTRAINTS" in guarded.goal
        assert task.goal in guarded.goal  # original still present

    def test_allowed_paths_in_goal(self):
        task = make_task(allowed_paths=["src/auth/**", "tests/auth/**"])
        guarded = inject_constraints(task)
        assert "src/auth/**" in guarded.goal
        assert "tests/auth/**" in guarded.goal

    def test_forbidden_actions_in_goal(self):
        task = make_task(forbidden_actions=["delete_database", "push_to_main"])
        guarded = inject_constraints(task)
        assert "delete_database" in guarded.goal
        assert "push_to_main" in guarded.goal

    def test_success_criteria_in_goal(self):
        task = make_task(success_criteria=["all tests pass"])
        guarded = inject_constraints(task)
        assert "all tests pass" in guarded.goal

    def test_output_schema_in_goal(self):
        task = make_task(output_schema=["summary", "files_changed"])
        guarded = inject_constraints(task)
        assert "summary" in guarded.goal
        assert "files_changed" in guarded.goal

    def test_no_constraints_no_block(self):
        """A task with nothing to enforce should not get a constraints block."""
        task = make_task(
            allowed_paths=[],
            forbidden_actions=[],
            success_criteria=[],
            output_schema=[],
        )
        guarded = inject_constraints(task)
        assert "ORCHESTRATOR CONSTRAINTS" not in guarded.goal

    def test_worktree_path_injected(self, tmp_path):
        task = make_task()
        wt = tmp_path / "child-1"
        guarded = inject_constraints(task, worktree_path=wt, child_id="child-1")
        assert str(wt) in guarded.goal
        assert guarded.workspace is not None
        assert guarded.workspace.path == str(wt)
        assert "child-1" in guarded.workspace.branch

    def test_workspace_from_task_spec_used_when_no_wt(self):
        task = make_task(
            workspace=WorkspaceSpec(path="/repo/main", branch="main"),
            allowed_paths=["src/**"],
        )
        guarded = inject_constraints(task)
        assert "/repo/main" in guarded.goal

    def test_child_id_in_goal(self, tmp_path):
        task = make_task()
        guarded = inject_constraints(task, worktree_path=tmp_path, child_id="child-2")
        assert "child-2" in guarded.goal


class TestSplitForDelegation:
    def test_produces_one_contract_per_worktree(self, tmp_path):
        task = make_task()
        worktrees = [
            ("child-1", tmp_path / "child-1"),
            ("child-2", tmp_path / "child-2"),
        ]
        children = split_for_delegation(task, worktrees)
        assert len(children) == 2

    def test_each_child_has_delegation_header(self, tmp_path):
        task = make_task()
        worktrees = [("child-1", tmp_path / "c1"), ("child-2", tmp_path / "c2")]
        children = split_for_delegation(task, worktrees)
        assert "sub-agent 1/2" in children[0].goal
        assert "sub-agent 2/2" in children[1].goal

    def test_each_child_has_its_own_workspace(self, tmp_path):
        task = make_task()
        p1, p2 = tmp_path / "c1", tmp_path / "c2"
        children = split_for_delegation(task, [("child-1", p1), ("child-2", p2)])
        assert children[0].workspace.path == str(p1)
        assert children[1].workspace.path == str(p2)

    def test_original_not_mutated(self, tmp_path):
        task = make_task()
        original_goal = task.goal
        _ = split_for_delegation(task, [("c1", tmp_path / "c1")])
        assert task.goal == original_goal

    def test_constraints_present_in_each_child(self, tmp_path):
        task = make_task(forbidden_actions=["push_to_main"])
        children = split_for_delegation(task, [("c1", tmp_path / "c1"), ("c2", tmp_path / "c2")])
        for child in children:
            assert "push_to_main" in child.goal

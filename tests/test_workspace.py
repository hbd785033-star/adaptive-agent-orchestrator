"""Git worktree delivery lifecycle tests."""
from __future__ import annotations

import os
import subprocess

import pytest

from orchestrator.workspace import WorkspaceManager, WorktreeStatus


def git(repo, *args):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def test_successful_worktree_is_committed_integrated_and_cleaned(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    (repo / "README.md").write_text("base")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    policy = tmp_path / "policy.yaml"
    policy.write_text("worktree:\n  base_path: .worktrees\n  readonly_task_types: []\n")
    manager = WorkspaceManager(repo, policy)

    record = manager.allocate("task-1", "child-1")
    manager.activate("task-1", "child-1")
    artifact = record.worktree_path / "artifact.txt"
    artifact.write_text("delivered")

    commit_sha = manager.commit_changes("task-1", "child-1")
    manager.integrate("task-1", "child-1")

    assert commit_sha
    assert (repo / "artifact.txt").read_text() == "delivered"
    assert record.status == WorktreeStatus.MERGING

    manager.clean("task-1", "child-1")
    assert record.status == WorktreeStatus.CLEANED
    assert not record.worktree_path.exists()


def test_delivery_refuses_untracked_root_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "commit", "--allow-empty", "-m", "base")
    policy = tmp_path / "policy.yaml"
    policy.write_text("worktree:\n  base_path: .worktrees\n  readonly_task_types: []\n")
    manager = WorkspaceManager(repo, policy)
    (repo / "untracked.txt").write_text("must not be hidden")

    with pytest.raises(RuntimeError, match="untracked"):
        manager.ensure_root_clean()


def test_manager_integrate_blocks_root_branch_switch_without_aborting_user_state(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    (repo / "README.md").write_text("base")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    policy = tmp_path / "policy.yaml"
    policy.write_text("worktree:\n  base_path: .worktrees\n  readonly_task_types: []\n")
    manager = WorkspaceManager(repo, policy)
    record = manager.allocate("task-switch", "child-1")
    (record.worktree_path / "artifact.txt").write_text("delivered")
    manager.commit_changes("task-switch", "child-1")
    git(repo, "switch", "-c", "user-branch")

    with pytest.raises(RuntimeError, match="delivery boundary"):
        manager.integrate("task-switch", "child-1")

    assert git(repo, "symbolic-ref", "--short", "HEAD") == "user-branch"
    assert record.status == WorktreeStatus.ABANDONED
    assert record.worktree_path.exists()
    manager.clean("task-switch", "child-1")


def test_manager_integrate_blocks_same_branch_user_commit_and_preserves_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    (repo / "README.md").write_text("base")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    policy = tmp_path / "policy.yaml"
    policy.write_text("worktree:\n  base_path: .worktrees\n  readonly_task_types: []\n")
    manager = WorkspaceManager(repo, policy)
    record = manager.allocate("task-head-move", "child-1")
    (record.worktree_path / "artifact.txt").write_text("delivered")
    manager.commit_changes("task-head-move", "child-1")

    (repo / "user.txt").write_text("user commit")
    git(repo, "add", "user.txt")
    git(repo, "commit", "-m", "concurrent user commit")
    user_head = git(repo, "rev-parse", "HEAD")

    with pytest.raises(RuntimeError, match="delivery boundary"):
        manager.integrate("task-head-move", "child-1")

    assert git(repo, "rev-parse", "HEAD") == user_head
    assert (repo / "README.md").read_text() == "base"
    assert (repo / "user.txt").read_text() == "user commit"
    assert not (repo / "artifact.txt").exists()
    assert record.status == WorktreeStatus.ABANDONED
    assert record.worktree_path.exists()
    manager.clean("task-head-move", "child-1")

"""R2 execution workspace lifecycle tests."""

from __future__ import annotations

import subprocess

import pytest

from orchestrator.workspace import ExecutionWorkspace, WorkspaceUnavailableError


def _git(path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "seed.txt").write_text("seed\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "seed")
    return tmp_path


def test_execution_workspace_integrates_trusted_diff_and_cleans(tmp_path):
    repo = _repo(tmp_path)

    workspace = ExecutionWorkspace.create(repo, task_id="task-1", execution_id="single")
    (workspace.path / "allowed.txt").write_text("delivered\n")

    assert workspace.changed_files() == ["allowed.txt"]
    assert not (repo / "allowed.txt").exists()

    workspace.integrate()

    assert (repo / "allowed.txt").read_text() == "delivered\n"
    assert not workspace.path.exists()
    assert workspace.cleaned
    assert "agent/task-1/single" not in _git(repo, "branch", "--list")


def test_execution_workspace_rollback_removes_all_artifacts(tmp_path):
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    workspace = ExecutionWorkspace.create(repo, task_id="task-2", execution_id="single")
    (workspace.path / "forbidden.txt").write_text("nope\n")

    workspace.rollback()

    assert _git(repo, "rev-parse", "HEAD") == base
    assert _git(repo, "status", "--porcelain") == ""
    assert not workspace.path.exists()
    assert workspace.cleaned


def test_execution_workspace_fails_closed_without_git_repo(tmp_path):
    with pytest.raises(WorkspaceUnavailableError, match="usable Git repository"):
        ExecutionWorkspace.create(tmp_path, task_id="task-3", execution_id="single")

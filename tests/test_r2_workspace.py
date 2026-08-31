"""R2 execution workspace lifecycle tests."""

from __future__ import annotations

import subprocess

import pytest

import orchestrator.workspace as workspace_module
from orchestrator.workspace import (
    ExecutionWorkspace,
    RepositoryBaseline,
    WorkspaceUnavailableError,
    _staging_path,
)


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


def test_execution_workspace_rollback_never_resets_moved_root(tmp_path):
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    workspace = ExecutionWorkspace.create(repo, task_id="task-rollback-boundary")

    (repo / "user.txt").write_text("user change\n")
    _git(repo, "add", "user.txt")
    _git(repo, "commit", "-m", "user commit")
    moved = _git(repo, "rev-parse", "HEAD")

    workspace.rollback()

    assert _git(repo, "rev-parse", "HEAD") == moved
    assert _git(repo, "rev-parse", "HEAD") != base
    assert not workspace.path.exists()
    assert workspace.cleaned


def test_execution_workspace_rollback_preserves_active_user_merge(tmp_path):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-merge-boundary")
    merge_head = repo / ".git" / "MERGE_HEAD"
    merge_head.write_text(_git(repo, "rev-parse", "HEAD") + "\n")

    with pytest.raises(RuntimeError, match="merge state"):
        workspace.rollback()

    assert merge_head.exists()
    assert _git(repo, "rev-parse", "HEAD") == workspace.base_sha
    assert not workspace.cleaned
    merge_head.unlink()


def test_execution_workspace_delivery_preserves_active_user_merge(tmp_path):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-merge-delivery")
    (workspace.path / "allowed.txt").write_text("delivered\n")
    merge_head = repo / ".git" / "MERGE_HEAD"
    merge_head.write_text(_git(repo, "rev-parse", "HEAD") + "\n")

    with pytest.raises(RuntimeError, match="merge state"):
        workspace.integrate()

    assert merge_head.exists()
    assert _git(repo, "rev-parse", "HEAD") == workspace.base_sha
    assert not (repo / "allowed.txt").exists()
    assert not workspace.cleaned
    merge_head.unlink()


def test_execution_workspace_blocks_branch_switch_before_delivery(tmp_path):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-branch-boundary")
    (workspace.path / "allowed.txt").write_text("delivered\n")

    _git(repo, "switch", "-c", "user-branch")
    root_before = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(RuntimeError, match="delivery boundary"):
        workspace.integrate()

    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "user-branch"
    assert _git(repo, "rev-parse", "HEAD") == root_before
    assert not (repo / "allowed.txt").exists()
    assert not workspace.path.exists()
    assert workspace.cleaned


def test_execution_workspace_blocks_unexpected_ref_movement(tmp_path):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-ref-boundary")
    (workspace.path / "allowed.txt").write_text("delivered\n")
    _git(repo, "branch", "unexpected-ref")

    with pytest.raises(RuntimeError, match="ref"):
        workspace.integrate()

    assert not (repo / "allowed.txt").exists()
    assert not workspace.path.exists()
    assert workspace.cleaned


def test_execution_workspace_guarded_target_ref_rejects_racing_movement(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-target-ref-race")
    (workspace.path / "allowed.txt").write_text("delivered\n")
    target_ref = workspace.symbolic_head
    assert target_ref is not None
    tree = _git(repo, "rev-parse", f"{workspace.base_sha}^{{tree}}")
    unexpected_sha = _git(
        repo,
        "commit-tree",
        tree,
        "-p",
        workspace.base_sha,
        "-m",
        "concurrent target movement",
    )
    real_git = workspace_module._git
    moved = False

    def move_target_before_guarded_update(path, args):
        nonlocal moved
        if not moved and args[:2] == ["update-ref", target_ref]:
            moved = True
            _git(repo, "update-ref", target_ref, unexpected_sha, workspace.base_sha)
        return real_git(path, args)

    monkeypatch.setattr(workspace_module, "_git", move_target_before_guarded_update)

    with pytest.raises(RuntimeError, match="update-ref"):
        workspace.integrate()

    assert moved
    assert _git(repo, "rev-parse", target_ref) == unexpected_sha
    assert (repo / "seed.txt").read_text() == "seed\n"
    assert not (repo / "allowed.txt").exists()
    assert not workspace.path.exists()
    assert workspace.cleaned


def test_execution_workspace_merges_in_off_root_staging_and_runs_hooks(tmp_path):
    repo = _repo(tmp_path)
    hook_log = tmp_path.parent / "merge-hook.log"
    hook = repo / ".git" / "hooks" / "post-merge"
    hook.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" >> "' + hook_log.as_posix() + '"\n')
    hook.chmod(0o755)
    workspace = ExecutionWorkspace.create(repo, task_id="task-staging")
    (workspace.path / "allowed.txt").write_text("delivered\n")

    workspace.integrate()

    hook_paths = hook_log.read_text().splitlines()
    assert any(path != str(repo) for path in hook_paths)
    assert (repo / "allowed.txt").read_text() == "delivered\n"
    assert not workspace.path.exists()


def test_execution_workspace_removes_stale_off_root_staging_before_delivery(tmp_path):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-stale-staging")
    stale = _staging_path(repo, workspace.task_id, workspace.execution_id)
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("stale\n")
    (workspace.path / "allowed.txt").write_text("delivered\n")

    workspace.integrate()

    assert not stale.exists()
    assert (repo / "allowed.txt").read_text() == "delivered\n"


def test_execution_workspace_fails_closed_without_git_repo(tmp_path):
    with pytest.raises(WorkspaceUnavailableError, match="usable Git repository"):
        ExecutionWorkspace.create(tmp_path, task_id="task-3", execution_id="single")


def test_cleanup_failure_is_observable_and_never_claimed_clean(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-4", execution_id="single")
    real_git = workspace_module._git

    def fail_worktree_removal(path, args):
        if args[:2] == ["worktree", "remove"]:
            raise RuntimeError("injected worktree removal failure")
        return real_git(path, args)

    monkeypatch.setattr(workspace_module, "_git", fail_worktree_removal)
    with pytest.raises(RuntimeError, match="worktree removal failure"):
        workspace.cleanup()

    assert not workspace.cleaned
    assert workspace.path.exists()

    monkeypatch.setattr(workspace_module, "_git", real_git)
    workspace.cleanup()


def test_branch_cleanup_failure_is_observable_and_never_claimed_clean(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-5", execution_id="single")
    real_git = workspace_module._git

    def fail_branch_removal(path, args):
        if args[:2] == ["branch", "-D"]:
            raise RuntimeError("injected branch deletion failure")
        return real_git(path, args)

    monkeypatch.setattr(workspace_module, "_git", fail_branch_removal)
    with pytest.raises(RuntimeError, match="branch deletion failure"):
        workspace.cleanup()

    assert not workspace.cleaned
    assert workspace.branch in _git(repo, "branch", "--list")
    monkeypatch.setattr(workspace_module, "_git", real_git)
    workspace.cleanup()


def test_cleanup_rejects_active_root_merge_state(tmp_path):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-6", execution_id="single")
    merge_head = repo / ".git" / "MERGE_HEAD"
    merge_head.write_text(_git(repo, "rev-parse", "HEAD") + "\n")

    with pytest.raises(RuntimeError, match="merge state"):
        workspace.cleanup()

    assert not workspace.cleaned
    merge_head.unlink()


def test_rollback_preserves_residual_root_mutation_and_cleans_artifacts(tmp_path):
    repo = _repo(tmp_path)
    workspace = ExecutionWorkspace.create(repo, task_id="task-7", execution_id="single")
    (repo / "residual.txt").write_text("unsafe root mutation\n")

    workspace.rollback()

    assert (repo / "residual.txt").read_text() == "unsafe root mutation\n"
    assert workspace.cleaned
    assert not workspace.path.exists()


def test_repository_baseline_attributes_ignored_mutations_not_existing_state(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("*.db\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore databases")
    (repo / "existing.db").write_text("baseline")
    baseline = RepositoryBaseline.capture(repo)

    assert baseline.changed() == []
    (repo / "existing.db").write_text("mutated")
    assert baseline.changed() == ["existing.db"]


def test_repository_baseline_detects_new_ignored_file_and_clean_commit(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("*.db\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore databases")
    baseline = RepositoryBaseline.capture(repo)

    (repo / "new.db").write_text("new ignored evidence")
    assert baseline.changed() == ["new.db"]

    (repo / "new.db").unlink()
    (repo / "committed.txt").write_text("committed mutation")
    _git(repo, "add", "committed.txt")
    _git(repo, "commit", "-m", "runtime commit")
    changes = baseline.changed()
    assert "<HEAD>" in changes
    assert "committed.txt" in changes


def test_repository_baseline_detects_symbolic_head_and_ref_mutations(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "branch", "same-commit")
    baseline = RepositoryBaseline.capture(repo)

    _git(repo, "symbolic-ref", "HEAD", "refs/heads/same-commit")
    assert "<symbolic-HEAD>" in baseline.changed()

    baseline = RepositoryBaseline.capture(repo)
    _git(repo, "branch", "runtime-generated")
    assert "<refs>" in baseline.changed()


def test_repository_baseline_detects_object_only_mutation(tmp_path):
    repo = _repo(tmp_path)
    baseline = RepositoryBaseline.capture(repo)

    subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input="persistent object without ref\n",
        capture_output=True,
        text=True,
        check=True,
    )

    assert "<objects>" in baseline.changed()


def test_repository_baseline_detects_extra_worktree_registration(tmp_path):
    repo = _repo(tmp_path)
    extra = tmp_path.parent / f"{tmp_path.name}-extra-worktree"
    baseline = RepositoryBaseline.capture(repo)

    _git(repo, "worktree", "add", "--detach", str(extra), "HEAD")
    try:
        assert "<worktrees>" in baseline.changed()
    finally:
        _git(repo, "worktree", "remove", "--force", str(extra))


def test_repository_baseline_separately_covers_excluded_internal_worktrees(
    tmp_path,
):
    repo = _repo(tmp_path)
    internal = repo / ".worktrees"
    existing = internal / "existing" / "state.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("baseline")
    baseline = RepositoryBaseline.capture(
        repo,
        excluded_paths=(".worktrees",),
        protected_paths=(internal,),
    )

    assert baseline.changed() == []
    sibling = internal / "unrelated-sibling" / "leak.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("runtime mutation")

    assert "<internal-worktrees>" in baseline.changed()


def test_repository_baseline_accepts_unchanged_preexisting_control_plane(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "branch", "pre-existing")
    subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input="pre-existing unreachable object\n",
        capture_output=True,
        text=True,
        check=True,
    )
    internal = repo / ".worktrees"
    internal.mkdir()
    (internal / "pre-existing.txt").write_text("baseline")

    baseline = RepositoryBaseline.capture(
        repo,
        excluded_paths=(".worktrees",),
        protected_paths=(internal,),
    )

    assert baseline.changed() == []


@pytest.mark.parametrize(
    ("set_flag", "clear_flag"),
    [
        ("--assume-unchanged", "--no-assume-unchanged"),
        ("--skip-worktree", "--no-skip-worktree"),
    ],
)
def test_repository_baseline_detects_persistent_index_flags(
    tmp_path, set_flag, clear_flag
):
    repo = _repo(tmp_path)
    baseline = RepositoryBaseline.capture(repo)

    _git(repo, "update-index", set_flag, "seed.txt")
    try:
        assert baseline.changed() == ["<index-flags>"]
    finally:
        _git(repo, "update-index", clear_flag, "seed.txt")

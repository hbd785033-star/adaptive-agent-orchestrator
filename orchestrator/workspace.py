"""
WorkspaceManager — Git worktree lifecycle for code-writing delegation tasks.

States:
    ALLOCATED → ACTIVE → MERGING → CLEANED
                       ↘ ABANDONED          (eval fail; keep for inspection)

Rules:
    - Read-only task types (research, review) → no worktree allocated.
    - Every writing child agent gets exactly one isolated worktree + branch.
    - Orchestrator allocates the path and injects it into TaskContract.workspace.
    - Cleanup of ABANDONED worktrees requires explicit human confirmation.
"""
from __future__ import annotations

import contextlib
import dataclasses
import re
import subprocess
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

import yaml


class WorkspaceUnavailableError(RuntimeError):
    """Raised when a write task cannot obtain a safe Git workspace."""


@dataclasses.dataclass
class ExecutionWorkspace:
    """Isolated Git worktree used by every write execution."""

    repo_path: Path
    path: Path
    branch: str
    task_id: str
    execution_id: str
    base_sha: str
    cleaned: bool = False

    @classmethod
    def create(
        cls,
        repo_path: str | Path,
        task_id: str,
        execution_id: str = "single",
        base_path: str | Path = ".worktrees",
    ) -> ExecutionWorkspace:
        repo = Path(repo_path).resolve()
        if not _is_git_repo(repo):
            raise WorkspaceUnavailableError(
                f"write task requires a usable Git repository: {repo}"
            )
        _validate_identifier(task_id, "task_id")
        _validate_identifier(execution_id, "execution_id")
        dirty = _git(repo, ["status", "--porcelain", "--untracked-files=all"])
        if dirty.strip():
            raise WorkspaceUnavailableError("root repository must be clean before execution")

        root = Path(base_path)
        if not root.is_absolute():
            root = repo / root
        path = root / task_id / execution_id
        branch = f"agent/{task_id}/{execution_id}"
        base_sha = _git(repo, ["rev-parse", "HEAD"]).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _git(repo, ["worktree", "add", "-b", branch, str(path), base_sha])
        except RuntimeError as exc:
            raise WorkspaceUnavailableError(str(exc)) from exc
        return cls(
            repo_path=repo,
            path=path,
            branch=branch,
            task_id=task_id,
            execution_id=execution_id,
            base_sha=base_sha,
        )

    @property
    def worktree_path(self) -> Path:
        """Backward-compatible alias used by existing workspace callers."""
        return self.path

    def changed_files(self) -> list[str]:
        """Return the trusted Git diff, including untracked files."""
        diff = _git(self.path, ["diff", "--name-only", "-z", self.base_sha])
        untracked = _git(
            self.path, ["ls-files", "--others", "--exclude-standard", "-z"]
        )
        return list(dict.fromkeys(name for name in (diff + untracked).split("\0") if name))

    def integrate(self) -> None:
        """Commit and merge this workspace, rolling back on any error."""
        if self.cleaned:
            raise RuntimeError("execution workspace is already cleaned")
        try:
            status = _git(self.path, ["status", "--porcelain"])
            if status.strip():
                _git(self.path, ["add", "-A"])
                _git(
                    self.path,
                    [
                        "-c",
                        "user.name=Adaptive Agent Orchestrator",
                        "-c",
                        "user.email=adaptive-agent-orchestrator@users.noreply.github.com",
                        "commit",
                        "-m",
                        f"agent({self.execution_id}): deliver {self.task_id}",
                    ],
                )
            _git(
                self.repo_path,
                [
                    "-c",
                    "user.name=Adaptive Agent Orchestrator",
                    "-c",
                    "user.email=adaptive-agent-orchestrator@users.noreply.github.com",
                    "merge",
                    "--no-ff",
                    self.branch,
                    "-m",
                    f"integrate {self.task_id}/{self.execution_id}",
                ],
            )
        except Exception:
            self.rollback()
            raise
        self.cleanup()

    def rollback(self) -> None:
        """Restore the integration base and remove all execution artifacts."""
        if self.cleaned:
            return
        with contextlib.suppress(RuntimeError):
            _git(self.repo_path, ["merge", "--abort"])
        current = _git(self.repo_path, ["rev-parse", "HEAD"]).strip()
        if current != self.base_sha:
            _git(self.repo_path, ["reset", "--merge", self.base_sha])
        self.cleanup()

    def cleanup(self) -> None:
        """Idempotently remove the worktree and its temporary branch."""
        if self.cleaned:
            return
        with contextlib.suppress(RuntimeError):
            _git(self.repo_path, ["worktree", "remove", "--force", str(self.path)])
        with contextlib.suppress(RuntimeError):
            _git(self.repo_path, ["branch", "-D", self.branch])
        self.cleaned = True


@dataclasses.dataclass
class WorkspaceExecution:
    """Shared lifecycle abstraction used by single and delegated executions."""

    workspace: ExecutionWorkspace
    read_only: bool = False

    def changed_files(self) -> list[str]:
        changed = self.workspace.changed_files()
        if self.read_only and changed:
            raise RuntimeError(f"read-only execution changed files: {changed}")
        return changed

    def finish(self, *, passed: bool) -> None:
        if passed:
            self.workspace.integrate()
        else:
            self.workspace.rollback()

    def cleanup(self) -> None:
        self.workspace.cleanup()


def _is_git_repo(repo: Path) -> bool:
    if not repo.is_dir():
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _validate_identifier(value: str, name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"unsafe {name}: {value!r}")


class WorktreeStatus(StrEnum):
    ALLOCATED  = "allocated"
    ACTIVE     = "active"
    MERGING    = "merging"
    ABANDONED  = "abandoned"
    CLEANED    = "cleaned"


@dataclasses.dataclass
class WorktreeRecord:
    task_id: str
    child_id: str
    repo_path: Path
    worktree_path: Path
    branch: str
    status: WorktreeStatus = WorktreeStatus.ALLOCATED


class WorkspaceManager:
    """
    Usage:
        wm = WorkspaceManager(repo_path="/d/myrepo")
        record = wm.allocate(task_id, child_id="child-1")
        # record.worktree_path and record.branch are injected into TaskContract
        ...
        wm.activate(task_id, "child-1")
        wm.abandon(task_id, "child-1")          # eval failed
        wm.clean(task_id, "child-1")            # after human review
    """

    def __init__(
        self,
        repo_path: str | Path,
        policy_path: str | Path = "policies/default.yaml",
    ) -> None:
        self._repo = Path(repo_path)
        raw = yaml.safe_load(Path(policy_path).read_text())
        wt_cfg = raw.get("worktree", {})
        self._base: Path = self._repo / wt_cfg.get("base_path", ".worktrees")
        self._readonly_types: set[str] = set(wt_cfg.get("readonly_task_types", []))
        self._records: dict[tuple[str, str], WorktreeRecord] = {}
        self._exclude_internal_worktrees()

    @property
    def repo_path(self) -> Path:
        return self._repo

    def _exclude_internal_worktrees(self) -> None:
        """Keep internal worktrees out of trusted root changed-file scans."""
        try:
            relative = self._base.resolve().relative_to(self._repo.resolve())
        except ValueError:
            return
        exclude_file = self._repo / ".git" / "info" / "exclude"
        if not exclude_file.parent.exists():
            return
        pattern = f"/{relative.as_posix().rstrip('/')}/"
        existing = exclude_file.read_text(errors="ignore") if exclude_file.exists() else ""
        if pattern not in existing.splitlines():
            exclude_file.write_text(existing.rstrip("\n") + f"\n{pattern}\n")

    def needs_worktree(self, task_type: str) -> bool:
        return task_type not in self._readonly_types

    def allocate(self, task_id: str, child_id: str) -> WorktreeRecord:
        """
        Create a new isolated worktree + branch for a writing child agent.
        If a stale ABANDONED record for the same (task_id, child_id) exists,
        cleans it up first so retries don't fail on duplicate branch names.
        """
        key = (task_id, child_id)
        if key in self._records:
            existing = self._records[key]
            if existing.status == WorktreeStatus.ABANDONED:
                # Stale from a previous failed run — clean up before re-creating
                _git(self._repo, ["worktree", "remove", "--force", str(existing.worktree_path)])
                with contextlib.suppress(RuntimeError):
                    _git(self._repo, ["branch", "-D", existing.branch])
                del self._records[key]
            else:
                raise ValueError(
                    f"Worktree for {task_id}/{child_id} already exists "
                    f"with status={existing.status}"
                )
        worktree_path = self._base / task_id / child_id
        branch = f"agent/{task_id}/{child_id}"

        # Create branch + worktree in a single atomic git command.
        # Do NOT `git checkout -b` first — that checks the branch out in the main
        # worktree and causes `git worktree add` to fail with "already used by worktree".
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        _git(self._repo, ["worktree", "add", "-b", branch, str(worktree_path), "HEAD"])

        record = WorktreeRecord(
            task_id=task_id,
            child_id=child_id,
            repo_path=self._repo,
            worktree_path=worktree_path,
            branch=branch,
            status=WorktreeStatus.ALLOCATED,
        )
        self._records[(task_id, child_id)] = record
        return record

    def activate(self, task_id: str, child_id: str) -> None:
        self._get(task_id, child_id).status = WorktreeStatus.ACTIVE

    def mark_merging(self, task_id: str, child_id: str) -> None:
        self._get(task_id, child_id).status = WorktreeStatus.MERGING

    def current_head(self) -> str:
        """Return the root repository HEAD used as the integration baseline."""
        return _git(self._repo, ["rev-parse", "HEAD"]).strip()

    def ensure_root_clean(self) -> None:
        """Refuse delivery when any non-ignored root change could be overwritten."""
        dirty = _git(self._repo, ["status", "--porcelain", "--untracked-files=all"])
        if dirty.strip():
            raise RuntimeError("root repository has tracked or untracked changes")

    def rollback(self, base_sha: str) -> None:
        """Atomically return the clean root to its pre-integration commit."""
        with contextlib.suppress(RuntimeError):
            _git(self._repo, ["merge", "--abort"])
        _git(self._repo, ["reset", "--merge", base_sha])

    def commit_changes(self, task_id: str, child_id: str) -> str:
        """Commit all child worktree changes and return the child branch tip."""
        record = self._get(task_id, child_id)
        status = _git(record.worktree_path, ["status", "--porcelain"])
        if status.strip():
            _git(record.worktree_path, ["add", "-A"])
            _git(
                record.worktree_path,
                [
                    "-c", "user.name=Adaptive Agent Orchestrator",
                    "-c", "user.email=adaptive-agent-orchestrator@users.noreply.github.com",
                    "commit", "-m", f"agent({record.child_id}): deliver {record.task_id}",
                ],
            )
        return _git(record.worktree_path, ["rev-parse", "HEAD"]).strip()

    def integrate(self, task_id: str, child_id: str) -> None:
        """Merge a validated child branch into the root repository."""
        record = self._get(task_id, child_id)
        record.status = WorktreeStatus.MERGING
        try:
            _git(
                self._repo,
                [
                    "-c", "user.name=Adaptive Agent Orchestrator",
                    "-c", "user.email=adaptive-agent-orchestrator@users.noreply.github.com",
                    "merge", "--no-ff", record.branch,
                    "-m", f"integrate {record.task_id}/{record.child_id}",
                ],
            )
        except RuntimeError:
            with contextlib.suppress(RuntimeError):
                _git(self._repo, ["merge", "--abort"])
            record.status = WorktreeStatus.ABANDONED
            raise

    def abandon(self, task_id: str, child_id: str) -> None:
        """Eval failed — keep worktree intact for human inspection."""
        self._get(task_id, child_id).status = WorktreeStatus.ABANDONED

    def clean(self, task_id: str, child_id: str) -> None:
        """
        Remove the worktree and branch after human has reviewed.
        Safe to call on MERGING or ABANDONED records.
        ABANDONED worktrees are NEVER auto-cleaned — this must be called explicitly.
        """
        record = self._get(task_id, child_id)
        _git(self._repo, ["worktree", "remove", "--force", str(record.worktree_path)])
        with contextlib.suppress(RuntimeError):
            _git(self._repo, ["branch", "-D", record.branch])
        record.status = WorktreeStatus.CLEANED

    def list_records(self) -> Iterator[WorktreeRecord]:
        yield from self._records.values()

    def _get(self, task_id: str, child_id: str) -> WorktreeRecord:
        key = (task_id, child_id)
        if key not in self._records:
            raise KeyError(f"No worktree record for task={task_id} child={child_id}")
        return self._records[key]


def _git(repo: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout

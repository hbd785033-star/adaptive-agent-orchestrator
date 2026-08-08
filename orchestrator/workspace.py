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

import dataclasses
import subprocess
from enum import Enum
from pathlib import Path
from typing import Iterator

import yaml


class WorktreeStatus(str, Enum):
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

    def needs_worktree(self, task_type: str) -> bool:
        return task_type not in self._readonly_types

    def allocate(self, task_id: str, child_id: str) -> WorktreeRecord:
        """
        Create the branch and worktree directory.
        Returns a WorktreeRecord whose paths can be injected into TaskContract.workspace.
        """
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
        try:
            _git(self._repo, ["branch", "-D", record.branch])
        except RuntimeError:
            pass  # branch may already be deleted
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

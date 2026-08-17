"""
WorkspaceManager — Git worktree lifecycle for repository-backed tasks.

States:
    ALLOCATED → ACTIVE → MERGING → CLEANED
                       ↘ ABANDONED          (observable cleanup failure only)

Rules:
    - Every runtime execution gets exactly one isolated worktree + branch.
    - Orchestrator allocates the path and injects it into TaskContract.workspace.
    - Failure paths never claim CLEANED without verified artifact removal.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import re
import subprocess
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

import yaml


class WorkspaceUnavailableError(RuntimeError):
    """Raised when a write task cannot obtain a safe Git workspace."""


@dataclasses.dataclass(frozen=True)
class RepositoryBaseline:
    """Frozen Git and filesystem evidence used for final read-only enforcement."""

    repo_path: Path
    excluded_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    head: str
    symbolic_head: str | None
    refs: tuple[tuple[str, str], ...]
    objects: tuple[tuple[str, str], ...]
    worktrees: tuple[tuple[str, ...], ...]
    protected: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    index_entries: str
    index_flags: str
    tracked: tuple[tuple[str, str], ...]
    untracked: tuple[tuple[str, str], ...]
    ignored: tuple[tuple[str, str], ...]

    @classmethod
    def capture(
        cls,
        repo_path: str | Path,
        *,
        excluded_paths: tuple[str, ...] = (),
        protected_paths: tuple[str | Path, ...] = (),
    ) -> RepositoryBaseline:
        repo = Path(repo_path).resolve()
        normalized_exclusions = tuple(
            item.replace("\\", "/").strip("/") for item in excluded_paths if item
        )
        normalized_protected = tuple(
            str(Path(item).resolve()) for item in protected_paths
        )

        def excluded(name: str) -> bool:
            normalized = name.replace("\\", "/")
            while normalized.startswith("./"):
                normalized = normalized[2:]
            normalized = normalized.rstrip("/")
            return any(
                normalized == prefix or normalized.startswith(f"{prefix}/")
                for prefix in normalized_exclusions
            )

        def names(args: list[str]) -> list[str]:
            return [
                name
                for name in _git(repo, args).split("\0")
                if name and not excluded(name)
            ]

        def fingerprints(paths: list[str]) -> tuple[tuple[str, str], ...]:
            return tuple(
                sorted((name, _fingerprint(repo / name)) for name in paths)
            )

        tracked_names = names(["ls-files", "-z"])
        untracked_names = names(
            ["ls-files", "--others", "--exclude-standard", "-z"]
        )
        ignored_names = names(
            ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]
        )
        return cls(
            repo_path=repo,
            excluded_paths=normalized_exclusions,
            protected_paths=normalized_protected,
            head=_git(repo, ["rev-parse", "HEAD"]).strip(),
            symbolic_head=_symbolic_head(repo),
            refs=_refs_snapshot(repo),
            objects=_object_store_snapshot(repo),
            worktrees=_worktree_snapshot(repo),
            protected=tuple(
                (path, _tree_snapshot(Path(path)))
                for path in normalized_protected
            ),
            index_entries=_git(repo, ["ls-files", "--stage", "-z"]),
            index_flags=_git(repo, ["ls-files", "-v", "-z"]),
            tracked=fingerprints(tracked_names),
            untracked=fingerprints(untracked_names),
            ignored=fingerprints(ignored_names),
        )

    def changed(self) -> list[str]:
        current = type(self).capture(
            self.repo_path,
            excluded_paths=self.excluded_paths,
            protected_paths=self.protected_paths,
        )
        changed: list[str] = []
        if current.head != self.head:
            changed.append("<HEAD>")
        if current.symbolic_head != self.symbolic_head:
            changed.append("<symbolic-HEAD>")
        if current.refs != self.refs:
            changed.append("<refs>")
        if current.objects != self.objects:
            changed.append("<objects>")
        if current.worktrees != self.worktrees:
            changed.append("<worktrees>")
        if current.protected != self.protected:
            changed.append("<internal-worktrees>")
        if current.index_entries != self.index_entries:
            changed.append("<index>")
        if current.index_flags != self.index_flags:
            changed.append("<index-flags>")
        for before, after in (
            (dict(self.tracked), dict(current.tracked)),
            (dict(self.untracked), dict(current.untracked)),
            (dict(self.ignored), dict(current.ignored)),
        ):
            changed.extend(
                name
                for name in sorted(before.keys() | after.keys())
                if before.get(name) != after.get(name)
            )
        return list(dict.fromkeys(changed))


def _fingerprint(path: Path) -> str:
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    if not path.exists():
        return "<missing>"
    if path.is_dir():
        return "<directory>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _symbolic_head(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if result.returncode == 1:
        return None
    raise RuntimeError(f"git symbolic-ref -q HEAD failed:\n{result.stderr}")


def _refs_snapshot(repo: Path) -> tuple[tuple[str, str], ...]:
    output = _git(
        repo,
        [
            "for-each-ref",
            "--sort=refname",
            "--format=%(refname)%09%(objectname)",
        ],
    )
    return tuple(
        tuple(line.split("\t", 1))
        for line in output.splitlines()
        if line
    )


def _object_store_snapshot(repo: Path) -> tuple[tuple[str, str], ...]:
    raw = _git(repo, ["rev-parse", "--git-common-dir"]).strip()
    common_dir = Path(raw)
    if not common_dir.is_absolute():
        common_dir = (repo / common_dir).resolve()
    return _tree_snapshot(common_dir / "objects", ignore_transient=True)


def _worktree_snapshot(repo: Path) -> tuple[tuple[str, ...], ...]:
    output = _git(repo, ["worktree", "list", "--porcelain"])
    records: list[tuple[str, ...]] = []
    for block in re.split(r"\n\s*\n", output.strip()):
        fields: list[str] = []
        for line in block.splitlines():
            key, separator, value = line.partition(" ")
            if key == "worktree" and separator:
                value = Path(value).resolve().as_posix()
            fields.append(f"{key} {value}".rstrip())
        if fields:
            records.append(tuple(fields))
    return tuple(sorted(records))


def _tree_snapshot(
    root: Path,
    *,
    ignore_transient: bool = False,
) -> tuple[tuple[str, str], ...]:
    if not root.exists():
        return (("<root>", "<missing>"),)
    entries: list[tuple[str, str]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories.sort()
        filenames.sort()
        current_path = Path(current)
        for name in list(directories):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                entries.append((relative, _fingerprint(path)))
                directories.remove(name)
            else:
                entries.append((relative + "/", "<directory>"))
        for name in filenames:
            if ignore_transient and (
                name.endswith(".lock") or name.startswith("tmp_obj_")
            ):
                continue
            path = current_path / name
            entries.append((path.relative_to(root).as_posix(), _fingerprint(path)))
    return tuple(entries)


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
        _assert_no_merge_state(repo)
        dirty = _git(repo, ["status", "--porcelain", "--untracked-files=all"])
        if dirty.strip():
            raise WorkspaceUnavailableError("root repository must be clean before execution")

        root = Path(base_path)
        if not root.is_absolute():
            root = repo / root
        _exclude_repo_path(repo, root)
        path = root / task_id / execution_id
        branch = f"agent/{task_id}/{execution_id}"
        base_sha = _git(repo, ["rev-parse", "HEAD"]).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _git(repo, ["worktree", "add", "-b", branch, str(path), base_sha])
        except RuntimeError as exc:
            cleanup_errors = _remove_workspace_artifacts(repo, path, branch)
            detail = str(exc)
            if cleanup_errors:
                detail += "; allocation cleanup failed: " + "; ".join(cleanup_errors)
            raise WorkspaceUnavailableError(detail) from exc
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
        """Return the trusted base-relative diff, including ignored mutations."""
        diff = _git(self.path, ["diff", "--name-only", "-z", self.base_sha])
        untracked = _git(
            self.path, ["ls-files", "--others", "--exclude-standard", "-z"]
        )
        ignored = _git(
            self.path,
            ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        )
        return list(
            dict.fromkeys(name for name in (diff + untracked + ignored).split("\0") if name)
        )

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
            self.cleanup()
        except Exception:
            self.rollback()
            raise

    def rollback(self) -> None:
        """Restore the integration base and remove all execution artifacts."""
        if self.cleaned:
            return
        with contextlib.suppress(RuntimeError):
            _git(self.repo_path, ["merge", "--abort"])
        current = _git(self.repo_path, ["rev-parse", "HEAD"]).strip()
        if current != self.base_sha:
            _git(self.repo_path, ["reset", "--merge", self.base_sha])
        _assert_no_merge_state(self.repo_path)
        dirty = _git(
            self.repo_path, ["status", "--porcelain", "--untracked-files=all"]
        )
        if dirty.strip():
            raise RuntimeError("root repository is not clean after rollback")
        self.cleanup()

    def cleanup(self) -> None:
        """Remove all artifacts and claim success only after postcondition checks."""
        if self.cleaned:
            return
        _assert_no_merge_state(self.repo_path)
        errors = _remove_workspace_artifacts(self.repo_path, self.path, self.branch)
        if errors:
            raise RuntimeError("workspace cleanup failed: " + "; ".join(errors))
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
        _exclude_repo_path(self._repo, self._base)

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
                self.clean(task_id, child_id)
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
        try:
            _git(
                self._repo,
                ["worktree", "add", "-b", branch, str(worktree_path), "HEAD"],
            )
        except RuntimeError as exc:
            cleanup_errors = _remove_workspace_artifacts(
                self._repo, worktree_path, branch
            )
            detail = str(exc)
            if cleanup_errors:
                detail += "; allocation cleanup failed: " + "; ".join(cleanup_errors)
            raise RuntimeError(detail) from exc

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
        if self.current_head() != base_sha:
            raise RuntimeError("root rollback could not restore integration base")
        _assert_no_merge_state(self._repo)
        self.ensure_root_clean()

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
        if record.status == WorktreeStatus.CLEANED:
            return
        _assert_no_merge_state(self._repo)
        errors = _remove_workspace_artifacts(
            self._repo, record.worktree_path, record.branch
        )
        if errors:
            raise RuntimeError("workspace cleanup failed: " + "; ".join(errors))
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


def _git_ref_exists(repo: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", ref,
        ], cwd=repo, capture_output=True, text=True
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(
        f"git ref probe failed ({result.returncode}) for {ref}: {result.stderr.strip()}"
    )


def _worktree_registered(repo: Path, path: Path) -> bool:
    target = path.resolve()
    output = _git(repo, ["worktree", "list", "--porcelain"])
    return any(
        Path(line.removeprefix("worktree ")).resolve() == target
        for line in output.splitlines()
        if line.startswith("worktree ")
    )


def _assert_no_merge_state(repo: Path) -> None:
    active: list[str] = []
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD"):
        raw = _git(repo, ["rev-parse", "--git-path", marker]).strip()
        marker_path = Path(raw)
        if not marker_path.is_absolute():
            marker_path = repo / marker_path
        if marker_path.exists():
            active.append(marker)
    if active:
        raise RuntimeError(f"root repository has active merge state: {', '.join(active)}")


def _exclude_repo_path(repo: Path, path: Path) -> None:
    """Exclude AAO-owned worktree storage from root status calculations."""
    try:
        relative = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return
    exclude_file = repo / ".git" / "info" / "exclude"
    if not exclude_file.parent.exists():
        return
    pattern = f"/{relative.as_posix().rstrip('/')}/"
    existing = exclude_file.read_text(errors="ignore") if exclude_file.exists() else ""
    if pattern not in existing.splitlines():
        exclude_file.write_text(existing.rstrip("\n") + f"\n{pattern}\n")


def _remove_workspace_artifacts(repo: Path, path: Path, branch: str) -> list[str]:
    """Best-effort every removal, then return all command/postcondition failures."""
    errors: list[str] = []
    if path.exists():
        try:
            _git(repo, ["worktree", "remove", "--force", str(path)])
        except RuntimeError as exc:
            errors.append(str(exc))
    try:
        branch_exists = _git_ref_exists(repo, f"refs/heads/{branch}")
    except RuntimeError as exc:
        errors.append(str(exc))
        branch_exists = False
    if branch_exists:
        try:
            _git(repo, ["branch", "-D", branch])
        except RuntimeError as exc:
            errors.append(str(exc))
    if path.exists():
        errors.append(f"worktree path still exists: {path}")
    try:
        if _worktree_registered(repo, path):
            errors.append(f"worktree still registered: {path}")
    except RuntimeError as exc:
        errors.append(str(exc))
    try:
        if _git_ref_exists(repo, f"refs/heads/{branch}"):
            errors.append(f"temporary branch still exists: {branch}")
    except RuntimeError as exc:
        errors.append(str(exc))
    return errors

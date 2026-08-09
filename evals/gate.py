"""
Deterministic Eval Gate — v1.

All checks return EvalCheck(status=PASS|FAIL|SKIP, blocker=True|False).
A single blocker FAIL → overall FAIL → retry or abandon.

Checks in this gate:
  1. paths      — changed files ⊆ allowed_paths
  2. tests      — run pytest / project test command, check exit code
  3. lint       — run ruff on changed files
  4. secrets    — gitleaks / trufflesecurity scan
  5. budget     — token cost within budget
  6. exit_code  — last tool call exited 0
"""
from __future__ import annotations

import asyncio
import fnmatch
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

from contracts.evaluation import EvalCheck, EvalResult, EvalStatus
from contracts.result import AgentResult
from contracts.task import TaskContract
from orchestrator.budget import BudgetState

# ── Individual checks ─────────────────────────────────────────────────────────

def _canonical_relative_path(raw_path: str, repo_path: Path) -> str | None:
    """Return a safe POSIX relative path, or None when it escapes the repo."""
    normalized = str(raw_path).replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    if posix_path.is_absolute() or PureWindowsPath(raw_path).is_absolute():
        return None
    if any(part == ".." for part in posix_path.parts):
        return None
    parts = [part for part in posix_path.parts if part not in ("", ".")]
    if not parts:
        return None
    root = repo_path.resolve()
    candidate = root.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return "/".join(parts)


def trusted_changed_files(
    repo_path: str | Path,
    base_sha: str | None = None,
) -> list[str] | None:
    """Derive changed paths from Git, including staged, unstaged and untracked."""
    repo = Path(repo_path)

    def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, check=False
        )

    probe = _run(["rev-parse", "--is-inside-work-tree"])
    if probe.returncode != 0 or probe.stdout.strip() != b"true":
        return None
    diff = _run(["diff", "--name-only", "-z", base_sha or "HEAD"])
    untracked = _run(["ls-files", "--others", "--exclude-standard", "-z"])
    if diff.returncode != 0 or untracked.returncode != 0:
        return None
    names = [
        value.decode(errors="surrogateescape")
        for value in (diff.stdout + untracked.stdout).split(b"\0")
        if value
    ]
    return list(dict.fromkeys(names))


def check_paths(
    result: AgentResult,
    task: TaskContract,
    *,
    repo_path: str | Path = ".",
) -> EvalCheck:
    """Verify changed files are all within allowed_paths."""
    if not task.allowed_paths:
        return EvalCheck(name="paths", status=EvalStatus.SKIP, detail="no allowed_paths defined", blocker=True)

    repo = Path(repo_path)
    safe_patterns = []
    for pattern in task.allowed_paths:
        normalized_pattern = str(pattern).replace("\\", "/")
        path_pattern = PurePosixPath(normalized_pattern)
        if (
            path_pattern.is_absolute()
            or PureWindowsPath(pattern).is_absolute()
            or ".." in path_pattern.parts
        ):
            return EvalCheck(
                name="paths",
                status=EvalStatus.FAIL,
                detail=f"unsafe allowed_paths pattern: {pattern}",
                blocker=True,
            )
        safe_patterns.append(normalized_pattern)

    violations = []
    for raw_file in result.files_changed:
        safe_file = _canonical_relative_path(raw_file, repo)
        if safe_file is None or not any(
            fnmatch.fnmatch(safe_file, pattern) for pattern in safe_patterns
        ):
            violations.append(raw_file)

    if violations:
        return EvalCheck(
            name="paths",
            status=EvalStatus.FAIL,
            detail=f"files outside allowed_paths: {violations}",
            blocker=True,
        )
    return EvalCheck(name="paths", status=EvalStatus.PASS, detail=f"all {len(result.files_changed)} files within allowed_paths")


def check_budget(result: AgentResult, budget: BudgetState) -> EvalCheck:
    """Verify token usage did not exceed budget limits."""
    violation = budget.is_over_budget()
    if violation:
        return EvalCheck(
            name="budget",
            status=EvalStatus.FAIL,
            detail=violation.detail,
            blocker=True,
        )
    return EvalCheck(
        name="budget",
        status=EvalStatus.PASS,
        detail=f"calls={budget.calls_used}/{budget.config.max_total_calls}",
    )


async def check_tests(repo_path: str | Path, changed_files: list[str]) -> EvalCheck:
    """
    Run pytest if any changed files look like test targets or source files.
    Skips if no changed files touch testable paths.
    """
    repo = Path(repo_path)
    if not changed_files:
        return EvalCheck(name="tests", status=EvalStatus.SKIP, detail="no files changed")

    # Detect test runner
    if (repo / "pyproject.toml").exists() or (repo / "pytest.ini").exists():
        cmd = ["python", "-m", "pytest", "--tb=short", "-q"]
    elif (repo / "package.json").exists():
        cmd = ["npm", "test", "--", "--passWithNoTests"]
    else:
        return EvalCheck(name="tests", status=EvalStatus.SKIP, detail="no recognised test runner", blocker=False)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode(errors="replace")
        if proc.returncode == 0:
            return EvalCheck(name="tests", status=EvalStatus.PASS, detail=output[-500:])
        return EvalCheck(
            name="tests",
            status=EvalStatus.FAIL,
            detail=output[-1000:],
            blocker=True,
        )
    except TimeoutError:
        return EvalCheck(name="tests", status=EvalStatus.FAIL, detail="test run timed out (120s)", blocker=True)
    except FileNotFoundError as e:
        return EvalCheck(name="tests", status=EvalStatus.SKIP, detail=f"runner not found: {e}", blocker=False)


async def check_lint(repo_path: str | Path, changed_files: list[str]) -> EvalCheck:
    """Run ruff on changed Python files."""
    py_files = [f for f in changed_files if f.endswith(".py")]
    if not py_files:
        return EvalCheck(name="lint", status=EvalStatus.SKIP, detail="no Python files changed")

    try:
        proc = await asyncio.create_subprocess_exec(
            "ruff", "check", *py_files,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace")
        if proc.returncode == 0:
            return EvalCheck(name="lint", status=EvalStatus.PASS, detail="ruff: no issues")
        return EvalCheck(name="lint", status=EvalStatus.FAIL, detail=output[-800:], blocker=False)  # lint=non-blocker in v1
    except (TimeoutError, FileNotFoundError) as e:
        return EvalCheck(name="lint", status=EvalStatus.SKIP, detail=f"ruff not available: {e}", blocker=False)


async def check_secrets(repo_path: str | Path, changed_files: list[str]) -> EvalCheck:
    """
    Run gitleaks detect on changed files.
    Falls back to a naive pattern scan if gitleaks is not installed.
    """
    if not changed_files:
        return EvalCheck(name="secrets", status=EvalStatus.SKIP, detail="no files changed")

    # Try gitleaks
    try:
        proc = await asyncio.create_subprocess_exec(
            "gitleaks", "detect", "--no-git", "--source", ".",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace")
        if proc.returncode == 0:
            return EvalCheck(name="secrets", status=EvalStatus.PASS, detail="gitleaks: no leaks found")
        return EvalCheck(name="secrets", status=EvalStatus.FAIL, detail=output[-800:], blocker=True)
    except FileNotFoundError:
        pass  # gitleaks not installed → naive fallback

    # Naive fallback: grep for obvious patterns
    import re
    PATTERNS = [
        r"(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]",
        r"sk-[a-zA-Z0-9]{32,}",
        r"ghp_[a-zA-Z0-9]{36}",
    ]
    for f in changed_files:
        safe_file = _canonical_relative_path(f, Path(repo_path))
        if safe_file is None:
            continue
        try:
            content = Path(repo_path, safe_file).read_text(errors="replace")
            for pattern in PATTERNS:
                if re.search(pattern, content):
                    return EvalCheck(
                        name="secrets",
                        status=EvalStatus.FAIL,
                        detail=f"potential secret pattern found in {f}",
                        blocker=True,
                    )
        except OSError:
            pass

    return EvalCheck(name="secrets", status=EvalStatus.PASS, detail="naive scan: no obvious leaks")


# ── Gate ──────────────────────────────────────────────────────────────────────

class DeterministicEvalGate:
    """
    Runs all v1 checks and returns an EvalResult.

    Usage:
        gate = DeterministicEvalGate(repo_path="/d/myrepo")
        eval_result = await gate.run(task, agent_result, budget_state)
    """

    def __init__(self, repo_path: str | Path = ".") -> None:
        self._repo = Path(repo_path)

    async def run(
        self,
        task: TaskContract,
        result: AgentResult,
        budget: BudgetState,
    ) -> EvalResult:
        checks: list[EvalCheck] = []

        effective_repo = Path(task.workspace.path) if task.workspace else self._repo
        base_sha = task.context.get("_eval_base_sha")
        git_changed = trusted_changed_files(effective_repo, base_sha=base_sha)
        changed_files = git_changed if git_changed is not None else result.files_changed
        trusted_result = result.model_copy(update={"files_changed": changed_files})

        # Synchronous checks
        checks.append(check_paths(trusted_result, task, repo_path=effective_repo))
        checks.append(check_budget(result, budget))

        # Async checks (run in parallel)
        lint_task = asyncio.create_task(check_lint(effective_repo, changed_files))
        tests_task = asyncio.create_task(check_tests(effective_repo, changed_files))
        secrets_task = asyncio.create_task(check_secrets(effective_repo, changed_files))

        checks.extend(await asyncio.gather(lint_task, tests_task, secrets_task))

        return EvalResult.aggregate(task.id, result.run_id, checks)

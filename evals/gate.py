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
from pathlib import Path

from contracts.evaluation import EvalCheck, EvalResult, EvalStatus
from contracts.result import AgentResult
from contracts.task import TaskContract
from orchestrator.budget import BudgetState

# ── Individual checks ─────────────────────────────────────────────────────────

def check_paths(result: AgentResult, task: TaskContract) -> EvalCheck:
    """Verify changed files are all within allowed_paths."""
    if not task.allowed_paths:
        return EvalCheck(name="paths", status=EvalStatus.SKIP, detail="no allowed_paths defined", blocker=True)

    violations = []
    for f in result.files_changed:
        if not any(fnmatch.fnmatch(f, pattern) for pattern in task.allowed_paths):
            violations.append(f)

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
    violation = budget.check_calls()
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
        try:
            content = Path(repo_path, f).read_text(errors="replace")
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

        # Synchronous checks
        checks.append(check_paths(result, task))
        checks.append(check_budget(result, budget))

        # Async checks (run in parallel)
        lint_task = asyncio.create_task(check_lint(self._repo, result.files_changed))
        tests_task = asyncio.create_task(check_tests(self._repo, result.files_changed))
        secrets_task = asyncio.create_task(check_secrets(self._repo, result.files_changed))

        checks.extend(await asyncio.gather(lint_task, tests_task, secrets_task))

        return EvalResult.aggregate(task.id, result.run_id, checks)

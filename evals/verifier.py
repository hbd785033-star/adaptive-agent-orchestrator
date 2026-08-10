"""Deterministic success-criterion verifier v1."""
from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from contracts.execution import CriterionResult, SuccessCriterion, TaskOutcome

RegisteredVerifier = Callable[[SuccessCriterion, Path], tuple[bool, str] | bool]


class CriterionVerifier:
    def __init__(
        self,
        repo_path: str | Path,
        registered: dict[str, RegisteredVerifier] | None = None,
    ) -> None:
        self._repo = Path(repo_path).resolve()
        self._registered = registered or {}

    def verify(
        self, criteria: Iterable[SuccessCriterion], *, completed: bool
    ) -> TaskOutcome:
        return TaskOutcome(
            completed=completed,
            criteria=[self._verify_one(criterion) for criterion in criteria],
        )

    def _verify_one(self, criterion: SuccessCriterion) -> CriterionResult:
        try:
            if criterion.type == "pytest":
                command = ["python", "-m", "pytest", "-q"]
                if criterion.target:
                    command.append(criterion.target)
                return self._run(criterion, command)
            if criterion.type == "command":
                if not criterion.command:
                    return CriterionResult(criterion=criterion, passed=False, detail="command missing")
                command = criterion.command
                if isinstance(command, str):
                    command = [command]
                return self._run(criterion, command)
            if criterion.type in {"file_exists", "file_contains"}:
                path = self._safe_path(criterion.target)
                exists = path.is_file()
                if criterion.type == "file_exists":
                    return CriterionResult(criterion=criterion, passed=exists, detail=str(path))
                contains = exists and criterion.value is not None and criterion.value in path.read_text(errors="replace")
                return CriterionResult(criterion=criterion, passed=contains, detail=str(path))
            if criterion.type == "git_diff":
                args = ["git", "diff", "--quiet", "HEAD", "--"]
                if criterion.target:
                    args.append(criterion.target)
                proc = subprocess.run(args, cwd=self._repo, capture_output=True, check=False)
                return CriterionResult(
                    criterion=criterion,
                    passed=proc.returncode == 1,
                    detail="diff present" if proc.returncode == 1 else "no diff or git error",
                )
            if criterion.type == "registered":
                callback = self._registered.get(criterion.name or "")
                if callback is None:
                    return CriterionResult(criterion=criterion, passed=False, detail="unknown verifier")
                result = callback(criterion, self._repo)
                passed, detail = result if isinstance(result, tuple) else (result, "")
                return CriterionResult(criterion=criterion, passed=passed, detail=detail)
        except (OSError, subprocess.SubprocessError) as exc:
            return CriterionResult(criterion=criterion, passed=False, detail=str(exc))
        return CriterionResult(criterion=criterion, passed=False, detail="unsupported criterion")

    def _run(self, criterion: SuccessCriterion, command: list[str]) -> CriterionResult:
        try:
            proc = subprocess.run(
                command,
                cwd=self._repo,
                capture_output=True,
                text=True,
                timeout=criterion.timeout_s,
                check=False,
            )
            output = (proc.stdout + proc.stderr).strip()[-1000:]
            return CriterionResult(criterion=criterion, passed=proc.returncode == 0, detail=output)
        except (OSError, subprocess.SubprocessError) as exc:
            return CriterionResult(criterion=criterion, passed=False, detail=str(exc))

    def _safe_path(self, raw: str | None) -> Path:
        if not raw:
            raise OSError("target missing")
        candidate = (self._repo / raw).resolve()
        candidate.relative_to(self._repo)
        return candidate

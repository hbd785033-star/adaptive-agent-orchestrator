from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from adapters.hermes.gateway import HermesAdapter
from contracts.task import RiskLevel, TaskContract, TaskType
from orchestrator.engine import Orchestrator

PROMPT = (
    "Read README.md from the permitted workspace and return exactly the first "
    "non-empty line after the Markdown H1 heading, verbatim, with no quotation "
    "marks or explanation."
)

_NESTED_CONTEXT_PREFIXES = ("HERMES_KANBAN_", "HERMES_SESSION_")


def _nested_runtime_env(source: dict[str, str] | None = None) -> dict[str, str]:
    """Copy the worker environment without dispatcher-owned nested identity."""
    environment = dict(os.environ if source is None else source)
    for name in tuple(environment):
        if name.startswith(_NESTED_CONTEXT_PREFIXES) or name == "HERMES_TENANT":
            environment.pop(name)
    return environment


def _normalized_path(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).resolve(strict=False))))


def _observed_read_path(raw_path: str, authoritative_workspace: Path) -> Path:
    """Resolve absolute paths directly and plain relative paths in the session cwd."""
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve(strict=False)
    if len(raw_path) >= 2 and raw_path[1] == ":":
        raise AssertionError(f"ambiguous Windows drive-relative path: {raw_path}")
    if raw_path.startswith(("/", "\\")):
        raise AssertionError(f"ambiguous root-relative path: {raw_path}")
    return (authoritative_workspace / path).resolve(strict=False)


def _assert_exact_workspace_read(raw_path: str, authoritative_workspace: Path) -> None:
    workspace = authoritative_workspace.resolve(strict=False)
    expected = (workspace / "README.md").resolve(strict=False)
    observed = _observed_read_path(raw_path, workspace)
    try:
        if os.path.commonpath((str(workspace), str(observed))) != str(workspace):
            raise AssertionError("observed read path is outside the authoritative workspace")
    except ValueError as exc:
        raise AssertionError("observed read path is on another drive") from exc
    assert _normalized_path(observed) == _normalized_path(expected)


def _acceptance_pass(pytest_exit_code: int, target_pass: bool) -> bool:
    """The pytest process is authoritative for live acceptance."""
    return pytest_exit_code == 0 and target_pass


def _nested_kanban_event_count(events: list[dict[str, Any]]) -> int:
    return sum(1 for event in events if event.get("event_type", "").startswith("kanban_"))


def test_nested_runtime_env_removes_only_dispatcher_identity() -> None:
    source = {
        "HERMES_HOME": "keep",
        "HERMES_PROFILE": "keep",
        "PATH": "keep",
        "HERMES_KANBAN_TASK": "outer",
        "HERMES_KANBAN_WORKSPACE": "outer-workspace",
        "HERMES_SESSION_ID": "outer-session",
        "HERMES_TENANT": "outer-tenant",
        "NORMAL_OS_SETTING": "keep",
    }

    nested = _nested_runtime_env(source)

    assert nested == {
        "HERMES_HOME": "keep",
        "HERMES_PROFILE": "keep",
        "PATH": "keep",
        "NORMAL_OS_SETTING": "keep",
    }
    assert source["HERMES_KANBAN_TASK"] == "outer"


def test_exact_workspace_read_accepts_absolute_and_plain_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "authoritative"
    workspace.mkdir()

    _assert_exact_workspace_read(str(workspace / "README.md"), workspace)
    _assert_exact_workspace_read("README.md", workspace)


def test_exact_workspace_read_rejects_wrong_workspace_and_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "authoritative"
    other = tmp_path / "other"
    workspace.mkdir()
    other.mkdir()

    with pytest.raises(AssertionError):
        _assert_exact_workspace_read(str(other / "README.md"), workspace)
    with pytest.raises(AssertionError):
        _assert_exact_workspace_read("../other/README.md", workspace)


def test_nested_kanban_events_invalidate_acceptance() -> None:
    events = [{"event_type": "kanban_complete"}]

    assert _nested_kanban_event_count(events) == 1
    assert not _acceptance_pass(0, _nested_kanban_event_count(events) == 0)


def test_failed_pytest_acceptance_gate_is_never_pass() -> None:
    assert not _acceptance_pass(1, True)
    assert not _acceptance_pass(0, False)

class ObservedHermesAdapter(HermesAdapter):
    """Real Gateway adapter with transparent RPC-request observation."""

    def __init__(self) -> None:
        super().__init__()
        self.observed_calls: list[tuple[str, dict[str, Any]]] = []

    async def _call(self, method: str, params: dict) -> Any:
        self.observed_calls.append((method, dict(params)))
        return await super()._call(method, params)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _tracked_tree_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    tracked = _git(repo, "ls-files", "-z").split("\0")
    for name in tracked:
        if not name:
            continue
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((repo / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _readme_oracle(readme: Path) -> str:
    found_h1 = False
    for line in readme.read_text(encoding="utf-8").splitlines():
        if not found_h1:
            found_h1 = line.startswith("# ")
        elif line.strip():
            return line
    raise AssertionError("README.md has no non-empty line after its Markdown H1")


def _completed_tool_payloads(db_path: Path, run_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT payload FROM telemetry_events "
            "WHERE run_id = ? AND event_type = 'event_tool_complete' "
            "ORDER BY recorded_at",
            (run_id,),
        ).fetchall()
    return [json.loads(payload) for (payload,) in rows]


def _telemetry_events(db_path: Path, run_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT event_type, payload FROM telemetry_events "
            "WHERE run_id = ? ORDER BY recorded_at",
            (run_id,),
        ).fetchall()
    return [{"event_type": event_type, **json.loads(payload)} for event_type, payload in rows]
@pytest.mark.asyncio
async def test_real_filesystem_read_traverses_hmc_aao_hermes(tmp_path: Path) -> None:
    if os.environ.get("AAO_RUN_LIVE_E2E") != "1":
        pytest.skip("LIVE E2E NOT OBSERVED: set AAO_RUN_LIVE_E2E=1 to opt in")

    repo = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "live-e2e-2.db"
    oracle = _readme_oracle(repo / "README.md")
    assert oracle == "Control plane for Hermes-backed multi-agent workflows."
    assert oracle not in PROMPT

    preflight = {
        "head": _git(repo, "rev-parse", "HEAD").strip(),
        "status": _git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "staged": _git(repo, "diff", "--cached", "--name-only"),
        "tracked_hash": _tracked_tree_hash(repo),
    }
    assert preflight["status"] == ""
    assert preflight["staged"] == ""

    runtime = ObservedHermesAdapter()
    capabilities = await runtime.capabilities()
    assert capabilities.filesystem_read is True
    assert capabilities.filesystem_write is False
    assert capabilities.shell is False
    assert capabilities.web is False

    task = TaskContract(
        task_type=TaskType.CODE_REVIEW,
        goal=PROMPT,
        allowed_paths=["README.md"],
        output_schema=[],
        risk=RiskLevel.LOW,
        complexity=1,
        subtasks=[],
    )

    async with await Orchestrator.build(
        runtime=runtime,
        repo_path=str(repo),
        db_path=str(db_path),
        policy_path=str(repo / "policies" / "default.yaml"),
    ) as orchestrator:
        result = await orchestrator.run(task)

    assert result["outcome"] == "completed", result
    assert result["planned"]["hmc"]["request_type"] == "PlannerRequest"
    assert result["planned"]["hmc"]["recommendation_type"] == "PlannerRecommendation"
    assert result["planned"]["requirements"]["filesystem_read"] is True
    assert result["planned"]["requirements"]["filesystem_write"] is False
    assert result["planned"]["runtime_selection"]["selected_runtime"] == "hermes"
    assert result["planned"]["runtime_plan"]["executor"] == "hermes"
    assert result["observed"]["runtime_adapter_invoked"] is True
    assert result["observed"]["runtime_status"] == "completed"
    assert result["observed"]["output"] == oracle
    assert result["files_changed"] == []

    session_creates = [
        params for method, params in runtime.observed_calls if method == "session.create"
    ]
    assert session_creates == [{"cwd": result["workspace_root"]}]
    assert not Path(result["workspace_root"]).exists()

    prompt_submissions = [
        params for method, params in runtime.observed_calls if method == "prompt.submit"
    ]
    assert len(prompt_submissions) == 1
    assert PROMPT in prompt_submissions[0]["text"]
    assert oracle not in prompt_submissions[0]["text"]

    tool_payloads = _completed_tool_payloads(db_path, result["observed"]["run_id"])
    read_file_payloads = [
        payload for payload in tool_payloads if payload.get("name") == "read_file"
    ]
    assert len(read_file_payloads) == 1
    _assert_exact_workspace_read(
        read_file_payloads[0]["args"]["path"], Path(result["workspace_root"])
    )
    assert read_file_payloads[0]["tool_id"]

    events = _telemetry_events(db_path, result["observed"]["run_id"])
    assert _nested_kanban_event_count(events) == 0
    assert _acceptance_pass(0, result["outcome"] == "completed")
    postflight = {
        "head": _git(repo, "rev-parse", "HEAD").strip(),
        "status": _git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "staged": _git(repo, "diff", "--cached", "--name-only"),
        "tracked_hash": _tracked_tree_hash(repo),
    }
    assert postflight == preflight

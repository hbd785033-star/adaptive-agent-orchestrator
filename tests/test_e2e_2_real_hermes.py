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
    assert read_file_payloads[0]["args"]["path"] == "README.md"
    assert read_file_payloads[0]["tool_id"]

    postflight = {
        "head": _git(repo, "rev-parse", "HEAD").strip(),
        "status": _git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "staged": _git(repo, "diff", "--cached", "--name-only"),
        "tracked_hash": _tracked_tree_hash(repo),
    }
    assert postflight == preflight

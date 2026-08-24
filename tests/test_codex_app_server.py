"""Deterministic Runtime-B contract tests for the Codex App Server adapter."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from model_council.inventory import ModelSpec

from adapters.codex.app_server import (
    CodexAppServerAdapter,
    resolve_codex_launch_command,
)
from contracts.result import RunStatus
from contracts.runtime_selection import RuntimeSelectionPolicy
from contracts.task import TaskContract, WorkspaceSpec
from orchestrator.engine import Orchestrator
from orchestrator.runtime_registry import RuntimeRegistry

_FAKE_SERVER = r'''
import json
import sys
from pathlib import Path

scenario = sys.argv[1]
log_path = Path(sys.argv[2])
thread_id = "01999999-1111-7777-8888-999999999999"
turn_id = "01999999-2222-7777-8888-999999999999"

def emit(obj):
    print(json.dumps(obj, separators=(",", ":")), flush=True)

def turn(status, text_marker="normal"):
    items = []
    if text_marker != "missing":
        text = "" if text_marker == "empty" else "OK"
        items.append({"id":"agent-1","type":"agentMessage","text":text})
    if scenario == "normal":
        items.extend([
            {
                "id":"cmd-1","type":"commandExecution","command":"python -V",
                "commandActions":[],"cwd":"C:/tmp","status":"completed",
                "exitCode":0,"durationMs":5,"aggregatedOutput":"Python"
            },
            {
                "id":"file-1","type":"fileChange","status":"completed",
                "changes":[{"path":"changed.txt","kind":{"update":None},"diff":"+x"}]
            }
        ])
    return {
        "id":turn_id,"status":status,"items":items,
        "itemsView":"summary" if scenario == "summary_view" else "full",
        "startedAt":1,"completedAt":2,"durationMs":10,"error":None
    }

received = []
for raw in sys.stdin:
    message = json.loads(raw)
    received.append(message)
    log_path.write_text(json.dumps(received), encoding="utf-8")
    method = message.get("method")
    if method == "initialize":
        emit({"id":message["id"],"result":{
            "userAgent":"fake-codex/0.146.1","platformFamily":"windows",
            "platformOs":"windows","codexHome":"C:/redacted"
        }})
    elif method == "initialized":
        pass
    elif method == "thread/start":
        emit({"id":message["id"],"result":{
            "thread":{
                "id":thread_id,"sessionId":thread_id,"cliVersion":"0.146.1",
                "createdAt":1,"updatedAt":1,"cwd":message["params"]["cwd"],
                "ephemeral":True,"modelProvider":"openai","preview":"",
                "source":"appServer","status":{"type":"idle"},"turns":[]
            },
            "model":"gpt-test","modelProvider":"openai","cwd":message["params"]["cwd"],
            "sandbox":{"type":"readOnly","networkAccess":False},
            "approvalPolicy":"never","approvalsReviewer":"user"
        }})
    elif method == "turn/start":
        emit({"id":message["id"],"result":{"turn":turn("inProgress", "missing")}})
        if scenario == "malformed":
            print("not-json", flush=True)
        elif scenario == "approval":
            emit({
                "id":900,"method":"item/commandExecution/requestApproval",
                "params":{"threadId":thread_id,"turnId":turn_id,"itemId":"cmd-1","startedAtMs":1}
            })
        elif scenario not in {"interrupt", "timeout"}:
            marker = "missing" if scenario == "missing" else "empty" if scenario == "empty" else "normal"
            status = "failed" if scenario == "failed" else "completed"
            if marker != "missing":
                emit({"method":"item/completed","params":{
                    "threadId":thread_id,"turnId":turn_id,"item":
                    {"id":"agent-1","type":"agentMessage","text":"" if marker=="empty" else "OK"}
                }})
            emit({"method":"thread/tokenUsage/updated","params":{
                "threadId":thread_id,"turnId":turn_id,
                "tokenUsage":{"last":{
                    "totalTokens":7,"inputTokens":5,"cachedInputTokens":2,
                    "cacheWriteInputTokens":0,"outputTokens":2,"reasoningOutputTokens":0
                },"total":{
                    "totalTokens":7,"inputTokens":5,"cachedInputTokens":2,
                    "cacheWriteInputTokens":0,"outputTokens":2,"reasoningOutputTokens":0
                },"modelContextWindow":100}
            }})
            emit({"method":"turn/completed","params":{
                "threadId":thread_id,"turn":turn(status, marker)
            }})
    elif method == "turn/interrupt":
        emit({"id":message["id"],"result":{}})
        emit({"method":"turn/completed","params":{
            "threadId":thread_id,"turn":turn("interrupted", "missing")
        }})
    elif message.get("id") == 900:
        emit({"method":"turn/completed","params":{
            "threadId":thread_id,"turn":turn("interrupted", "missing")
        }})
'''


def _launch(tmp_path: Path, scenario: str) -> tuple[list[str], Path]:
    script = tmp_path / "fake_codex_server.py"
    log = tmp_path / "requests.json"
    script.write_text(_FAKE_SERVER, encoding="utf-8")
    return [sys.executable, str(script), scenario, str(log)], log


def _task(tmp_path: Path) -> TaskContract:
    return TaskContract(
        id="task-codex",
        goal="Reply with exactly OK. Do not modify files.",
        workspace=WorkspaceSpec(path=str(tmp_path), branch="test"),
    )


def test_resolver_prefers_direct_node_entrypoint(tmp_path):
    root = tmp_path / "node-root"
    node = root / "node.exe"
    shim = root / "codex.cmd"
    js = root / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    js.parent.mkdir(parents=True)
    node.write_text("", encoding="utf-8")
    shim.write_text("", encoding="utf-8")
    js.write_text("", encoding="utf-8")

    assert resolve_codex_launch_command([shim]) == [str(node), str(js)]


@pytest.mark.asyncio
async def test_connect_initializes_without_model_turn_and_reports_conservative_capabilities(tmp_path):
    launch, log = _launch(tmp_path, "normal")
    adapter = CodexAppServerAdapter(launch_command=launch)

    await adapter.connect()
    capabilities = await adapter.capabilities()

    assert adapter.runtime_version == "0.146.1"
    assert adapter.initialize_evidence["platformOs"] == "windows"
    assert capabilities.streaming_events is True
    assert capabilities.cancellation is True
    assert capabilities.usage_observable is True
    assert capabilities.cost_observable is False
    assert capabilities.max_concurrent_runs == 1
    assert capabilities.filesystem_read is False
    assert capabilities.filesystem_write is False
    assert capabilities.shell is False
    assert adapter.initialized_sent is True
    await adapter.disconnect()
    methods = [entry.get("method") for entry in json.loads(log.read_text())]
    assert methods == ["initialize", "initialized"]


@pytest.mark.asyncio
async def test_real_lifecycle_uses_turn_id_and_preserves_structured_evidence(tmp_path):
    launch, log = _launch(tmp_path, "normal")
    adapter = CodexAppServerAdapter(launch_command=launch)
    await adapter.connect()

    handle = await adapter.submit(_task(tmp_path))
    events = [event async for event in adapter.events(handle)]
    result = await adapter.wait(handle)
    usage = await adapter.usage(handle)

    assert handle.run_id == "01999999-2222-7777-8888-999999999999"
    assert handle.session_id == "01999999-1111-7777-8888-999999999999"
    assert result.run_id == handle.run_id
    assert result.status == RunStatus.COMPLETED
    assert result.summary == "OK"
    assert result.model == "gpt-test"
    assert result.provider == "openai"
    assert result.runtime_version == "0.146.1"
    assert result.files_changed == ["changed.txt"]
    assert result.tool_calls is not None
    assert [item["type"] for item in result.tool_calls] == ["commandExecution", "fileChange"]
    assert result.provenance["tool_evidence_completeness"] == "partial"
    assert usage.input_tokens == 5
    assert usage.output_tokens == 2
    assert usage.cached_tokens == 2
    assert usage.estimated_cost_usd is None
    assert events[-1].type == "completed"
    requests = json.loads(log.read_text())
    thread_params = next(item["params"] for item in requests if item.get("method") == "thread/start")
    assert thread_params["cwd"] == str(tmp_path)
    assert thread_params["ephemeral"] is True
    assert thread_params["approvalPolicy"] == "never"
    assert thread_params["sandbox"] == "read-only"
    assert thread_params["dynamicTools"] == []
    assert thread_params["environments"] == []
    await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(("scenario", "expected"), [("missing", None), ("empty", "")])
async def test_output_none_and_empty_remain_distinct(tmp_path, scenario, expected):
    launch, _ = _launch(tmp_path, scenario)
    adapter = CodexAppServerAdapter(launch_command=launch)
    await adapter.connect()
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    assert result.summary == expected
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_summary_turn_view_retains_observed_agent_message(tmp_path):
    launch, _ = _launch(tmp_path, "summary_view")
    adapter = CodexAppServerAdapter(launch_command=launch)
    await adapter.connect()
    handle = await adapter.submit(_task(tmp_path))

    result = await adapter.wait(handle)

    assert result.summary == "OK"
    assert result.provenance["turn_items_view"] == "summary"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_reasoning_or_command_output_never_becomes_final_output(tmp_path):
    launch, _ = _launch(tmp_path, "missing")
    adapter = CodexAppServerAdapter(launch_command=launch)
    await adapter.connect()
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    assert result.summary is None
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_native_interrupt_maps_only_explicit_cancel_to_cancelled(tmp_path):
    launch, log = _launch(tmp_path, "interrupt")
    adapter = CodexAppServerAdapter(launch_command=launch)
    await adapter.connect()
    handle = await adapter.submit(_task(tmp_path))

    await adapter.cancel(handle)
    result = await adapter.wait(handle)

    assert result.status == RunStatus.CANCELLED
    requests = json.loads(log.read_text())
    interrupt = next(item for item in requests if item.get("method") == "turn/interrupt")
    assert interrupt["params"] == {"threadId": handle.session_id, "turnId": handle.run_id}
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_host_deadline_maps_to_timeout_not_cancelled(tmp_path):
    launch, _ = _launch(tmp_path, "timeout")
    adapter = CodexAppServerAdapter(
        launch_command=launch,
        wait_timeout=0.05,
        interrupt_grace=1.0,
    )
    await adapter.connect()
    handle = await adapter.submit(_task(tmp_path))

    result = await adapter.wait(handle)

    assert result.status == RunStatus.TIMEOUT
    assert result.provenance["native_turn_status"] == "interrupted"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_approval_request_is_denied_and_preserved(tmp_path):
    launch, log = _launch(tmp_path, "approval")
    adapter = CodexAppServerAdapter(launch_command=launch)
    await adapter.connect()
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)

    assert result.status == RunStatus.FAILED
    assert result.provenance["approval_requests"] == ["item/commandExecution/requestApproval"]
    response = next(item for item in json.loads(log.read_text()) if item.get("id") == 900 and "method" not in item)
    assert response["result"]["decision"] == "cancel"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_malformed_protocol_fails_without_false_success(tmp_path):
    launch, _ = _launch(tmp_path, "malformed")
    adapter = CodexAppServerAdapter(launch_command=launch, wait_timeout=1.0)
    await adapter.connect()
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    assert result.status == RunStatus.FAILED
    assert result.summary is None
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_orchestrator_explicitly_selects_codex_without_router_special_case(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """policy_version: routing-v1.0
routing:
  delegation:
    min_independent_subtasks: 2
    min_estimated_input_tokens: 8000
    allowed_task_types: [multi_file_refactor, parallel_research, test_and_implement]
  single:
    max_complexity: 2
    max_affected_modules: 1
  constraints:
    sequential_dependency_forces_single: true
budget:
  max_children: 2
  max_depth: 1
  max_retries: 1
  max_total_calls: 8
  require_approval_above_calls: 5
approval:
  always_require: []
  require_for_risk_levels: [3, 4]
worktree:
  base_path: .worktrees
  readonly_task_types: [parallel_research, code_review]
""",
        encoding="utf-8",
    )
    launch, _ = _launch(tmp_path, "normal")
    adapter = CodexAppServerAdapter(launch_command=launch)
    model_inventory = [
        ModelSpec(
            provider="openai",
            model="gpt-test",
            family="openai",
            is_current=True,
            reasoning=True,
            fast=True,
            healthy=True,
        )
    ]
    orchestrator = await Orchestrator.build(
        runtime_registry=RuntimeRegistry(entries=[("codex-app-server", adapter)]),
        runtime_selection_policy=RuntimeSelectionPolicy(
            policy_version="runtime-selection-test-v1",
            runtime_priority=("codex-app-server",),
            allow_degraded_fallback=False,
        ),
        model_discoverer=lambda: model_inventory,
        db_path=str(tmp_path / "orchestrator.db"),
        repo_path=str(repo),
        policy_path=str(policy_path),
    )
    try:
        result = await orchestrator.run(
            TaskContract(id="task-codex", goal="Reply with exactly OK. Do not modify files.")
        )
    finally:
        await orchestrator.close()

    assert result["outcome"] == "completed"
    assert result["observed"]["runtime_adapter"] == "codex-app-server"
    assert result["observed"]["run_id"] == "01999999-2222-7777-8888-999999999999"
    assert result["observed"]["session_id"] == "01999999-1111-7777-8888-999999999999"
    assert result["observed"]["model"] == "gpt-test"
    assert result["observed"]["provider"] == "openai"
    assert result["observed"]["output"] == "OK"
    assert result["usage"]["cached_tokens"] == 2
    assert result["files_changed"] == []

"""Deterministic contract tests for controlled Codex native Exec."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from adapters.codex.exec import CodexExecAdapter, CodexExecControl
from contracts.result import RunStatus
from contracts.task import TaskContract, WorkspaceSpec

THREAD_ID = "01999999-1111-7777-8888-999999999999"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="codex-native-exec requires Windows Job Objects"
)

_FAKE_EXEC = r'''
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

scenario, log_path, child_pid_path = sys.argv[1:4]
args = sys.argv[4:]
if args == ["--version"]:
    if scenario == "preflight_hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        Path(child_pid_path).write_text(str(child.pid), encoding="utf-8")
        Path(log_path).write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(300)
    print("codex-cli 0.149.1")
    raise SystemExit(0)

log = Path(log_path)
if scenario == "prompt_failure":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(child_pid_path).write_text(str(child.pid), encoding="utf-8")
    log.write_text(str(os.getpid()), encoding="utf-8")
    os.close(0)
    time.sleep(300)
    raise SystemExit(9)
if scenario == "prompt_hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(child_pid_path).write_text(str(child.pid), encoding="utf-8")
    log.write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(300)
    raise SystemExit(9)
prompt = sys.stdin.read()
log.write_text(json.dumps({"argv": args, "prompt": prompt}), encoding="utf-8")
thread_id = "01999999-1111-7777-8888-999999999999"
trace_thread = "different-thread" if scenario == "thread_mismatch" else thread_id
workspace = Path(args[args.index("-C") + 1]).resolve()

def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)

def make_trace():
    if scenario == "trace_missing":
        return
    root = Path(os.environ["CODEX_ROLLOUT_TRACE_ROOT"])
    bundle = root / thread_id
    payloads = bundle / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)
    metadata = {
        "thread_id": trace_thread,
        "model": "wrong-model" if scenario == "model_mismatch" else "gpt-5.6-sol",
        "provider_name": "wrong-provider" if scenario == "provider_mismatch" else "openai",
        "cwd": str(workspace.parent if scenario == "workspace_mismatch" else workspace),
        "approval_policy": "never",
        "sandbox_policy": "WorkspaceWrite { writable_roots: [], network_access: false, exclude_tmpdir_env_var: false, exclude_slash_tmp: false }",
    }
    if scenario == "oversized_metadata":
        metadata["padding"] = "x" * (300 * 1024)
    if scenario == "metadata_not_object":
        metadata = []
    metadata_path = payloads / "session.json"
    if scenario == "deep_metadata_json":
        metadata_path.write_text("[" * 1200 + "0" + "]" * 1200, encoding="utf-8")
    elif scenario != "metadata_target_missing":
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "trace_id": "trace-1",
        "rollout_id": thread_id,
        "root_thread_id": thread_id,
        "raw_event_log": "trace.jsonl",
        "payloads_dir": "payloads",
    }
    manifest_schema_values = {
        "manifest_schema_true": True,
        "manifest_schema_false": False,
        "manifest_schema_float": 1.0,
        "manifest_schema_string": "1",
    }
    if scenario in manifest_schema_values:
        manifest["schema_version"] = manifest_schema_values[scenario]
    if scenario == "oversized_manifest":
        manifest["padding"] = "x" * (80 * 1024)
    if scenario == "manifest_not_object":
        manifest = []
    if scenario == "malformed_manifest_json":
        (bundle / "manifest.json").write_text("{not-json", encoding="utf-8")
    elif scenario == "deep_manifest_json":
        (bundle / "manifest.json").write_text("[" * 1200 + "0" + "]" * 1200, encoding="utf-8")
    else:
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "payload": {
            "type": "thread_started",
            "thread_id": trace_thread,
            "metadata_payload": {
                "kind": {"type": "session_metadata"},
                "path": "payloads/session.json",
            },
        },
    }
    event_schema_values = {
        "event_schema_true": True,
        "event_schema_false": False,
        "event_schema_float": 1.0,
        "event_schema_string": "1",
    }
    if scenario in event_schema_values:
        payload["schema_version"] = event_schema_values[scenario]
    if scenario == "metadata_reference_kind_not_object":
        payload["payload"]["metadata_payload"]["kind"] = []
    if scenario == "metadata_target_missing":
        payload["payload"]["metadata_payload"]["path"] = "payloads/missing.json"
    if scenario == "metadata_drive_relative":
        payload["payload"]["metadata_payload"]["path"] = "C:payloads/session.json"
    if scenario == "metadata_root_relative":
        payload["payload"]["metadata_payload"]["path"] = r"\payloads\session.json"
    if scenario == "metadata_parent_escape":
        payload["payload"]["metadata_payload"]["path"] = "payloads/../session.json"
    if scenario == "metadata_absolute":
        payload["payload"]["metadata_payload"]["path"] = str(workspace / "session.json")
    if scenario == "metadata_unc":
        payload["payload"]["metadata_payload"]["path"] = r"\\server\share\session.json"
    if scenario == "oversized_trace_line":
        payload["padding"] = "x" * (1024 * 1024)
        (bundle / "trace.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    elif scenario == "oversized_trace":
        line = json.dumps({"schema_version": 1, "payload": {"type": "other", "padding": "x" * 1024}}) + "\n"
        (bundle / "trace.jsonl").write_text(line * (9 * 1024), encoding="utf-8")
    elif scenario == "deep_trace_json":
        (bundle / "trace.jsonl").write_text("[" * 1200 + "0" + "]" * 1200 + "\n", encoding="utf-8")
    elif scenario == "trace_event_not_object":
        (bundle / "trace.jsonl").write_text(json.dumps([]) + "\n", encoding="utf-8")
    elif scenario != "thread_started_missing":
        (bundle / "trace.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        (bundle / "trace.jsonl").write_text(json.dumps({"schema_version": 1, "payload": {"type": "other"}}) + "\n", encoding="utf-8")

if scenario in {"timeout", "cancel"}:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(child_pid_path).write_text(str(child.pid), encoding="utf-8")
    emit({"type": "thread.started", "thread_id": thread_id})
    emit({"type": "turn.started"})
    time.sleep(300)
    raise SystemExit(9)

if scenario == "surviving_descendant":
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.75)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(child_pid_path).write_text(str(child.pid), encoding="utf-8")

if scenario == "invalid_jsonl":
    print("not-json", flush=True)
    time.sleep(30)
    raise SystemExit(2)
if scenario == "oversized_jsonl":
    print("x" * (2 * 1024 * 1024), flush=True)
    time.sleep(30)
    raise SystemExit(9)

emit({"type": "thread.started", "thread_id": thread_id})
emit({"type": "turn.started"})
if scenario == "model_reroute":
    emit({"type": "item.completed", "item": {"id": "err-1", "type": "error", "message": "model rerouted: gpt-5.6-sol -> other"}})
else:
    output = "€" * 1000 if scenario == "unicode_command_budget" else "x" * 40000
    command_count = 10 if scenario == "unicode_command_budget" else (9 if scenario == "command_budget" else 1)
    for index in range(command_count):
        command_id = f"cmd-{index + 1}"
        emit({"type": "item.started", "item": {"id": command_id, "type": "command_execution", "command": "python -V", "aggregated_output": "", "exit_code": None, "status": "in_progress"}})
        emit({"type": "item.completed", "item": {"id": command_id, "type": "command_execution", "command": "python -V", "aggregated_output": output, "exit_code": 0, "status": "completed"}})
    emit({"type": "item.completed", "item": {"id": "file-1", "type": "file_change", "changes": [{"path": "fixture.py", "kind": "update"}], "status": "completed"}})
    emit({"type": "item.completed", "item": {"id": "msg-1", "type": "agent_message", "text": "done"}})
    emit({"type": "future.event", "value": 1})

make_trace()
if scenario == "turn_failed":
    emit({"type": "turn.failed", "error": {"message": "task failed"}})
    raise SystemExit(1)
emit({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 3, "cache_write_input_tokens": 2, "output_tokens": 4, "reasoning_output_tokens": 1}})
if scenario in {"trace_root_file", "trace_root_missing"}:
    root = Path(os.environ["CODEX_ROLLOUT_TRACE_ROOT"])
    shutil.rmtree(root)
    if scenario == "trace_root_file":
        root.write_text("child-owned raw trace", encoding="utf-8")
raise SystemExit(0)
'''


def _control(tmp_path: Path, **overrides) -> CodexExecControl:
    values = {
        "codex_home": tmp_path / "codex-home",
        "run_timeout_seconds": 5.0,
        "termination_grace_seconds": 0.5,
    }
    values.update(overrides)
    Path(values["codex_home"]).mkdir(exist_ok=True)
    return CodexExecControl(**values)


def _launch(tmp_path: Path, scenario: str) -> tuple[list[str], Path, Path]:
    script = tmp_path / "fake_exec.py"
    log = tmp_path / "exec-log.json"
    child_pid = tmp_path / "child.pid"
    script.write_text(_FAKE_EXEC, encoding="utf-8")
    return [sys.executable, str(script), scenario, str(log), str(child_pid)], log, child_pid


def _task(tmp_path: Path) -> TaskContract:
    return TaskContract(
        id="task-native-exec",
        goal="Change fixture.py only.",
        allowed_paths=["fixture.py"],
        workspace=WorkspaceSpec(path=str(tmp_path), branch="test"),
    )


def _large_task(tmp_path: Path) -> TaskContract:
    return TaskContract(
        id="task-native-exec-prompt-failure",
        goal="x" * (2 * 1024 * 1024),
        allowed_paths=["fixture.py"],
        workspace=WorkspaceSpec(path=str(tmp_path), branch="test"),
    )


def _pid_alive(pid: int) -> bool:
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    exit_code = ctypes.c_ulong()
    try:
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _force_kill_tree(pid: int) -> None:
    if _pid_alive(pid):
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def test_control_is_immutable_and_rejects_unsupported_values(tmp_path):
    control = _control(tmp_path)
    with pytest.raises((AttributeError, TypeError)):
        control.model = "other"  # type: ignore[misc]
    for field, value in [
        ("model", "other"),
        ("provider", "other"),
        ("reasoning", "high"),
        ("sandbox", "danger-full-access"),
        ("windows_backend", "elevated"),
        ("network_enabled", True),
        ("serena_enabled", True),
        ("web_search", "live"),
        ("ephemeral", False),
        ("trace_required", False),
        ("run_timeout_seconds", 0),
        ("termination_grace_seconds", -1),
    ]:
        with pytest.raises(ValueError, match=field):
            _control(tmp_path, **{field: value})


@pytest.mark.asyncio
async def test_connect_performs_only_version_preflight(tmp_path, monkeypatch):
    import adapters.codex.exec as exec_module

    launch, log, _ = _launch(tmp_path, "success")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    created = []
    original_create = exec_module.create_process_in_job

    def record_create(*args, **kwargs):
        owned = original_create(*args, **kwargs)
        created.append(owned)
        return owned

    monkeypatch.setattr(exec_module, "create_process_in_job", record_create)
    await adapter.connect()
    assert adapter._version == "0.149.1"
    assert len(created) == 1
    assert created[0].creation_time_membership_verified is True
    assert created[0].closed is True
    assert not log.exists()
    assert adapter._runs == {}


@pytest.mark.asyncio
async def test_version_preflight_timeout_reaps_exact_owned_tree(tmp_path):
    launch, root_pid_path, child_pid_path = _launch(tmp_path, "preflight_hang")
    adapter = CodexExecAdapter(
        control=_control(
            tmp_path,
            run_timeout_seconds=0.2,
            termination_grace_seconds=0.2,
        ),
        launch_command=launch,
        trace_base=tmp_path / "traces",
    )
    root_pid = child_pid = None
    try:
        with pytest.raises(RuntimeError, match="version preflight timed out"):
            await asyncio.wait_for(adapter.connect(), timeout=2)
        root_pid = int(root_pid_path.read_text())
        child_pid = int(child_pid_path.read_text())
        assert not _pid_alive(root_pid)
        assert not _pid_alive(child_pid)
    finally:
        if root_pid is None and root_pid_path.exists():
            root_pid = int(root_pid_path.read_text())
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
        if root_pid is not None:
            _force_kill_tree(root_pid)
        if child_pid is not None:
            _force_kill_tree(child_pid)


@pytest.mark.asyncio
async def test_success_uses_exact_argv_stdin_jsonl_trace_and_bounds(tmp_path):
    launch, log, _ = _launch(tmp_path, "success")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )

    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]
    assert state.process.creation_time_membership_verified is True
    assert state.process.pre_resume_process_ids == (state.process.pid,)
    assert handle.session_id is None
    events = [event async for event in adapter.events(handle)]
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)
    await adapter.quiesce(handle)
    assert state.process.active_processes_at_close == 0
    assert state.process.closed is True
    assert result.provenance["raw_trace_cleanup"] == "PASS"
    assert result.provenance["job_handle_cleanup"] == "PASS"
    assert result.provenance["job_active_processes_at_close"] == 0

    recorded = json.loads(log.read_text(encoding="utf-8"))
    argv = recorded["argv"]
    assert argv == [
        "exec", "--strict-config", "--ignore-user-config", "--json", "--ephemeral",
        "-m", "gpt-5.6-sol", "-s", "workspace-write", "-C", str(tmp_path.resolve()),
        "-c", 'windows.sandbox="unelevated"', "-c", 'model_provider="openai"',
        "-c", 'model_reasoning_effort="low"', "-c", "mcp_servers={}",
        "-c", "features.multi_agent=false", "-c", "sandbox_workspace_write.network_access=false",
        "-c", 'web_search="disabled"', "-",
    ]
    assert recorded["prompt"] == _task(tmp_path).prompt_preamble()
    assert handle.session_id == THREAD_ID
    assert result.status == RunStatus.COMPLETED
    assert result.model == "gpt-5.6-sol"
    assert result.provider == "openai"
    assert result.runtime_version == "0.149.1"
    assert result.summary == "done"
    assert result.usage is not None
    assert result.usage.model_dump() == {
        "input_tokens": 10, "output_tokens": 4, "cached_tokens": 3,
        "total_tokens": 14, "estimated_cost_usd": None,
    }
    assert result.provenance["controlled_experiment_provenance_complete"] is True
    assert result.provenance["observed_windows_backend"] is None
    assert result.provenance["unknown_event_types"] == ["future.event"]
    command = next(item for item in result.tool_calls if item["type"] == "command_execution")
    assert command["aggregated_output_truncated"] is True
    assert command["aggregated_output_original_bytes"] == 40000
    assert len(command["aggregated_output"]) <= 32768
    assert len(command["aggregated_output_sha256"]) == 64
    assert [event.id for event in events] == sorted(event.id for event in events)
    assert events[-1].type == "completed"
    assert not any((tmp_path / "traces").iterdir())
    projection = result.provenance["trace_projection"]
    forbidden = {"prompt", "response", "reasoning", "terminal", "command", "tool_output", "auth"}
    assert forbidden.isdisjoint(projection)


@pytest.mark.asyncio
async def test_quiesce_uses_job_not_root_returncode_as_lifecycle_authority(tmp_path):
    launch, _, _ = _launch(tmp_path, "success")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]
    result = await adapter.wait(handle)
    assert state.process.active_processes() == 0
    state.process.returncode = None

    await adapter.quiesce(handle)

    assert result.provenance["process_tree_quiescence"] == "PASS"
    assert result.provenance["job_handle_cleanup"] == "PASS"
    assert state.process.closed is True


@pytest.mark.asyncio
async def test_run_wide_command_output_allowance_cannot_be_reopened(tmp_path):
    launch, _, _ = _launch(tmp_path, "command_budget")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    commands = [call for call in result.tool_calls if call["type"] == "command_execution"]
    retained = sum(len(call["aggregated_output"].encode("utf-8")) for call in commands)
    assert len(commands) == 9
    assert retained == 256 * 1024
    assert commands[-1]["aggregated_output"] == ""
    assert commands[-1]["aggregated_output_truncated"] is True
    assert commands[-1]["aggregated_output_original_bytes"] == 40000


def test_bounded_text_never_expands_a_split_utf8_limit():
    bounded = CodexExecAdapter._bounded_text("€" * 10, 10)
    assert bounded["truncated"] is True
    assert len(bounded["value"].encode("utf-8")) <= 10


@pytest.mark.asyncio
async def test_run_wide_utf8_command_output_never_exceeds_byte_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr("adapters.codex.exec._MAX_COMMAND_OUTPUT", 1024)
    monkeypatch.setattr("adapters.codex.exec._MAX_RUN_COMMAND_OUTPUT", 4096)
    launch, _, _ = _launch(tmp_path, "unicode_command_budget")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED, result.error
    commands = [call for call in result.tool_calls if call["type"] == "command_execution"]
    retained = sum(len(call["aggregated_output"].encode("utf-8")) for call in commands)
    assert len(commands) == 10
    assert all(len(call["aggregated_output"].encode("utf-8")) <= 1024 for call in commands)
    assert retained <= 4096
    assert commands[-1]["aggregated_output"] == ""


@pytest.mark.asyncio
async def test_failed_task_preserves_strongly_bound_observed_identity(tmp_path):
    launch, _, _ = _launch(tmp_path, "turn_failed")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.FAILED
    assert result.model == "gpt-5.6-sol"
    assert result.provider == "openai"
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    assert "task_not_completed" in result.provenance["provenance_incompleteness_reasons"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        ("oversized_manifest", "manifest_too_large"),
        ("oversized_trace", "trace_too_large"),
        ("oversized_trace_line", "trace_line_too_large"),
        ("oversized_metadata", "session_metadata_too_large"),
    ],
)
async def test_oversized_raw_trace_is_incomplete_and_still_cleaned(tmp_path, scenario, reason):
    launch, _, _ = _launch(tmp_path, scenario)
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    assert reason in result.provenance["provenance_incompleteness_reasons"]
    assert not any((tmp_path / "traces").iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "manifest_not_object",
        "trace_event_not_object",
        "metadata_reference_kind_not_object",
        "metadata_not_object",
    ],
)
async def test_malformed_bounded_trace_shape_is_incomplete_and_still_cleaned(tmp_path, scenario):
    launch, _, _ = _launch(tmp_path, scenario)
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    assert result.provenance["provenance_incompleteness_reasons"]
    assert not any((tmp_path / "traces").iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "manifest_schema_true",
        "manifest_schema_false",
        "manifest_schema_float",
        "manifest_schema_string",
        "event_schema_true",
        "event_schema_false",
        "event_schema_float",
        "event_schema_string",
    ],
)
async def test_schema_version_requires_exact_integer_one(tmp_path, scenario):
    launch, _, _ = _launch(tmp_path, scenario)
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    assert "trace_schema_unsupported" in result.provenance["provenance_incompleteness_reasons"]


@pytest.mark.asyncio
async def test_trace_root_replaced_by_file_preserves_runtime_truth_and_cleans_path(tmp_path):
    launch, _, _ = _launch(tmp_path, "trace_root_file")
    trace_base = tmp_path / "traces"
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=trace_base
    )
    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]

    result = await adapter.wait(handle)
    assert result.status == RunStatus.COMPLETED
    assert result.provenance["terminal_event_observed"] == "turn.completed"
    assert result.model is None
    assert result.provider is None
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    await adapter.quiesce(handle)
    assert not state.trace_root.exists()


@pytest.mark.asyncio
async def test_trace_root_reparse_attribute_is_rejected(tmp_path, monkeypatch):
    launch, _, _ = _launch(tmp_path, "success")
    trace_base = tmp_path / "traces"
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=trace_base
    )
    original_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | stat.FILE_ATTRIBUTE_REPARSE_POINT
            )

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def mark_trace_root_reparse(path):
        observed = original_lstat(path)
        return ReparseStat(observed) if path.parent == trace_base else observed

    monkeypatch.setattr(Path, "lstat", mark_trace_root_reparse)
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert "trace_root_reparse" in result.provenance["provenance_incompleteness_reasons"]
    assert not adapter._runs[handle.run_id].trace_root.exists()


@pytest.mark.asyncio
async def test_trace_bundle_reparse_attribute_is_rejected(tmp_path, monkeypatch):
    launch, _, _ = _launch(tmp_path, "success")
    trace_base = tmp_path / "traces"
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=trace_base
    )
    original_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | stat.FILE_ATTRIBUTE_REPARSE_POINT
            )

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def mark_bundle_reparse(path):
        observed = original_lstat(path)
        is_bundle = path.name == THREAD_ID and path.parent.parent == trace_base
        return ReparseStat(observed) if is_bundle else observed

    monkeypatch.setattr(Path, "lstat", mark_bundle_reparse)
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert "trace_component_reparse" in result.provenance["provenance_incompleteness_reasons"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component", ["manifest.json", "trace.jsonl", "payloads", "session.json"]
)
async def test_trace_evidence_reparse_attribute_is_rejected(tmp_path, monkeypatch, component):
    launch, _, _ = _launch(tmp_path, "success")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    original_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | stat.FILE_ATTRIBUTE_REPARSE_POINT
            )

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def mark_file_reparse(path):
        observed = original_lstat(path)
        return ReparseStat(observed) if path.name == component else observed

    monkeypatch.setattr(Path, "lstat", mark_file_reparse)
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert "trace_component_reparse" in result.provenance["provenance_incompleteness_reasons"]


@pytest.mark.asyncio
async def test_drive_relative_metadata_reference_is_rejected(tmp_path):
    launch, _, _ = _launch(tmp_path, "metadata_drive_relative")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert "session_metadata_path_invalid" in result.provenance[
        "provenance_incompleteness_reasons"
    ]


@pytest.mark.asyncio
async def test_root_relative_metadata_reference_is_rejected_before_traversal(tmp_path):
    launch, _, _ = _launch(tmp_path, "metadata_root_relative")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert result.provenance["provenance_incompleteness_reasons"] == [
        "session_metadata_path_invalid"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["metadata_parent_escape", "metadata_absolute", "metadata_unc"]
)
async def test_metadata_reference_escape_forms_are_rejected(tmp_path, scenario):
    launch, _, _ = _launch(tmp_path, scenario)
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert result.provenance["provenance_incompleteness_reasons"] == [
        "session_metadata_path_invalid"
    ]


@pytest.mark.asyncio
async def test_cyclic_metadata_reference_is_rejected(tmp_path, monkeypatch):
    launch, _, _ = _launch(tmp_path, "success")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    original_resolve = Path.resolve

    def reject_metadata_cycle(path, *args, **kwargs):
        if path.name == "session.json":
            raise RuntimeError("Symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", reject_metadata_cycle)
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert result.provenance["provenance_incompleteness_reasons"] == [
        "session_metadata_path_invalid"
    ]


def test_production_has_no_post_spawn_assignment_or_taskkill_path():
    codex_dir = Path(__file__).parents[1] / "adapters" / "codex"
    sources = "\n".join(
        (codex_dir / name).read_text(encoding="utf-8")
        for name in ("exec.py", "_win32_job.py")
    )

    assert "AssignProcessToJobObject" not in sources
    assert "taskkill" not in sources.casefold()


def test_non_windows_job_module_keeps_importable_adapter_surface():
    script = """
import asyncio
import collections.abc
import contextlib
import ctypes
import os
import pathlib
import subprocess
import typing
source_path = pathlib.Path("adapters/codex/_win32_job.py")
source = source_path.read_text(encoding="utf-8")
os.name = "posix"
namespace = {"__name__": "non_windows_job_probe", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
assert "JobProcess" in namespace
try:
    namespace["create_process_in_job"](["unused"], cwd=".", env={})
except OSError as exc:
    assert "requires Windows Job Objects" in str(exc)
else:
    raise AssertionError("non-Windows launch did not fail closed")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_trace_root_disappears_preserves_runtime_truth_and_quiesces(tmp_path):
    launch, _, _ = _launch(tmp_path, "trace_root_missing")
    trace_base = tmp_path / "traces"
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=trace_base
    )
    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]

    result = await adapter.wait(handle)
    assert result.status == RunStatus.COMPLETED
    assert result.provenance["terminal_event_observed"] == "turn.completed"
    assert result.model is None
    assert result.provider is None
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    await adapter.quiesce(handle)
    assert not state.trace_root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        ("trace_missing", "trace_missing"),
        ("thread_started_missing", "thread_started_missing"),
        ("thread_mismatch", "thread_binding_mismatch"),
        ("model_mismatch", "model_mismatch"),
        ("provider_mismatch", "provider_mismatch"),
        ("workspace_mismatch", "cwd_mismatch"),
    ],
)
async def test_trace_failure_preserves_task_truth_but_not_observed_identity(tmp_path, scenario, reason):
    launch, _, _ = _launch(tmp_path, scenario)
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.model is None
    assert result.provider is None
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    assert reason in result.provenance["provenance_incompleteness_reasons"]
    assert "controlled experiment provenance incomplete" in result.unresolved_risks
    assert not any((tmp_path / "traces").iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "status"),
    [("model_reroute", RunStatus.FAILED), ("invalid_jsonl", RunStatus.FAILED)],
)
async def test_runtime_failures_are_terminal_without_retry(tmp_path, scenario, status):
    launch, _, _ = _launch(tmp_path, scenario)
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)
    assert result.status == status
    assert result.model is None
    assert result.provider is None


@pytest.mark.asyncio
async def test_native_jsonl_reader_applies_bound_to_pipe_read(tmp_path, monkeypatch):
    import adapters.codex.exec as exec_module

    launch, _, _ = _launch(tmp_path, "oversized_jsonl")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    await adapter.connect()
    observed_limits = []
    original_create = exec_module.create_process_in_job

    def record_read_limit(*args, **kwargs):
        owned = original_create(*args, **kwargs)
        original_readline = owned.stdout.readline

        async def bounded_readline(size=-1):
            observed_limits.append(size)
            return await original_readline(size)

        owned.stdout.readline = bounded_readline
        return owned

    monkeypatch.setattr(exec_module, "create_process_in_job", record_read_limit)
    handle = await adapter.submit(_task(tmp_path))
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)

    assert result.status == RunStatus.FAILED
    assert observed_limits
    assert set(observed_limits) == {exec_module._MAX_JSONL_LINE + 1}


@pytest.mark.asyncio
async def test_timeout_reaps_owned_root_and_child(tmp_path):
    launch, _, child_pid_path = _launch(tmp_path, "timeout")
    adapter = CodexExecAdapter(
        control=_control(tmp_path, run_timeout_seconds=0.5, termination_grace_seconds=0.2),
        launch_command=launch,
        trace_base=tmp_path / "traces",
    )
    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]
    root_pid = state.process.pid
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)
    child_pid = int(child_pid_path.read_text())
    assert result.status == RunStatus.TIMEOUT
    assert state.process.returncode is not None
    assert state.process.terminate_job_calls >= 1
    assert state.process.active_processes_at_close == 0
    assert state.process.closed is True
    assert not _pid_alive(root_pid)
    assert not _pid_alive(child_pid)


@pytest.mark.asyncio
async def test_cancel_reaps_owned_root_and_child(tmp_path):
    launch, _, child_pid_path = _launch(tmp_path, "cancel")
    adapter = CodexExecAdapter(
        control=_control(tmp_path, run_timeout_seconds=10),
        launch_command=launch,
        trace_base=tmp_path / "traces",
    )
    handle = await adapter.submit(_task(tmp_path))
    for _ in range(100):
        if child_pid_path.exists():
            break
        await asyncio.sleep(0.02)
    state = adapter._runs[handle.run_id]
    root_pid = state.process.pid
    await adapter.cancel(handle)
    result = await adapter.wait(handle)
    await adapter.quiesce(handle)
    child_pid = int(child_pid_path.read_text())
    assert result.status == RunStatus.CANCELLED
    assert state.process.terminate_job_calls >= 1
    assert state.process.active_processes_at_close == 0
    assert state.process.closed is True
    assert not _pid_alive(root_pid)
    assert not _pid_alive(child_pid)


@pytest.mark.asyncio
async def test_prompt_transport_failure_reaps_owned_root_and_child(tmp_path, monkeypatch):
    import adapters.codex.exec as exec_module

    launch, root_pid_path, child_pid_path = _launch(tmp_path, "prompt_failure")
    adapter = CodexExecAdapter(
        control=_control(tmp_path, termination_grace_seconds=0.2),
        launch_command=launch,
        trace_base=tmp_path / "traces",
    )
    await adapter.connect()
    created = []
    original_create = exec_module.create_process_in_job

    def record_create(*args, **kwargs):
        owned = original_create(*args, **kwargs)
        created.append(owned)
        return owned

    monkeypatch.setattr(exec_module, "create_process_in_job", record_create)
    root_pid = child_pid = None

    async def fail_prompt(_process, _prompt):
        for _ in range(100):
            if root_pid_path.exists() and child_pid_path.exists():
                break
            await asyncio.sleep(0.01)
        raise ConnectionError("injected prompt drain failure")

    monkeypatch.setattr(adapter, "_write_prompt", fail_prompt)
    try:
        with pytest.raises(RuntimeError, match="prompt transport failed"):
            await asyncio.wait_for(adapter.submit(_large_task(tmp_path)), timeout=3)
        root_pid = int(root_pid_path.read_text())
        child_pid = int(child_pid_path.read_text())
        assert not _pid_alive(root_pid)
        assert not _pid_alive(child_pid)
        assert not any((tmp_path / "traces").iterdir())
        assert len(created) == 1
        assert created[0].terminate_job_calls >= 1
        assert created[0].active_processes_at_close == 0
        assert created[0].closed is True
    finally:
        if root_pid is None and root_pid_path.exists():
            root_pid = int(root_pid_path.read_text())
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
        if root_pid is not None:
            _force_kill_tree(root_pid)
        if child_pid is not None:
            _force_kill_tree(child_pid)
        if created and not created[0].closed:
            created[0].terminate_job()
            await created[0].wait()
            created[0].close()


@pytest.mark.asyncio
async def test_prompt_write_timeout_terminates_exact_job(tmp_path, monkeypatch):
    import adapters.codex.exec as exec_module

    launch, root_pid_path, child_pid_path = _launch(tmp_path, "prompt_hang")
    adapter = CodexExecAdapter(
        control=_control(
            tmp_path,
            run_timeout_seconds=0.2,
            termination_grace_seconds=0.2,
        ),
        launch_command=launch,
        trace_base=tmp_path / "traces",
    )
    await adapter.connect()
    created = []
    original_create = exec_module.create_process_in_job

    def record_create(*args, **kwargs):
        owned = original_create(*args, **kwargs)
        created.append(owned)
        return owned

    monkeypatch.setattr(exec_module, "create_process_in_job", record_create)
    root_pid = child_pid = None
    try:
        with pytest.raises(RuntimeError, match="prompt write timed out"):
            await asyncio.wait_for(adapter.submit(_large_task(tmp_path)), timeout=2)
        root_pid = int(root_pid_path.read_text())
        child_pid = int(child_pid_path.read_text())
        assert len(created) == 1
        assert created[0].terminate_job_calls >= 1
        assert created[0].active_processes_at_close == 0
        assert created[0].closed is True
        assert not _pid_alive(root_pid)
        assert not _pid_alive(child_pid)
    finally:
        if created and not created[0].closed:
            created[0].terminate_job()
            await created[0].wait()
            created[0].close()
        if root_pid is None and root_pid_path.exists():
            root_pid = int(root_pid_path.read_text())
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
        if root_pid is not None:
            _force_kill_tree(root_pid)
        if child_pid is not None:
            _force_kill_tree(child_pid)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "fault"),
    [
        ("trace_missing", None),
        ("trace_root_file", None),
        ("trace_root_missing", None),
        ("success", "enumeration_disappears"),
        ("malformed_manifest_json", None),
        ("manifest_not_object", None),
        ("trace_event_not_object", None),
        ("deep_manifest_json", None),
        ("deep_trace_json", None),
        ("deep_metadata_json", None),
        ("metadata_not_object", None),
        ("metadata_target_missing", None),
        ("success", "cyclic_metadata_resolution"),
        ("oversized_manifest", None),
        ("oversized_trace", None),
        ("oversized_metadata", None),
    ],
)
async def test_provenance_failure_matrix_preserves_terminal_truth_and_cleanup(
    tmp_path, monkeypatch, scenario, fault
):
    launch, _, _ = _launch(tmp_path, scenario)
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]
    cleanup_attempts = []
    original_cleanup = adapter._remove_trace_root

    def record_cleanup(run_state):
        cleanup_attempts.append(run_state.trace_root)
        original_cleanup(run_state)

    monkeypatch.setattr(adapter, "_remove_trace_root", record_cleanup)
    if fault == "enumeration_disappears":
        original_iterdir = Path.iterdir

        def disappear_during_enumeration(path):
            if path == state.trace_root:
                shutil.rmtree(path)
                raise FileNotFoundError(path)
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", disappear_during_enumeration)
    elif fault == "cyclic_metadata_resolution":
        original_resolve = Path.resolve

        def fail_cyclic_metadata(path, *args, **kwargs):
            if path.name == "session.json" and state.trace_root in path.parents:
                raise RuntimeError("Symlink loop from child-controlled metadata reference")
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", fail_cyclic_metadata)

    result = await adapter.wait(handle)

    assert state.result is result
    assert result.status == RunStatus.COMPLETED
    assert result.provenance["terminal_event_observed"] == "turn.completed"
    assert result.usage is not None
    assert result.summary == "done"
    assert result.files_changed == ["fixture.py"]
    assert result.model is None
    assert result.provider is None
    assert result.provenance["controlled_experiment_provenance_complete"] is False
    assert result.provenance["provenance_incompleteness_reasons"]

    await adapter.quiesce(handle)
    assert cleanup_attempts == [state.trace_root]
    assert not state.trace_root.exists()
    assert not state.trace_root.is_symlink()


@pytest.mark.asyncio
async def test_trace_cleanup_failure_raises_from_quiesce(tmp_path, monkeypatch):
    launch, _, _ = _launch(tmp_path, "success")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]
    await adapter.wait(handle)
    monkeypatch.setattr("adapters.codex.exec.shutil.rmtree", lambda _path: (_ for _ in ()).throw(OSError("blocked")))
    with pytest.raises(RuntimeError, match="trace cleanup failed"):
        await adapter.quiesce(handle)
    assert state.process.active_processes_at_close == 0
    assert state.process.closed is True


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Job Objects")
@pytest.mark.asyncio
async def test_win32_prompt_writer_completes_partial_pipe_writes():
    from adapters.codex._win32_job import _AsyncPipeWriter

    class PartialStream:
        def __init__(self):
            self.received = bytearray()
            self.flushes = 0

        def write(self, data):
            count = min(3, len(data))
            self.received.extend(data[:count])
            return count

        def flush(self):
            self.flushes += 1

        def close(self):
            return None

    stream = PartialStream()
    writer = _AsyncPipeWriter(stream)
    payload = b"complete-the-entire-prompt"
    writer.write(payload)
    await writer.drain()

    assert bytes(stream.received) == payload
    assert stream.flushes == 1


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Job Objects")
@pytest.mark.asyncio
async def test_win32_close_releases_handles_when_final_accounting_query_fails(
    tmp_path, monkeypatch
):
    import adapters.codex._win32_job as job_module

    owned = job_module.create_process_in_job(
        [sys.executable, "-c", "raise SystemExit(0)"],
        cwd=tmp_path,
        env=os.environ.copy(),
    )
    original_query = job_module._query_active_processes
    try:
        owned.stdin.close()
        assert await owned.wait() == 0
        for _ in range(100):
            if owned.active_processes() == 0:
                break
            await asyncio.sleep(0.01)

        def fail_query(_job):
            raise OSError("accounting unavailable")

        monkeypatch.setattr(job_module, "_query_active_processes", fail_query)
        with pytest.raises(OSError, match="accounting unavailable"):
            owned.close()

        assert owned.closed is True
        assert owned._process_handle == 0
        assert owned._job_handle == 0
    finally:
        monkeypatch.setattr(job_module, "_query_active_processes", original_query)
        if not owned.closed:
            owned.terminate_job()
            await owned.wait()
            owned.close()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Job Objects")
@pytest.mark.asyncio
async def test_win32_launcher_confirms_exact_job_before_root_resume(tmp_path):
    from adapters.codex._win32_job import create_process_in_job

    owned = create_process_in_job(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.stdin.buffer.read(); "
                "sys.stdout.buffer.write(b'out\\n'); sys.stdout.buffer.flush(); "
                "sys.stderr.buffer.write(b'err\\n'); sys.stderr.buffer.flush()"
            ),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
    )
    try:
        assert owned.creation_time_membership_verified is True
        assert owned.kill_on_job_close_enabled is True
        assert owned.breakaway_flags_disabled is True
        assert owned.active_processes() == 1
        owned.stdin.close()
        assert await owned.stdout.readline() == b"out\n"
        assert await owned.stderr.readline() == b"err\n"
        assert await owned.wait() == 0
        for _ in range(100):
            if owned.active_processes() == 0:
                break
            await asyncio.sleep(0.01)
        assert owned.active_processes() == 0
    finally:
        owned.terminate_job()
        await owned.wait()
        owned.close()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Job Objects")
@pytest.mark.asyncio
async def test_closing_exact_job_with_kill_limit_cannot_leak_descendants(tmp_path):
    from adapters.codex._win32_job import create_process_in_job

    child_pid_path = tmp_path / "kill-on-close-child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8'); "
        "time.sleep(300)"
    )
    owned = create_process_in_job(
        [sys.executable, "-c", script, str(child_pid_path)],
        cwd=tmp_path,
        env=os.environ.copy(),
    )
    root_pid = owned.pid
    child_pid = None
    try:
        owned.stdin.close()
        for _ in range(100):
            if child_pid_path.exists():
                break
            await asyncio.sleep(0.01)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert _pid_alive(root_pid)
        assert _pid_alive(child_pid)
        assert owned.active_processes() >= 2
        assert owned.terminate_job_calls == 0

        owned.close()

        for _ in range(100):
            if not _pid_alive(root_pid) and not _pid_alive(child_pid):
                break
            await asyncio.sleep(0.01)
        assert owned.active_processes_at_close is not None
        assert owned.active_processes_at_close >= 2
        assert not _pid_alive(root_pid)
        assert not _pid_alive(child_pid)
    finally:
        if not owned.closed:
            owned.terminate_job()
            await owned.wait()
            owned.close()
        if _pid_alive(root_pid):
            _force_kill_tree(root_pid)
        if child_pid is not None and _pid_alive(child_pid):
            _force_kill_tree(child_pid)


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Job Objects")
@pytest.mark.asyncio
async def test_provenance_waits_for_surviving_job_descendant(tmp_path, monkeypatch):
    launch, _, child_pid_path = _launch(tmp_path, "surviving_descendant")
    adapter = CodexExecAdapter(
        control=_control(tmp_path), launch_command=launch, trace_base=tmp_path / "traces"
    )
    extraction_observations = []
    original_extract = adapter._extract_trace

    def observe_extraction(state):
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        extraction_observations.append(
            (state.process.active_processes(), state.process.returncode, _pid_alive(child_pid))
        )
        return original_extract(state)

    monkeypatch.setattr(adapter, "_extract_trace", observe_extraction)
    handle = await adapter.submit(_task(tmp_path))
    state = adapter._runs[handle.run_id]
    child_pid = None
    try:
        for _ in range(100):
            if child_pid_path.exists() and state.process.returncode is not None:
                break
            await asyncio.sleep(0.01)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert state.process.returncode == 0
        assert _pid_alive(child_pid)
        assert state.process.active_processes() > 0

        result = await adapter.wait(handle)

        assert result.status == RunStatus.COMPLETED
        assert extraction_observations == [(0, 0, False)]
        assert state.process.active_processes() == 0
        assert result.provenance["job_creation_time_membership"] == "PASS"
        assert result.provenance["job_active_processes_before_provenance"] == 0
        assert result.provenance["process_tree_quiescence"] == "PASS"
        await adapter.quiesce(handle)
        assert state.process.active_processes_at_close == 0
        assert state.process.closed is True
    finally:
        try:
            state.process.active_processes()
        except RuntimeError:
            pass
        else:
            state.process.terminate_job()
            await state.process.wait()
            state.process.close()
        if child_pid is not None:
            assert not _pid_alive(child_pid)

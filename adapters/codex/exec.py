"""Controlled Codex native ``exec --json`` runtime adapter.

The native JSONL stream is authoritative for lifecycle, tools, messages and usage.
The rollout trace is a run-local auxiliary source for observed experiment identity.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.codex._win32_job import JobProcess, create_process_in_job
from adapters.codex.app_server import resolve_codex_launch_command
from adapters.runtime import RuntimeCapabilities
from contracts.result import AgentEvent, AgentResult, RunHandle, RunStatus, Usage
from contracts.task import TaskContract

_MAX_JSONL_LINE = 1024 * 1024
_MAX_COMMAND = 8 * 1024
_MAX_COMMAND_OUTPUT = 32 * 1024
_MAX_RUN_COMMAND_OUTPUT = 256 * 1024
_MAX_STDERR = 64 * 1024
_MAX_TRACE_MANIFEST = 64 * 1024
_MAX_TRACE_METADATA = 256 * 1024
_MAX_TRACE_FILE = 8 * 1024 * 1024
_MAX_TRACE_LINE = 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_VERSION = "0.149.1"
_TERMINAL = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.TIMEOUT}


@dataclass(frozen=True, slots=True)
class CodexExecControl:
    """Immutable, deliberately narrow controlled-experiment profile."""

    codex_home: Path
    model: str = "gpt-5.6-sol"
    provider: str = "openai"
    reasoning: str = "low"
    sandbox: str = "workspace-write"
    windows_backend: str = "unelevated"
    network_enabled: bool = False
    serena_enabled: bool = False
    web_search: str = "disabled"
    ephemeral: bool = True
    trace_required: bool = True
    run_timeout_seconds: float = 300.0
    termination_grace_seconds: float = 2.0

    def __post_init__(self) -> None:
        expected: dict[str, Any] = {
            "model": "gpt-5.6-sol",
            "provider": "openai",
            "reasoning": "low",
            "sandbox": "workspace-write",
            "windows_backend": "unelevated",
            "network_enabled": False,
            "serena_enabled": False,
            "web_search": "disabled",
            "ephemeral": True,
            "trace_required": True,
        }
        for name, wanted in expected.items():
            if getattr(self, name) != wanted:
                raise ValueError(f"{name} must be {wanted!r} for controlled V1")
        home = Path(self.codex_home)
        if not home.is_absolute():
            raise ValueError("codex_home must be absolute")
        if self.run_timeout_seconds <= 0:
            raise ValueError("run_timeout_seconds must be positive")
        if self.termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive")
        object.__setattr__(self, "codex_home", home.resolve())


@dataclass(slots=True)
class _RunState:
    handle: RunHandle
    task: TaskContract
    workspace: Path
    trace_root: Path
    process: JobProcess
    events: list[AgentEvent] = field(default_factory=list)
    event_ready: asyncio.Condition = field(default_factory=asyncio.Condition)
    result: AgentResult | None = None
    supervisor: asyncio.Task[AgentResult] | None = None
    stderr_task: asyncio.Task[bytes] | None = None
    cancel_requested: bool = False
    quiesced: bool = False
    cleanup_error: Exception | None = None
    job_cleanup_error: Exception | None = None
    terminal_event_observed: str | None = None
    thread_id: str | None = None
    usage: Usage | None = None
    summary: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    native_files: list[str] = field(default_factory=list)
    unknown_event_types: set[str] = field(default_factory=set)
    tool_evidence_complete: bool = True
    rerouted: bool = False
    sequence: int = 0
    retained_command_output: int = 0


class CodexExecAdapter:
    """One-process-per-run controlled native Exec adapter."""

    runtime_id = "codex-native-exec"

    def __init__(
        self,
        *,
        control: CodexExecControl,
        launch_command: list[str] | None = None,
        trace_base: Path | None = None,
    ) -> None:
        self.control = control
        self._launch_command = list(launch_command or resolve_codex_launch_command())
        if not self._launch_command:
            raise ValueError("launch command must not be empty")
        self._trace_base = Path(trace_base or (Path(tempfile.gettempdir()) / "aao-codex-traces"))
        self._runs: dict[str, _RunState] = {}
        self._version: str | None = None
        self._version_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Observe availability through the exact non-model version preflight."""
        await self._preflight_version()

    async def disconnect(self) -> None:
        """Refuse to hide an active owned run during adapter shutdown."""
        active = [state for state in self._runs.values() if state.result is None]
        if active:
            raise RuntimeError("codex-native-exec has active owned runs")

    async def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            streaming_events=True,
            mid_run_steer=False,
            native_delegation=False,
            cancellation=True,
            session_resume=False,
            max_concurrent_runs=1,
            filesystem_read=True,
            filesystem_write=True,
            shell=True,
            tests=True,
            web=False,
            background_execution=False,
            persistent_tasks=False,
            human_in_loop=False,
            native_kanban=False,
            structured_output=False,
            usage_observable=True,
            cost_observable=False,
        )

    async def _preflight_version(self) -> str:
        async with self._version_lock:
            if self._version is not None:
                return self._version
            process = create_process_in_job(
                [*self._launch_command, "--version"],
                cwd=Path.cwd(),
                env=self._child_environment(trace_root=None),
            )
            stdout_task = asyncio.create_task(self._read_stream_tail(process.stdout, 8 * 1024))
            stderr_task = asyncio.create_task(self._read_stream_tail(process.stderr, _MAX_STDERR))
            try:
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self.control.run_timeout_seconds
                    )
                    await self._wait_for_job_process(
                        process, timeout=self.control.run_timeout_seconds
                    )
                except TimeoutError as exc:
                    await self._terminate_job_process(
                        process, timeout=self.control.termination_grace_seconds
                    )
                    await asyncio.gather(stdout_task, stderr_task)
                    raise RuntimeError("Codex version preflight timed out") from exc
                except asyncio.CancelledError:
                    await self._terminate_job_process(
                        process, timeout=self.control.termination_grace_seconds
                    )
                    await asyncio.gather(stdout_task, stderr_task)
                    raise
                stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
                text = stdout.decode("utf-8", errors="replace").strip()
                match = re.fullmatch(r"codex-cli\s+(\d+\.\d+\.\d+)", text)
                if process.returncode != 0 or match is None:
                    detail = stderr.decode("utf-8", errors="replace")[-1024:]
                    raise RuntimeError(f"Codex version preflight failed: {text or detail}")
                if match.group(1) != _VERSION:
                    raise RuntimeError(
                        f"unsupported Codex version {match.group(1)}; expected {_VERSION}"
                    )
                self._version = match.group(1)
                return self._version
            except TimeoutError as exc:
                raise RuntimeError("Codex version preflight cleanup timed out") from exc
            finally:
                process.close()

    def _child_environment(self, *, trace_root: Path | None) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CODEX_HOME"] = str(self.control.codex_home)
        if trace_root is not None:
            environment["CODEX_ROLLOUT_TRACE_ROOT"] = str(trace_root)
        else:
            environment.pop("CODEX_ROLLOUT_TRACE_ROOT", None)
        return environment

    def _argv(self, workspace: Path) -> list[str]:
        return [
            *self._launch_command,
            "exec",
            "--strict-config",
            "--ignore-user-config",
            "--json",
            "--ephemeral",
            "-m",
            self.control.model,
            "-s",
            self.control.sandbox,
            "-C",
            str(workspace),
            "-c",
            f'windows.sandbox="{self.control.windows_backend}"',
            "-c",
            f'model_provider="{self.control.provider}"',
            "-c",
            f'model_reasoning_effort="{self.control.reasoning}"',
            "-c",
            "mcp_servers={}",
            "-c",
            "features.multi_agent=false",
            "-c",
            "sandbox_workspace_write.network_access=false",
            "-c",
            'web_search="disabled"',
            "-",
        ]

    async def submit(self, task: TaskContract) -> RunHandle:
        if task.workspace is None or not task.workspace.path:
            raise ValueError("TaskContract.workspace is required for codex-native-exec")
        workspace = Path(task.workspace.path).resolve()
        if not workspace.is_dir():
            raise ValueError("TaskContract.workspace.path must be an existing directory")
        await self._preflight_version()
        run_id = str(uuid.uuid4())
        self._trace_base.mkdir(parents=True, exist_ok=True)
        trace_root = self._trace_base / run_id
        trace_root.mkdir(exist_ok=False)
        handle = RunHandle(run_id=run_id, task_id=task.id, session_id=None)
        process = create_process_in_job(
            self._argv(workspace),
            cwd=str(workspace),
            env=self._child_environment(trace_root=trace_root),
        )
        state = _RunState(
            handle=handle,
            task=task,
            workspace=workspace,
            trace_root=trace_root,
            process=process,
        )
        self._runs[run_id] = state
        state.stderr_task = asyncio.create_task(self._read_stderr(process))
        prompt_task = asyncio.create_task(
            self._write_prompt(process, task.prompt_preamble().encode("utf-8"))
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(prompt_task), timeout=self.control.run_timeout_seconds
            )
        except BaseException as exc:
            await self._terminate_owned_tree(state)
            if not prompt_task.done():
                with contextlib.suppress(BaseException):
                    await prompt_task
            else:
                with contextlib.suppress(BaseException):
                    prompt_task.result()
            if state.stderr_task is not None:
                with contextlib.suppress(Exception):
                    await state.stderr_task
            try:
                self._remove_trace_root(state)
            except OSError as cleanup_exc:
                state.cleanup_error = cleanup_exc
                state.quiesced = True
                raise RuntimeError("trace cleanup failed") from cleanup_exc
            finally:
                process.close()
                self._runs.pop(run_id, None)
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, TimeoutError):
                raise RuntimeError("Codex prompt write timed out") from exc
            raise RuntimeError("Codex prompt transport failed") from exc
        state.supervisor = asyncio.create_task(self._supervise(state))
        return handle

    @staticmethod
    async def _write_prompt(process: JobProcess, prompt: bytes) -> None:
        process.stdin.write(prompt)
        await process.stdin.drain()
        process.stdin.close()

    async def _read_stderr(self, process: JobProcess) -> bytes:
        return await self._read_stream_tail(process.stderr, _MAX_STDERR)

    @staticmethod
    async def _read_stream_tail(stream: asyncio.StreamReader, limit: int) -> bytes:
        retained = bytearray()
        while chunk := await stream.read(8192):
            retained.extend(chunk)
            if len(retained) > limit:
                del retained[:-limit]
        return bytes(retained)

    async def _emit(self, state: _RunState, event_type: str, payload: dict[str, Any]) -> None:
        state.sequence += 1
        event = AgentEvent(
            id=f"{state.handle.run_id}:{state.sequence:06d}",
            run_id=state.handle.run_id,
            type=event_type,
            payload=payload,
        )
        async with state.event_ready:
            state.events.append(event)
            state.event_ready.notify_all()

    @staticmethod
    def _bounded_text(text: str, limit: int) -> dict[str, Any]:
        raw = text.encode("utf-8")
        if limit <= 0:
            bounded: dict[str, Any] = {
                "value": "",
                "truncated": bool(raw),
                "original_bytes": len(raw),
            }
            if raw:
                bounded["sha256"] = hashlib.sha256(raw).hexdigest()
            return bounded
        if len(raw) <= limit:
            return {"value": text, "truncated": False, "original_bytes": len(raw)}
        half = limit // 2
        head = raw[:half].decode("utf-8", errors="ignore").encode("utf-8")
        tail = raw[-(limit - half) :].decode("utf-8", errors="ignore").encode("utf-8")
        bounded = head + tail
        return {
            "value": bounded.decode("utf-8", errors="replace"),
            "truncated": True,
            "original_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    async def _consume_stdout(self, state: _RunState) -> None:
        process = state.process
        while True:
            line = await process.stdout.readline(_MAX_JSONL_LINE + 1)
            if not line:
                return
            if len(line) > _MAX_JSONL_LINE:
                raise RuntimeError("JSONL line exceeds 1 MiB")
            try:
                native = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("invalid native Exec JSONL") from exc
            if not isinstance(native, dict) or not isinstance(native.get("type"), str):
                raise RuntimeError("native Exec JSONL event must be an object with type")
            await self._map_event(state, native)

    async def _map_event(self, state: _RunState, native: dict[str, Any]) -> None:
        kind = native["type"]
        if kind == "thread.started":
            thread_id = native.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                raise RuntimeError("thread.started is missing thread_id")
            if state.thread_id is not None and state.thread_id != thread_id:
                raise RuntimeError("conflicting native thread IDs")
            state.thread_id = thread_id
            state.handle.session_id = thread_id
            return
        if kind == "turn.started":
            return
        if kind in {"item.started", "item.updated", "item.completed"}:
            item = native.get("item")
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                raise RuntimeError("native item event is malformed")
            await self._map_item(state, kind, item)
            return
        if kind == "turn.completed":
            state.terminal_event_observed = kind
            usage = native.get("usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                state.usage = Usage(
                    input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                    output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                    cached_tokens=(
                        usage.get("cached_input_tokens")
                        if isinstance(usage.get("cached_input_tokens"), int)
                        else None
                    ),
                    total_tokens=(
                        input_tokens + output_tokens
                        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
                        else None
                    ),
                    estimated_cost_usd=None,
                )
                await self._emit(state, "usage", state.usage.model_dump())
            return
        if kind == "turn.failed":
            state.terminal_event_observed = kind
            await self._emit(state, "error", {"code": "turn_failed", "message": str(native.get("error"))})
            return
        if kind == "error":
            await self._emit(state, "error", {"code": "native_error", "message": str(native.get("message") or native)})
            return
        state.unknown_event_types.add(kind)

    async def _map_item(self, state: _RunState, event_kind: str, item: dict[str, Any]) -> None:
        item_type = item["type"]
        if item_type == "error":
            message = str(item.get("message", "native item error"))
            if "model rerouted:" in message.lower():
                state.rerouted = True
            await self._emit(state, "error", {"code": "native_item_error", "message": message})
            return
        if item_type == "agent_message" and event_kind == "item.completed":
            text = item.get("text")
            if isinstance(text, str):
                state.summary = text
                await self._emit(state, "message", {"text": text, "native_item_id": item.get("id")})
            return
        if item_type == "command_execution":
            if event_kind == "item.started":
                await self._emit(state, "tool_start", {"tool_name": item_type, "native_item_id": item.get("id")})
                return
            if event_kind != "item.completed":
                return
            command = self._bounded_text(str(item.get("command", "")), _MAX_COMMAND)
            available = max(0, _MAX_RUN_COMMAND_OUTPUT - state.retained_command_output)
            output_limit = min(_MAX_COMMAND_OUTPUT, available)
            output = self._bounded_text(str(item.get("aggregated_output", "")), output_limit)
            state.retained_command_output += len(output["value"].encode("utf-8"))
            call = {
                "type": item_type,
                "id": item.get("id"),
                "command": command["value"],
                "command_truncated": command["truncated"],
                "command_original_bytes": command["original_bytes"],
                "aggregated_output": output["value"],
                "aggregated_output_truncated": output["truncated"],
                "aggregated_output_original_bytes": output["original_bytes"],
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
            }
            if command.get("sha256"):
                call["command_sha256"] = command["sha256"]
            if output.get("sha256"):
                call["aggregated_output_sha256"] = output["sha256"]
            state.tool_calls.append(call)
            await self._emit(state, "tool_complete", call)
            return
        if item_type == "file_change":
            if event_kind == "item.started":
                await self._emit(state, "tool_start", {"tool_name": item_type, "native_item_id": item.get("id")})
                return
            if event_kind != "item.completed":
                return
            changes = item.get("changes") if isinstance(item.get("changes"), list) else []
            files = [change.get("path") for change in changes if isinstance(change, dict) and isinstance(change.get("path"), str)]
            state.native_files.extend(files)
            call = {"type": item_type, "id": item.get("id"), "changes": changes, "status": item.get("status")}
            state.tool_calls.append(call)
            await self._emit(state, "tool_complete", call)
            return
        if item_type in {"mcp_tool_call", "web_search", "collaboration"}:
            state.tool_evidence_complete = False
            await self._emit(state, "error", {"code": "disabled_tool_observed", "message": item_type})
            return
        state.tool_evidence_complete = False

    async def _supervise(self, state: _RunState) -> AgentResult:
        status = RunStatus.FAILED
        error: str | None = None
        try:
            await asyncio.wait_for(self._consume_stdout(state), timeout=self.control.run_timeout_seconds)
            return_code = await asyncio.wait_for(
                state.process.wait(), timeout=self.control.termination_grace_seconds
            )
            await self._wait_for_owned_tree(
                state, timeout=self.control.run_timeout_seconds
            )
            if state.cancel_requested:
                status = RunStatus.CANCELLED
            elif state.rerouted:
                status = RunStatus.FAILED
                error = "model reroute observed"
            elif state.terminal_event_observed == "turn.completed" and return_code == 0:
                status = RunStatus.COMPLETED
            else:
                status = RunStatus.FAILED
                error = f"native Exec terminal mismatch (event={state.terminal_event_observed}, exit={return_code})"
        except TimeoutError:
            status = RunStatus.CANCELLED if state.cancel_requested else RunStatus.TIMEOUT
            error = "cancelled" if state.cancel_requested else "native Exec hard timeout"
            await self._terminate_owned_tree(state)
        except Exception as exc:
            status = RunStatus.CANCELLED if state.cancel_requested else RunStatus.FAILED
            error = str(exc)
            await self._terminate_owned_tree(state)
        finally:
            if state.process.returncode is None or state.process.active_processes() > 0:
                await self._terminate_owned_tree(state)
            if state.stderr_task is not None:
                with contextlib.suppress(Exception):
                    await state.stderr_task

        active_before_provenance = state.process.active_processes()
        if active_before_provenance != 0:
            raise RuntimeError("provenance blocked while Job ActiveProcesses is nonzero")

        base_projection: dict[str, Any] = {
            "schema_version": None,
            "thread_binding": "FAIL",
        }
        state.result = AgentResult(
            run_id=state.handle.run_id,
            task_id=state.task.id,
            status=status,
            usage=state.usage,
            files_changed=sorted(set(state.native_files)),
            summary=state.summary,
            tool_calls=state.tool_calls,
            model=None,
            provider=None,
            runtime_version=self._version,
            provenance={
                "runtime": self.runtime_id,
                "configured_model": self.control.model,
                "configured_provider": self.control.provider,
                "configured_windows_backend": self.control.windows_backend,
                "observed_windows_backend": None,
                "controlled_experiment_provenance_complete": False,
                "provenance_incompleteness_reasons": ["trace_enrichment_pending"],
                "tool_evidence_completeness": (
                    "complete" if state.tool_evidence_complete else "partial"
                ),
                "runtime_file_evidence_completeness": "partial",
                "unknown_event_types": sorted(state.unknown_event_types),
                "trace_projection": base_projection,
                "terminal_event_observed": state.terminal_event_observed,
                "process_exit_code": state.process.returncode,
                "job_creation_time_membership": "PASS",
                "job_active_processes_before_provenance": active_before_provenance,
                "process_tree_quiescence": "PASS",
                "raw_trace_cleanup": "PENDING",
                "job_handle_cleanup": "PENDING",
                "job_active_processes_at_close": None,
            },
            unresolved_risks=["controlled experiment provenance incomplete"],
            error=error,
        )

        model, provider, projection, reasons = self._extract_trace(state)
        identity_complete = model is not None and provider is not None and not reasons
        if state.rerouted:
            model = None
            provider = None
            identity_complete = False
            if "model_rerouted" not in reasons:
                reasons.append("model_rerouted")
        if status != RunStatus.COMPLETED and "task_not_completed" not in reasons:
            reasons.append("task_not_completed")
        if not state.tool_evidence_complete and "tool_evidence_partial" not in reasons:
            reasons.append("tool_evidence_partial")
        complete = identity_complete and status == RunStatus.COMPLETED and state.tool_evidence_complete
        risks = [] if complete else ["controlled experiment provenance incomplete"]
        provenance: dict[str, Any] = {
            "runtime": self.runtime_id,
            "configured_model": self.control.model,
            "configured_provider": self.control.provider,
            "configured_windows_backend": self.control.windows_backend,
            "observed_windows_backend": None,
            "controlled_experiment_provenance_complete": complete,
            "provenance_incompleteness_reasons": reasons[:16],
            "tool_evidence_completeness": "complete" if state.tool_evidence_complete else "partial",
            "runtime_file_evidence_completeness": "partial",
            "unknown_event_types": sorted(state.unknown_event_types),
            "trace_projection": projection,
            "terminal_event_observed": state.terminal_event_observed,
            "process_exit_code": state.process.returncode,
            "job_creation_time_membership": "PASS",
            "job_active_processes_before_provenance": active_before_provenance,
            "process_tree_quiescence": "PASS",
            "raw_trace_cleanup": "PENDING",
            "job_handle_cleanup": "PENDING",
            "job_active_processes_at_close": None,
        }
        state.result.model = model
        state.result.provider = provider
        state.result.provenance = provenance
        state.result.unresolved_risks = risks
        await self._emit(
            state,
            "completed" if status == RunStatus.COMPLETED else "error",
            {"status": status.value, "error": error},
        )
        return state.result

    @staticmethod
    def _read_bounded_file(path: Path, limit: int, reason: str) -> bytes:
        if path.stat().st_size > limit:
            raise ValueError(reason)
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(reason)
        return data

    @staticmethod
    def _decode_trace_json(data: bytes, reason: str) -> Any:
        try:
            return json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(reason) from exc

    @staticmethod
    def _stat_is_reparse(observed: os.stat_result) -> bool:
        return bool(
            getattr(observed, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
        )

    @classmethod
    def _require_regular_trace_file(cls, path: Path) -> None:
        observed = path.lstat()
        if cls._stat_is_reparse(observed):
            raise ValueError("trace_component_reparse")
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError("trace_layout_invalid")

    def _extract_trace(self, state: _RunState) -> tuple[str | None, str | None, dict[str, Any], list[str]]:
        projection: dict[str, Any] = {"schema_version": None, "thread_binding": "FAIL"}
        reasons: list[str] = []
        entries: list[Path] = []
        try:
            root_stat = state.trace_root.lstat()
            if self._stat_is_reparse(root_stat):
                return None, None, projection, ["trace_root_reparse"]
            if not stat.S_ISDIR(root_stat.st_mode):
                return None, None, projection, ["trace_missing"]
            bundles: list[Path] = []
            for entry in state.trace_root.iterdir():
                entries.append(entry)
                if len(entries) > 1:
                    return None, None, projection, ["trace_layout_invalid"]
                entry_stat = entry.lstat()
                if self._stat_is_reparse(entry_stat):
                    return None, None, projection, ["trace_component_reparse"]
                if stat.S_ISDIR(entry_stat.st_mode):
                    bundles.append(entry)
            if len(entries) != 1 or len(bundles) != 1:
                return None, None, projection, ["trace_missing"]
            bundle = bundles[0]
        except FileNotFoundError:
            return None, None, projection, ["trace_missing"]
        except OSError:
            return None, None, projection, ["trace_unavailable"]
        try:
            manifest_path = bundle / "manifest.json"
            self._require_regular_trace_file(manifest_path)
            manifest_bytes = self._read_bounded_file(
                manifest_path, _MAX_TRACE_MANIFEST, "manifest_too_large"
            )
            manifest = self._decode_trace_json(manifest_bytes, "trace_manifest_invalid")
            if not isinstance(manifest, dict):
                raise ValueError("trace_manifest_invalid")
            manifest_schema = manifest.get("schema_version")
            if type(manifest_schema) is not int or manifest_schema != 1:
                raise ValueError("trace_schema_unsupported")
            if manifest.get("raw_event_log") != "trace.jsonl" or manifest.get("payloads_dir") != "payloads":
                raise ValueError("trace_layout_invalid")
            started: list[dict[str, Any]] = []
            schemas: set[int] = set()
            trace_hasher = hashlib.sha256()
            trace_size = 0
            trace_path = bundle / "trace.jsonl"
            self._require_regular_trace_file(trace_path)
            if trace_path.stat().st_size > _MAX_TRACE_FILE:
                raise ValueError("trace_too_large")
            with trace_path.open("rb") as trace_stream:
                while raw_line := trace_stream.readline(_MAX_TRACE_LINE + 1):
                    if len(raw_line) > _MAX_TRACE_LINE:
                        raise ValueError("trace_line_too_large")
                    trace_size += len(raw_line)
                    if trace_size > _MAX_TRACE_FILE:
                        raise ValueError("trace_too_large")
                    trace_hasher.update(raw_line)
                    event = self._decode_trace_json(raw_line, "trace_event_invalid")
                    if not isinstance(event, dict):
                        raise ValueError("trace_event_invalid")
                    schema_version = event.get("schema_version")
                    if type(schema_version) is not int:
                        raise ValueError("trace_schema_unsupported")
                    schemas.add(schema_version)
                    payload = event.get("payload")
                    if isinstance(payload, dict) and payload.get("type") == "thread_started":
                        started.append(payload)
            if schemas != {1}:
                raise ValueError("trace_schema_unsupported")
            if len(started) != 1:
                raise ValueError("thread_started_missing")
            trace_started = started[0]
            reference = trace_started.get("metadata_payload")
            if not isinstance(reference, dict):
                raise ValueError("session_metadata_missing")
            kind = reference.get("kind")
            if not isinstance(kind, dict) or kind.get("type") != "session_metadata":
                raise ValueError("session_metadata_missing")
            relative = reference.get("path")
            if not isinstance(relative, str):
                raise ValueError("session_metadata_missing")
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or relative_path.drive
                or relative_path.root
                or ".." in relative_path.parts
            ):
                raise ValueError("session_metadata_path_invalid")
            metadata_path = bundle
            for index, part in enumerate(relative_path.parts):
                metadata_path /= part
                observed = metadata_path.lstat()
                if self._stat_is_reparse(observed):
                    raise ValueError("trace_component_reparse")
                final_component = index == len(relative_path.parts) - 1
                if final_component:
                    if not stat.S_ISREG(observed.st_mode):
                        raise ValueError("session_metadata_path_invalid")
                elif not stat.S_ISDIR(observed.st_mode):
                    raise ValueError("session_metadata_path_invalid")
            try:
                metadata_path = metadata_path.resolve()
                resolved_bundle = bundle.resolve()
            except RuntimeError as exc:
                raise ValueError("session_metadata_path_invalid") from exc
            if os.path.commonpath([str(resolved_bundle), str(metadata_path)]) != str(resolved_bundle):
                raise ValueError("session_metadata_path_invalid")
            metadata_bytes = self._read_bounded_file(
                metadata_path, _MAX_TRACE_METADATA, "session_metadata_too_large"
            )
            metadata = self._decode_trace_json(
                metadata_bytes, "session_metadata_invalid"
            )
            if not isinstance(metadata, dict):
                raise ValueError("session_metadata_invalid")
            ids = {
                state.thread_id,
                trace_started.get("thread_id"),
                metadata.get("thread_id"),
                manifest.get("root_thread_id"),
                manifest.get("rollout_id"),
            }
            if None in ids or len(ids) != 1:
                raise ValueError("thread_binding_mismatch")
            model = metadata.get("model")
            provider = metadata.get("provider_name")
            if model != self.control.model:
                reasons.append("model_mismatch")
            if provider != self.control.provider:
                reasons.append("provider_mismatch")
            if os.path.normcase(os.path.abspath(str(metadata.get("cwd", "")))) != os.path.normcase(str(state.workspace)):
                reasons.append("cwd_mismatch")
            if metadata.get("approval_policy") != "never":
                reasons.append("approval_mismatch")
            sandbox = metadata.get("sandbox_policy")
            if not isinstance(sandbox, str) or not re.fullmatch(r"WorkspaceWrite \{.*\}", sandbox) or "network_access: false" not in sandbox:
                reasons.append("sandbox_mismatch")
            projection = {
                "schema_version": 1,
                "raw_event_schema_versions": [1],
                "thread_binding": "PASS",
                "thread_id": state.thread_id,
                "manifest_trace_id": manifest.get("trace_id"),
                "manifest_rollout_id": manifest.get("rollout_id"),
                "model": model if not reasons else None,
                "provider_name": provider if not reasons else None,
                "cwd_match": "PASS" if "cwd_mismatch" not in reasons else "FAIL",
                "approval_match": "PASS" if "approval_mismatch" not in reasons else "FAIL",
                "sandbox_match": "PASS" if "sandbox_mismatch" not in reasons else "FAIL",
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "trace_jsonl_sha256": trace_hasher.hexdigest(),
                "session_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            }
            return (model if not reasons else None, provider if not reasons else None, projection, reasons)
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reason = str(exc) if str(exc) else "trace_invalid"
            return None, None, projection, [reason]

    async def _terminate_owned_tree(self, state: _RunState) -> None:
        await self._terminate_job_process(
            state.process, timeout=self.control.termination_grace_seconds
        )

    @staticmethod
    async def _wait_for_owned_tree(state: _RunState, *, timeout: float) -> None:
        await CodexExecAdapter._wait_for_job_process(state.process, timeout=timeout)

    @staticmethod
    async def _wait_for_job_process(process: JobProcess, *, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while process.active_processes() > 0:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("run-owned Job did not become quiescent")
            await asyncio.sleep(min(0.01, remaining))

    @staticmethod
    async def _terminate_job_process(process: JobProcess, *, timeout: float) -> None:
        process.terminate_job()
        await CodexExecAdapter._wait_for_job_process(process, timeout=timeout)
        await asyncio.wait_for(process.wait(), timeout=timeout)
        if process.active_processes() != 0:
            raise RuntimeError("exact Job termination did not establish quiescence")

    @staticmethod
    def _remove_trace_root(state: _RunState) -> None:
        path = state.trace_root
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
        except FileNotFoundError:
            return
        if path.exists() or path.is_symlink():
            raise OSError("trace root still exists")

    def _state(self, handle: RunHandle) -> _RunState:
        state = self._runs.get(handle.run_id)
        if state is None or state.handle.task_id != handle.task_id:
            raise ValueError("run handle is not owned by this adapter")
        return state

    async def wait(self, handle: RunHandle) -> AgentResult:
        state = self._state(handle)
        assert state.supervisor is not None
        return await state.supervisor

    async def events(self, handle: RunHandle, *, after: str | None = None) -> AsyncIterator[AgentEvent]:
        state = self._state(handle)
        index = 0
        if after is not None:
            while index < len(state.events) and state.events[index].id <= after:
                index += 1
        while True:
            async with state.event_ready:
                while index >= len(state.events):
                    if state.result is not None:
                        return
                    await state.event_ready.wait()
                event = state.events[index]
                index += 1
            yield event
            if event.type in {"completed", "error"} and state.result is not None:
                return

    async def usage(self, handle: RunHandle) -> Usage:
        result = await self.wait(handle)
        return result.usage or Usage()

    async def cancel(self, handle: RunHandle) -> None:
        state = self._state(handle)
        if state.result is not None:
            return
        state.cancel_requested = True
        await self._terminate_owned_tree(state)
        if state.supervisor is not None:
            await state.supervisor

    async def quiesce(self, handle: RunHandle) -> None:
        state = self._state(handle)
        if state.quiesced:
            if state.cleanup_error is not None:
                raise RuntimeError("trace cleanup failed") from state.cleanup_error
            if state.job_cleanup_error is not None:
                raise RuntimeError("Job handle cleanup failed") from state.job_cleanup_error
            return
        await self.wait(handle)
        if state.process.active_processes() != 0:
            raise RuntimeError("native Exec Job is not quiescent")
        try:
            self._remove_trace_root(state)
        except OSError as exc:
            state.cleanup_error = exc
        try:
            state.process.close()
            if (
                not state.process.closed
                or state.process.active_processes_at_close != 0
            ):
                raise RuntimeError("Job handle close lacked zero-active-process proof")
        except Exception as exc:
            state.job_cleanup_error = exc
        state.quiesced = True
        assert state.result is not None
        state.result.provenance["raw_trace_cleanup"] = (
            "FAIL" if state.cleanup_error is not None else "PASS"
        )
        state.result.provenance["job_handle_cleanup"] = (
            "FAIL" if state.job_cleanup_error is not None else "PASS"
        )
        state.result.provenance["job_active_processes_at_close"] = (
            state.process.active_processes_at_close
        )
        if state.cleanup_error is not None:
            raise RuntimeError("trace cleanup failed") from state.cleanup_error
        if state.job_cleanup_error is not None:
            raise RuntimeError("Job handle cleanup failed") from state.job_cleanup_error

    async def steer(self, handle: RunHandle, instruction: str) -> None:
        raise NotImplementedError("codex-native-exec does not support steering")

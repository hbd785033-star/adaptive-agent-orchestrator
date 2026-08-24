"""Generic result/export contracts required by Codex Runtime-B."""
from __future__ import annotations

from types import SimpleNamespace

from aao_cli.main import _build_execution_record, build_runtime_entry
from adapters.codex.app_server import CodexAppServerAdapter
from adapters.hermes.gateway import HermesAdapter
from contracts.result import AgentResult, RunHandle, RunStatus, Usage
from orchestrator.engine import Orchestrator, _runtime_terminal_outcome


def test_result_contract_preserves_timeout_cached_tokens_and_nullable_evidence():
    usage = Usage(input_tokens=0, output_tokens=0, cached_tokens=0, estimated_cost_usd=None)
    result = AgentResult(
        run_id=None,
        task_id="t",
        status=RunStatus.TIMEOUT,
        usage=usage,
        files_changed=None,
        summary=None,
        tool_calls=None,
        model=None,
        provider=None,
        runtime_version="0.146.1",
        provenance={"thread_id": "thread"},
    )

    assert result.status == RunStatus.TIMEOUT
    assert result.usage.cached_tokens == 0
    assert result.files_changed is None
    assert result.summary is None
    assert result.tool_calls is None
    assert result.model is None
    assert result.provider is None


def test_generic_terminal_outcome_preserves_all_four_execution_statuses():
    assert _runtime_terminal_outcome(RunStatus.COMPLETED) == "completed"
    assert _runtime_terminal_outcome(RunStatus.FAILED) == "failed"
    assert _runtime_terminal_outcome(RunStatus.CANCELLED) == "cancelled"
    assert _runtime_terminal_outcome(RunStatus.TIMEOUT) == "timeout"


def test_summary_preserves_runtime_result_provenance_without_identity_inference():
    record = SimpleNamespace(
        task=SimpleNamespace(id="task"), run_id="turn", route="single", retry_count=0
    )
    handle = RunHandle(run_id="turn", task_id="task", session_id="thread")
    usage = Usage(input_tokens=5, output_tokens=2, cached_tokens=1)
    runtime_result = AgentResult(
        run_id="turn",
        task_id="task",
        status=RunStatus.COMPLETED,
        usage=usage,
        files_changed=None,
        summary="OK",
        tool_calls=[],
        model="gpt-real",
        provider="openai",
        runtime_version="0.146.1",
        provenance={"tool_evidence_completeness": "partial"},
    )

    summary = Orchestrator._summary(
        record,
        "completed",
        "",
        agent_result=runtime_result,
        run_handle=handle,
        observed_runtime="codex-app-server",
        runtime_adapter_invoked=True,
        observed_events=["completed"],
        usage=usage,
    )

    assert summary["tool_calls"] == []
    assert summary["observed"]["runtime_adapter"] == "codex-app-server"
    assert summary["observed"]["runtime_version"] == "0.146.1"
    assert summary["observed"]["session_id"] == "thread"
    assert summary["observed"]["model"] == "gpt-real"
    assert summary["observed"]["provider"] == "openai"
    assert summary["observed"]["output"] == "OK"
    assert summary["usage"]["cached_tokens"] == 1
    assert summary["observed"]["provenance"]["tool_evidence_completeness"] == "partial"


def test_execution_record_exports_real_codex_identity_and_unknown_cost():
    result = {
        "outcome": "completed",
        "run_id": "turn-real",
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
            "cached_tokens": 1,
            "estimated_cost_usd": None,
        },
        "tool_calls": [],
        "files_changed": [],
        "observed": {
            "runtime_adapter": "codex-app-server",
            "runtime_adapter_invoked": True,
            "runtime_version": "0.146.1",
            "run_id": "turn-real",
            "session_id": "thread-real",
            "runtime_status": "completed",
            "model": "gpt-real",
            "provider": "openai",
            "output": "OK",
            "events": ["completed"],
            "provenance": {"tool_evidence_completeness": "partial"},
        },
    }

    record = _build_execution_record(
        task_id="task",
        result=result,
        mock=False,
        started_at="2026-08-21T00:00:00Z",
        finished_at="2026-08-21T00:00:01Z",
    )

    assert record.schema_version == "0.1"
    assert record.run_id == "turn-real"
    assert record.model == "gpt-real"
    assert record.provider == "openai"
    assert record.output == "OK"
    assert record.cached_tokens == 1
    assert record.cost_usd is None
    assert record.tool_calls == []
    assert record.metadata["observed"]["runtime_adapter"] == "codex-app-server"
    assert record.metadata["observed"]["session_id"] == "thread-real"


def test_mock_execution_record_behavior_remains_unchanged():
    record = _build_execution_record(
        task_id="task",
        result={"outcome": "completed", "observed": None},
        mock=True,
        started_at="2026-08-21T00:00:00Z",
        finished_at="2026-08-21T00:00:01Z",
    )
    assert record.model == "mock"
    assert record.provider == "fixture"


def test_bounded_runtime_factory_knows_adapters_but_not_selection_policy():
    hermes_identity, hermes = build_runtime_entry(
        "hermes", hermes_url="ws://localhost:4999", hermes_key=None
    )
    codex_identity, codex = build_runtime_entry(
        "codex-app-server", hermes_url="ws://localhost:4999", hermes_key=None
    )

    assert hermes_identity == "hermes"
    assert isinstance(hermes, HermesAdapter)
    assert codex_identity == "codex-app-server"
    assert isinstance(codex, CodexAppServerAdapter)

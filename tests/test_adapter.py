"""Tests for MockHermesAdapter and state machine."""
from __future__ import annotations

import asyncio
import json

import pytest

from adapters.hermes.gateway import HermesAdapter, _build_ws_url, _parse_event
from adapters.mock import MockHermesAdapter
from adapters.runtime import AgentRuntime
from contracts.result import RunHandle, RunStatus
from contracts.task import TaskContract, WorkspaceSpec


def make_task(**kwargs) -> TaskContract:
    return TaskContract(goal="test task", **kwargs)


def test_build_ws_url_uses_local_hermes_path_and_encodes_token() -> None:
    url = _build_ws_url("ws://127.0.0.1:4999", "secret token/+?")

    assert url.startswith("ws://127.0.0.1:4999/api/ws?")
    assert url.count("token=") == 1
    assert "secret%20token" in url or "secret+token" in url
    assert "Bearer" not in url


def test_build_ws_url_preserves_query_and_does_not_duplicate_explicit_path() -> None:
    url = _build_ws_url("ws://127.0.0.1:4999/api/ws/?existing=1", "fresh")

    assert url.startswith("ws://127.0.0.1:4999/api/ws?")
    assert "/api/ws/api/ws" not in url
    assert "existing=1" in url
    assert url.count("token=") == 1


@pytest.mark.asyncio
async def test_connect_uses_query_token_without_authorization_header(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class EmptyWebSocket:
        def __aiter__(self):
            async def messages():
                if False:
                    yield None

            return messages()

        async def close(self) -> None:
            return None

    async def fake_connect(url, **kwargs):  # noqa: ANN001, ANN003
        captured["url"] = url
        captured["kwargs"] = kwargs
        return EmptyWebSocket()

    monkeypatch.setattr("adapters.hermes.gateway.websockets.connect", fake_connect)
    secret = "unlogged-secret"
    adapter = HermesAdapter("ws://127.0.0.1:4999/api/ws?existing=1", api_key=secret)
    await adapter.connect()
    await adapter.disconnect()

    url = str(captured["url"])
    assert "/api/ws/api/ws" not in url
    assert url.count("token=") == 1
    assert secret not in repr(adapter)
    assert captured["kwargs"] == {"additional_headers": {}}


class FakeGatewayAdapter(HermesAdapter):
    """No-network Gateway contract harness."""

    def __init__(self):
        super().__init__()
        self.calls = []

    async def _call(self, method, params):
        self.calls.append((method, params))
        if method == "session.create":
            return {"session_id": "session-1"}
        if method == "prompt.submit":
            return {"run_id": "run-1"}
        if method == "session.subscribe":
            queue = self._event_queues[params["run_id"]]
            await queue.put({
                "id": "usage-1", "run_id": params["run_id"], "type": "usage",
                "payload": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            })
            await queue.put({
                "id": "done-1", "run_id": params["run_id"], "type": "completed",
                "payload": {"summary": "done", "files_changed": ["src/a.py"]},
            })
            return {}
        if method == "session.usage":
            return {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return {}


class InventoryGatewayAdapter(HermesAdapter):
    """No-network harness for authenticated model inventory retrieval."""

    def __init__(self, response=None, error: Exception | None = None):
        super().__init__()
        self.calls = []
        self.response = response
        self.error = error

    async def _call(self, method, params):
        self.calls.append((method, params))
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_model_inventory_payload_uses_picker_rpc_without_task_submission() -> None:
    payload = {
        "provider": "openai",
        "model": "gpt-test",
        "providers": [{"slug": "openai", "models": ["gpt-test"]}],
    }
    adapter = InventoryGatewayAdapter(payload)

    result = await adapter.model_inventory_payload()

    assert result is payload
    assert adapter.calls == [("model.options", {"refresh": False})]
    assert not any(
        method in {"session.create", "prompt.submit"}
        for method, _ in adapter.calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [None, [], "invalid", {}, {"providers": None}, {"providers": {}}],
)
async def test_model_inventory_payload_rejects_malformed_responses(payload) -> None:
    adapter = InventoryGatewayAdapter(payload)

    with pytest.raises(RuntimeError, match="invalid model inventory response"):
        await adapter.model_inventory_payload()

    assert adapter.calls == [("model.options", {"refresh": False})]


@pytest.mark.asyncio
async def test_model_inventory_payload_propagates_rpc_failure_without_submission() -> None:
    adapter = InventoryGatewayAdapter(error=RuntimeError("inventory unavailable"))

    with pytest.raises(RuntimeError, match="inventory unavailable"):
        await adapter.model_inventory_payload()

    assert adapter.calls == [("model.options", {"refresh": False})]
    assert not any(
        method in {"session.create", "prompt.submit"}
        for method, _ in adapter.calls
    )


class StreamingGatewayAdapter(HermesAdapter):
    """Deterministic harness for the installed TUI streaming contract."""

    def __init__(self, events: list[dict], *, timeout: float = 1.0):
        super().__init__(wait_timeout=timeout)
        self.calls = []
        self._stream_events = events

    async def _call(self, method, params):
        self.calls.append((method, params))
        if method == "session.create":
            return {"session_id": "session-stream"}
        if method == "prompt.submit":
            queue = self._event_queues["tui:session-stream"]
            for event in self._stream_events:
                await queue.put(event)
            return {"status": "streaming"}
        if method == "session.usage":
            return {"input": 4, "output": 6, "total": 10}
        raise AssertionError(f"unexpected RPC: {method}")


@pytest.mark.asyncio
async def test_streaming_ack_is_accepted_without_native_run_id() -> None:
    adapter = StreamingGatewayAdapter([
        {"run_id": "tui:session-stream", "type": "message", "payload": {"text": "partial"}},
        {"run_id": "tui:session-stream", "type": "completed", "payload": {"summary": "AAO_E2E_1_OK"}},
    ])

    handle = await adapter.submit(make_task())
    result = await adapter.wait(handle)

    assert handle.run_id == "tui:session-stream"
    assert handle.session_id == "session-stream"
    assert result.run_id == "tui:session-stream"
    assert result.summary == "AAO_E2E_1_OK"
    assert not any(method == "session.subscribe" for method, _ in adapter.calls)


@pytest.mark.asyncio
async def test_streaming_events_require_matching_session_and_complete_canonically() -> None:
    class EventWebSocket:
        def __aiter__(self):
            async def messages():
                yield json.dumps({
                    "method": "event",
                    "params": {
                        "type": "message.complete",
                        "session_id": "other-session",
                        "payload": {"text": "wrong"},
                    },
                })
                yield json.dumps({
                    "method": "event",
                    "params": {
                        "type": "message.delta",
                        "session_id": "session-stream",
                        "payload": {"text": "partial"},
                    },
                })
                yield json.dumps({
                    "method": "event",
                    "params": {
                        "type": "message.complete",
                        "session_id": "session-stream",
                        "payload": {"text": "canonical"},
                    },
                })

            return messages()

    adapter = HermesAdapter()
    handle = RunHandle(run_id="tui:session-stream", task_id="task-stream", session_id="session-stream")
    adapter._handles[handle.run_id] = handle
    adapter._session_to_run[handle.session_id] = handle.run_id
    adapter._event_queues[handle.run_id] = asyncio.Queue()
    adapter._ws = EventWebSocket()
    await adapter._recv_loop()

    result = adapter._completed_results[handle.run_id]
    assert result.status == RunStatus.COMPLETED
    assert result.summary == "canonical"


@pytest.mark.asyncio
async def test_streaming_tool_and_delta_events_do_not_terminate_before_complete() -> None:
    adapter = StreamingGatewayAdapter([
        {"run_id": "tui:session-stream", "type": "tool_start", "payload": {"tool_name": "x"}},
        {"run_id": "tui:session-stream", "type": "tool_complete", "payload": {"tool_name": "x", "exit_code": 0}},
        {"run_id": "tui:session-stream", "type": "message", "payload": {"text": "delta"}},
        {"run_id": "tui:session-stream", "type": "completed", "payload": {"summary": "done"}},
    ])

    handle = await adapter.submit(make_task())
    result = await adapter.wait(handle)

    assert result.status == RunStatus.COMPLETED
    assert result.summary == "done"


@pytest.mark.asyncio
async def test_streaming_error_is_terminal_failure() -> None:
    adapter = StreamingGatewayAdapter([
        {"run_id": "tui:session-stream", "type": "error", "payload": {"message": "gateway failure"}},
    ])

    handle = await adapter.submit(make_task())
    result = await adapter.wait(handle)

    assert result.status == RunStatus.FAILED
    assert result.error == "gateway failure"


@pytest.mark.asyncio
async def test_streaming_wait_times_out_before_terminal_event() -> None:
    adapter = StreamingGatewayAdapter([], timeout=0.01)
    handle = await adapter.submit(make_task())

    with pytest.raises(TimeoutError, match="streaming runtime wait timed out"):
        await adapter.wait(handle)


@pytest.mark.asyncio
async def test_clean_websocket_close_fails_active_stream_run() -> None:
    class CleanCloseWebSocket:
        def __aiter__(self):
            async def messages():
                if False:
                    yield None

            return messages()

    adapter = HermesAdapter()
    handle = RunHandle(
        run_id="tui:session-clean-close",
        task_id="task-clean-close",
        session_id="session-clean-close",
    )
    adapter._handles[handle.run_id] = handle
    adapter._session_to_run[handle.session_id] = handle.run_id
    adapter._streaming_runs.add(handle.run_id)
    adapter._event_queues[handle.run_id] = asyncio.Queue()
    adapter._run_state[handle.run_id] = {
        "files_changed": [],
        "summary": "",
        "error": None,
        "usage": None,
        "status": RunStatus.RUNNING,
    }
    adapter._ws = CleanCloseWebSocket()

    await adapter._recv_loop()

    result = adapter._completed_results[handle.run_id]
    assert result.status == RunStatus.FAILED
    assert "disconnected" in (result.error or "").lower()


class TestHermesAdapterContract:
    @pytest.mark.asyncio
    async def test_real_adapter_satisfies_runtime_protocol_and_uses_run_handles(self):
        adapter = FakeGatewayAdapter()
        assert isinstance(adapter, AgentRuntime)
        caps = await adapter.capabilities()
        assert caps.streaming_events is True
        assert caps.session_resume is False
        assert caps.filesystem_read is True
        assert caps.filesystem_write is False
        assert caps.shell is False
        assert caps.tests is False
        assert caps.web is False
        handle = await adapter.submit(make_task())
        result = await adapter.wait(handle)
        usage = await adapter.usage(handle)
        await adapter.cancel(handle)
        await adapter.steer(handle, "continue")

        assert result.task_id == handle.task_id
        assert result.status == RunStatus.COMPLETED
        assert result.summary == "done"
        assert usage.total_tokens == 15
        run_scoped = [params for method, params in adapter.calls if method.startswith("session.") and "run_id" in params]
        assert run_scoped
        assert all(params["run_id"] == "run-1" for params in run_scoped)
        assert all(not isinstance(params["run_id"], RunHandle) for params in run_scoped)

    def test_workspace_prompt_uses_declared_path_field(self):
        task = make_task(workspace=WorkspaceSpec(path="D:/worktree", branch="agent/test"))
        prompt = task.prompt_preamble()
        assert "D:/worktree" in prompt
        assert "agent/test" in prompt

    @pytest.mark.asyncio
    async def test_submit_targets_authoritative_workspace_as_session_cwd(self):
        adapter = FakeGatewayAdapter()
        task = make_task(
            workspace=WorkspaceSpec(
                path="D:/worktrees/task-exact",
                branch="agent/task-exact",
            )
        )

        await adapter.submit(task)

        assert adapter.calls[0] == (
            "session.create",
            {"cwd": "D:/worktrees/task-exact"},
        )

    @pytest.mark.asyncio
    async def test_submit_rejects_scoped_filesystem_task_without_workspace(self):
        adapter = FakeGatewayAdapter()
        task = make_task(allowed_paths=["README.md"])

        with pytest.raises(
            RuntimeError,
            match="filesystem task requires an authoritative workspace",
        ):
            await adapter.submit(task)

        assert adapter.calls == []

    @pytest.mark.asyncio
    async def test_receive_loop_buffers_event_before_submit_registers_queue(self):
        class EarlyEventWebSocket:
            def __aiter__(self):
                async def messages():
                    yield json.dumps({
                        "type": "event",
                        "run_id": "run-early",
                        "event": {
                            "run_id": "run-early",
                            "type": "completed",
                            "payload": {"summary": "early", "files_changed": []},
                        },
                    })
                return messages()

        adapter = HermesAdapter()
        adapter._ws = EarlyEventWebSocket()
        await adapter._recv_loop()

        assert "run-early" in adapter._event_queues
        buffered = await adapter._event_queues["run-early"].get()
        assert buffered["type"] == "completed"

    @pytest.mark.asyncio
    async def test_submit_preserves_early_event_queue_and_wait_completes(self):
        class EarlyEventAdapter(HermesAdapter):
            async def _call(self, method, params):
                if method == "session.create":
                    return {"session_id": "session-early"}
                if method == "prompt.submit":
                    queue = self._event_queues.setdefault("run-early", asyncio.Queue())
                    await queue.put({
                        "run_id": "run-early",
                        "type": "completed",
                        "payload": {"summary": "arrived before submit returned"},
                    })
                    return {"run_id": "run-early"}
                if method == "session.subscribe":
                    return {}
                raise AssertionError(method)

        adapter = EarlyEventAdapter()
        early_queue = adapter._event_queues.setdefault("run-early", asyncio.Queue())
        handle = await adapter.submit(make_task())

        assert adapter._event_queues[handle.run_id] is early_queue
        result = await asyncio.wait_for(adapter.wait(handle), timeout=1)
        assert result.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_disconnect_fails_active_run_and_wait_returns(self):
        class DummyWebSocket:
            async def close(self):
                return None

        adapter = HermesAdapter()
        handle = RunHandle(run_id="run-disconnect", task_id="task-disconnect")
        adapter._handles[handle.run_id] = handle
        adapter._event_queues[handle.run_id] = asyncio.Queue()
        adapter._ws = DummyWebSocket()
        adapter._connected.set()

        await adapter.disconnect()
        result = await asyncio.wait_for(adapter.wait(handle), timeout=1)

        assert result.status == RunStatus.FAILED
        assert "disconnected" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_wait_and_event_stream_do_not_compete_for_terminal_event(self):
        class SubscriptionAdapter(HermesAdapter):
            async def _call(self, method, params):
                assert method == "session.subscribe"
                return {}

        adapter = SubscriptionAdapter()
        handle = RunHandle(run_id="run-shared", task_id="task-shared")
        adapter._handles[handle.run_id] = handle
        queue = adapter._event_queues.setdefault(handle.run_id, asyncio.Queue())

        stream = adapter.events(handle)
        stream_task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        wait_task = asyncio.create_task(adapter.wait(handle))
        await asyncio.sleep(0)

        await queue.put({
            "run_id": handle.run_id,
            "type": "completed",
            "payload": {"summary": "done"},
        })
        streamed = await asyncio.wait_for(stream_task, timeout=1)
        result = await asyncio.wait_for(wait_task, timeout=1)

        assert streamed.type == "completed"
        assert result.status == RunStatus.COMPLETED
        await stream.aclose()

    @pytest.mark.asyncio
    async def test_disconnect_unblocks_pending_subscription_rpc(self):
        class PendingWebSocket:
            def __init__(self):
                self.sent = asyncio.Event()
                self.close_count = 0

            async def send(self, _payload):
                self.sent.set()

            async def close(self):
                self.close_count += 1

        adapter = HermesAdapter()
        ws = PendingWebSocket()
        adapter._ws = ws
        adapter._connected.set()
        handle = RunHandle(run_id="run-pending", task_id="task-pending")
        adapter._handles[handle.run_id] = handle
        adapter._event_queues[handle.run_id] = asyncio.Queue()

        waiter = asyncio.create_task(adapter.wait(handle))
        await asyncio.wait_for(ws.sent.wait(), timeout=1)
        await adapter.disconnect()
        result = await asyncio.wait_for(waiter, timeout=1)

        assert result.status == RunStatus.FAILED
        assert not adapter._pending
        assert ws.close_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_during_submit_registration_gap_returns_failed_handle(self):
        class GapAdapter(HermesAdapter):
            async def _call(self, method, params):
                if method == "session.create":
                    self._connected.set()
                    return {"session_id": "session-gap"}
                if method == "prompt.submit":
                    await self._fail_active_runs("gap disconnect")
                    self._connected.clear()
                    return {"run_id": "run-gap"}
                if method == "session.subscribe":
                    return {}
                raise AssertionError(method)

        adapter = GapAdapter()
        handle = await adapter.submit(make_task())
        result = await asyncio.wait_for(adapter.wait(handle), timeout=1)
        assert result.status == RunStatus.FAILED
        assert "disconnect" in (result.error or "")

    @pytest.mark.asyncio
    async def test_early_buffered_stream_is_still_emitted_in_order(self):
        class EarlyStreamAdapter(HermesAdapter):
            async def _call(self, method, params):
                if method == "session.create":
                    return {"session_id": "session-stream"}
                if method == "prompt.submit":
                    queue = self._event_queues.setdefault("run-stream", asyncio.Queue())
                    for event_type, payload in (
                        ("tool_complete", {"files_written": ["early.py"]}),
                        ("usage", {"total_tokens": 10}),
                        ("completed", {"summary": "early"}),
                    ):
                        await queue.put({
                            "run_id": "run-stream", "type": event_type,
                            "payload": payload,
                        })
                    return {"run_id": "run-stream"}
                if method == "session.subscribe":
                    return {}
                raise AssertionError(method)

        adapter = EarlyStreamAdapter()
        handle = await adapter.submit(make_task())
        result = await adapter.wait(handle)
        streamed = [event.type async for event in adapter.events(handle)]

        assert result.status == RunStatus.COMPLETED
        assert streamed == ["tool_complete", "usage", "completed"]
        assert adapter._event_queues[handle.run_id].empty()

    @pytest.mark.asyncio
    async def test_explicit_disconnect_cancels_reconnect_and_is_idempotent(self):
        class ReconnectAdapter(HermesAdapter):
            def __init__(self):
                super().__init__(reconnect_delay=0.01)
                self.connect_count = 0

            async def connect(self):
                self.connect_count += 1
                self._connected.set()

        class DummyWebSocket:
            def __init__(self):
                self.close_count = 0

            async def close(self):
                self.close_count += 1

        adapter = ReconnectAdapter()
        ws = DummyWebSocket()
        adapter._ws = ws
        reconnect = asyncio.create_task(adapter._reconnect())
        await adapter.disconnect()
        await adapter.disconnect()
        await reconnect

        assert adapter.connect_count == 0
        assert ws.close_count == 1
        assert not adapter._connected.is_set()

    def test_first_terminal_event_wins(self):
        adapter = HermesAdapter()
        handle = RunHandle(run_id="run-first", task_id="task-first")
        adapter._record_event(handle, _parse_event({
            "run_id": handle.run_id,
            "type": "completed",
            "payload": {"summary": "first"},
        }))
        adapter._record_event(handle, _parse_event({
            "run_id": handle.run_id,
            "type": "error",
            "payload": {"message": "late"},
        }))

        result = adapter._completed_results[handle.run_id]
        assert result.status == RunStatus.COMPLETED
        assert result.summary == "first"
        assert result.error is None

    def test_terminal_result_without_usage_preserves_run_id_and_unknown_usage(self):
        adapter = HermesAdapter()
        handle = RunHandle(run_id="run-no-usage", task_id="task-no-usage")

        adapter._record_event(handle, _parse_event({
            "run_id": handle.run_id,
            "type": "completed",
            "payload": {"summary": "done without telemetry"},
        }))

        result = adapter._completed_results[handle.run_id]
        assert result.run_id == "run-no-usage"
        assert result.status == RunStatus.COMPLETED
        assert result.usage is None

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                },
                (0, 0, 0, 0.0),
            ),
            ({"input_tokens": 100}, (100, None, None, None)),
            (
                {"input_tokens": 100, "output_tokens": 20},
                (100, 20, 120, None),
            ),
        ],
    )
    def test_usage_event_preserves_zero_partial_and_derived_truth(
        self, payload, expected
    ):
        adapter = HermesAdapter()
        handle = RunHandle(run_id="run-usage", task_id="task-usage")

        adapter._record_event(handle, _parse_event({
            "run_id": handle.run_id,
            "type": "usage",
            "payload": payload,
        }))
        adapter._record_event(handle, _parse_event({
            "run_id": handle.run_id,
            "type": "completed",
            "payload": {},
        }))

        usage = adapter._completed_results[handle.run_id].usage
        assert usage is not None
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.estimated_cost_usd,
        ) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rpc_result", "expected"),
        [
            ({}, (None, None, None, None)),
            ({"input_tokens": 0, "output_tokens": 0}, (0, 0, 0, None)),
            ({"input_tokens": 7}, (7, None, None, None)),
            ({"input_tokens": 7, "output_tokens": 3}, (7, 3, 10, None)),
        ],
    )
    async def test_usage_rpc_preserves_missing_fields(self, rpc_result, expected):
        class UsageAdapter(HermesAdapter):
            async def _call(self, method, params):
                assert method == "session.usage"
                assert params == {"run_id": "run-rpc"}
                return rpc_result

        usage = await UsageAdapter().usage(
            RunHandle(run_id="run-rpc", task_id="task-rpc")
        )

        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.estimated_cost_usd,
        ) == expected


class TestMockAdapter:
    @pytest.mark.asyncio
    async def test_submit_returns_handle(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        handle = await adapter.submit(make_task())
        assert handle.run_id
        assert handle.task_id

    @pytest.mark.asyncio
    async def test_pass_scenario(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", files_changed=["src/auth.py"], summary="done")
        handle = await adapter.submit(make_task())
        result = await adapter.wait(handle)
        assert result.status == RunStatus.COMPLETED
        assert "src/auth.py" in result.files_changed
        assert result.summary == "done"

    @pytest.mark.asyncio
    async def test_fail_scenario(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("fail", error_message="tool crashed")
        handle = await adapter.submit(make_task())
        result = await adapter.wait(handle)
        assert result.status == RunStatus.FAILED
        assert "tool crashed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cancel(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        handle = await adapter.submit(make_task())
        await adapter.cancel(handle)
        # After cancel the internal state is CANCELLED
        assert adapter._runs[handle.run_id].status == RunStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_usage_returned(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", input_tokens=1000, output_tokens=400)
        handle = await adapter.submit(make_task())
        usage = await adapter.usage(handle)
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 400
        assert usage.total_tokens == 1400

    @pytest.mark.asyncio
    async def test_event_stream_pass(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", files_changed=["a.py"])
        handle = await adapter.submit(make_task())
        events = []
        async for evt in adapter.events(handle):
            events.append(evt.type)
        assert "completed" in events
        assert "tool_complete" in events

    @pytest.mark.asyncio
    async def test_event_stream_ends_on_error(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("fail")
        handle = await adapter.submit(make_task())
        events = []
        async for evt in adapter.events(handle):
            events.append(evt.type)
        assert "error" in events
        assert "completed" not in events

    @pytest.mark.asyncio
    async def test_approval_required_scenario(self):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("approval_required")
        handle = await adapter.submit(make_task())
        # Internal status is PENDING before streaming
        assert adapter._runs[handle.run_id].status == RunStatus.PENDING
        events = [e.type async for e in adapter.events(handle)]
        assert "approval_request" in events

    @pytest.mark.asyncio
    async def test_capabilities(self):
        adapter = MockHermesAdapter()
        caps = await adapter.capabilities()
        assert caps.streaming_events is True
        assert caps.cancellation is True

    @pytest.mark.asyncio
    async def test_cost_estimated(self):
        """estimated_cost_usd must be non-zero for non-trivial token counts."""
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", input_tokens=1000, output_tokens=500)
        handle = await adapter.submit(make_task())
        usage = await adapter.usage(handle)
        assert usage.estimated_cost_usd is not None
        assert usage.estimated_cost_usd > 0


class TestStateMachine:
    @pytest.mark.asyncio
    async def test_happy_path_transitions(self, tmp_path):
        from orchestrator.state_machine import StateMachine, TaskStatus
        from storage.database import Database

        db = Database(tmp_path / "test.db")
        await db.connect()
        sm = StateMachine(db)

        task = make_task()
        record = await sm.create(task)
        assert record.status == TaskStatus.RECEIVED

        await record.transition(TaskStatus.PROFILED)
        assert record.status == TaskStatus.PROFILED

        await record.mark_routed("delegation")
        assert record.status == TaskStatus.ROUTED
        assert record.route == "delegation"

        await record.mark_running("run-abc")
        assert record.status == TaskStatus.RUNNING

        await record.mark_evaluating()
        assert record.status == TaskStatus.EVALUATING

        await record.mark_completed()
        assert record.status == TaskStatus.COMPLETED
        await db.close()

    @pytest.mark.asyncio
    async def test_illegal_transition_raises(self, tmp_path):
        from orchestrator.state_machine import IllegalTransitionError, StateMachine, TaskStatus
        from storage.database import Database

        db = Database(tmp_path / "test.db")
        await db.connect()
        sm = StateMachine(db)

        task = make_task()
        record = await sm.create(task)
        with pytest.raises(IllegalTransitionError):
            await record.transition(TaskStatus.COMPLETED)  # RECEIVED → COMPLETED illegal
        await db.close()

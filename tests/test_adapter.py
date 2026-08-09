"""Tests for MockHermesAdapter and state machine."""
from __future__ import annotations

import json

import pytest

from adapters.hermes.gateway import HermesAdapter
from adapters.mock import MockHermesAdapter
from adapters.runtime import AgentRuntime
from contracts.result import RunHandle, RunStatus
from contracts.task import TaskContract, WorkspaceSpec


def make_task(**kwargs) -> TaskContract:
    return TaskContract(goal="test task", **kwargs)


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


class TestHermesAdapterContract:
    @pytest.mark.asyncio
    async def test_real_adapter_satisfies_runtime_protocol_and_uses_run_handles(self):
        adapter = FakeGatewayAdapter()
        assert isinstance(adapter, AgentRuntime)
        caps = await adapter.capabilities()
        assert caps.streaming_events is True
        assert caps.session_resume is False
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

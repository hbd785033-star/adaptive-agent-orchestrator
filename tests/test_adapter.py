"""Tests for MockHermesAdapter and state machine."""
from __future__ import annotations

import pytest

from adapters.mock import MockHermesAdapter
from contracts.result import RunStatus
from contracts.task import TaskContract


def make_task(**kwargs) -> TaskContract:
    return TaskContract(goal="test task", **kwargs)


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

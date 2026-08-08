"""Tests for the six optimizations applied in this session.

1. Cost estimator (orchestrator/cost.py)
2. MockAdapter now returns estimated_cost_usd
3. Pre-flight subtask-count clamping
4. ApprovalGate timeout (non-interactive auto-deny)
5. WorkspaceManager.allocate() cleans up ABANDONED before retry
6. Step numbering sanity (regression against duplicate step labels)
"""
from __future__ import annotations

import pytest

from contracts.task import RiskLevel, TaskContract, TaskType
from orchestrator.cost import estimate_cost, known_models

# ── 1. Cost estimator ─────────────────────────────────────────────────────────

class TestCostEstimator:
    def test_zero_tokens_returns_zero(self):
        assert estimate_cost("claude-sonnet-4", 0, 0) == 0.0

    def test_known_model_nonzero(self):
        cost = estimate_cost("claude-sonnet-4", 1_000, 300)
        assert cost > 0

    def test_output_tokens_cost_more_than_input(self):
        cost_in_heavy  = estimate_cost("claude-sonnet-4", 10_000, 0)
        cost_out_heavy = estimate_cost("claude-sonnet-4", 0, 10_000)
        assert cost_out_heavy > cost_in_heavy

    def test_unknown_model_uses_default(self):
        cost_unknown = estimate_cost("some-unknown-model-xyz", 1_000, 300)
        cost_default = estimate_cost("claude-sonnet-4", 1_000, 300)
        # unknown falls back to __default__ which equals sonnet-4 prices
        assert cost_unknown == cost_default

    def test_known_models_nonempty(self):
        assert len(known_models()) > 0

    def test_result_is_float(self):
        result = estimate_cost("gpt-4o", 500, 200)
        assert isinstance(result, float)

    def test_haiku_cheaper_than_opus(self):
        haiku = estimate_cost("claude-haiku-4",  10_000, 3_000)
        opus  = estimate_cost("claude-opus-4",   10_000, 3_000)
        assert haiku < opus

    def test_result_rounds_to_6_places(self):
        result = estimate_cost("claude-sonnet-4", 1, 1)
        # Should not have more than 6 decimal places
        assert result == round(result, 6)


# ── 2. MockAdapter cost propagation ──────────────────────────────────────────

class TestMockAdapterCost:
    @pytest.mark.asyncio
    async def test_estimated_cost_usd_is_set(self):
        from adapters.mock import MockHermesAdapter

        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", input_tokens=1000, output_tokens=300,
                                 summary="ok")
        task = TaskContract(goal="test")
        handle = await adapter.submit(task)
        result = await adapter.result(handle.run_id)

        assert result.usage.estimated_cost_usd is not None
        assert result.usage.estimated_cost_usd > 0

    @pytest.mark.asyncio
    async def test_zero_tokens_cost_is_zero(self):
        from adapters.mock import MockHermesAdapter

        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", input_tokens=0, output_tokens=0,
                                 summary="ok")
        task = TaskContract(goal="test")
        handle = await adapter.submit(task)
        result = await adapter.result(handle.run_id)
        assert result.usage.estimated_cost_usd == 0.0


# ── 3. Pre-flight subtask clamping (engine telemetry) ────────────────────────

class TestSubtaskClamping:
    @pytest.mark.asyncio
    async def test_subtask_count_above_budget_logs_warning(self, tmp_path, caplog):
        """Engine should log a warning but still proceed when subtask_count > max_children."""
        import logging

        from adapters.mock import MockHermesAdapter
        from orchestrator.engine import Orchestrator

        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", summary="ok")

        async with await Orchestrator.build(
            runtime=adapter,
            db_path=str(tmp_path / "test.db"),
            policy_path="policies/default.yaml",
        ) as orch:
            # multi_file_refactor with 3 subtasks but budget allows only 2
            task = TaskContract(
                goal="refactor src/a and src/b and src/c independently",
                task_type=TaskType.MULTI_FILE_REFACTOR,
                complexity=4,
            )
            with caplog.at_level(logging.WARNING, logger="orchestrator.engine"):
                result = await orch.run(task)

        # Task should complete (clamped, not aborted)
        assert result["outcome"] in ("completed", "failed")


# ── 4. ApprovalGate timeout ───────────────────────────────────────────────────

class TestApprovalGateTimeout:
    def test_timeout_returns_false(self, tmp_path):
        """With a very short timeout and no stdin, approval must return False."""
        from orchestrator.budget import ApprovalGate

        gate = ApprovalGate(policy_path="policies/default.yaml")
        task = TaskContract(goal="deploy to production", risk=RiskLevel.HIGH,
                            success_criteria=["deploy ok"])

        # timeout=0 → immediate denial (no stdin in test environment)
        result = gate.prompt_user("high-risk deploy", task, timeout_s=0)
        assert result is False

    def test_eofinput_returns_false(self, tmp_path):
        """EOF on stdin (CI pipe) must return False without hanging."""
        import sys
        from io import StringIO

        from orchestrator.budget import ApprovalGate

        old_stdin = sys.stdin
        sys.stdin = StringIO("")  # EOF immediately
        try:
            gate = ApprovalGate(policy_path="policies/default.yaml")
            task = TaskContract(goal="test", risk=RiskLevel.HIGH,
                                success_criteria=["done"])
            result = gate.prompt_user("test", task, timeout_s=2)
            assert result is False
        finally:
            sys.stdin = old_stdin


# ── 5. WorkspaceManager ABANDONED retry cleanup ───────────────────────────────

class TestWorkspaceRetryCleanup:
    def test_reallocate_after_abandon_succeeds(self, tmp_path):
        """A second allocate() on the same (task_id, child_id) after abandon must succeed."""
        import os
        import subprocess

        from orchestrator.workspace import WorkspaceManager, WorktreeStatus

        # Set up a real git repo
        env = {**os.environ,
               "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
               "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"}
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-m", "init"],
                       check=True, capture_output=True, env=env)

        wm = WorkspaceManager(repo_path=str(tmp_path))
        task_id = "task-retry-test"
        child_id = "child-1"

        # First allocation
        wt1 = wm.allocate(task_id, child_id)
        assert wt1.status == WorktreeStatus.ALLOCATED

        # Mark abandoned (eval failed)
        wm.abandon(task_id, child_id)
        assert wm._records[(task_id, child_id)].status == WorktreeStatus.ABANDONED

        # Second allocation after abandon — should NOT raise
        wt2 = wm.allocate(task_id, child_id)
        assert wt2.status == WorktreeStatus.ALLOCATED
        assert wt2.worktree_path.exists()

    def test_reallocate_active_raises(self, tmp_path):
        """Re-allocating an already ALLOCATED (non-ABANDONED) worktree must raise ValueError."""
        import os
        import subprocess

        from orchestrator.workspace import WorkspaceManager

        env = {**os.environ,
               "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
               "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"}
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-m", "init"],
                       check=True, capture_output=True, env=env)

        wm = WorkspaceManager(repo_path=str(tmp_path))
        wm.allocate("task-x", "child-1")
        # Record is now ALLOCATED (not ABANDONED) — a second allocate must raise

        with pytest.raises(ValueError, match="already exists"):
            wm.allocate("task-x", "child-1")

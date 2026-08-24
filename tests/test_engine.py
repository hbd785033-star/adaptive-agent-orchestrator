"""
End-to-end orchestrator integration test using MockHermesAdapter.

Exercises the full pipeline:
  TaskContract → Profiler → Router → Budget → Execution → EvalGate → Telemetry → DB
without requiring a live Hermes instance.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from adapters.mock import MockHermesAdapter
from contracts.task import SubtaskSpec, TaskContract, TaskType
from orchestrator.engine import Orchestrator


async def build_orch(tmp_path: Path, runtime) -> Orchestrator:
    """Helper: build Orchestrator pointed at a temp DB and tmp policy."""
    import os
    import subprocess

    if not (tmp_path / ".git").exists():
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / ".gitignore").write_text(".worktrees/\n*.db\n*.db-*\n*.yaml\n")
        subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "test bootstrap"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com"},
        )
    policy = tmp_path / "default.yaml"
    policy.write_text(
        """
policy_version: "routing-v1.0"
routing:
  delegation:
    min_independent_subtasks: 2
    min_estimated_input_tokens: 8000
    allowed_task_types:
      - multi_file_refactor
      - parallel_research
      - test_and_implement
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
  always_require:
    - delete_files
    - deploy
  require_for_risk_levels:
    - 3
    - 4
worktree:
  base_path: ".worktrees"
  readonly_task_types:
    - parallel_research
    - code_review
"""
    )
    return await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "test.db"),
        repo_path=str(tmp_path),
        policy_path=str(policy),
    )


class TestOrchestratorHappyPath:
    @staticmethod
    def _deny_approval(orch):
        class DenyingApproval:
            @staticmethod
            def requires_approval(_task):
                return False, ""

            @staticmethod
            def prompt_user(_reason, _task):
                return False

        orch._approval = DenyingApproval()

    @pytest.mark.asyncio
    async def test_single_write_fails_closed_without_workspace_manager(self, tmp_path):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        async with await build_orch(tmp_path, adapter) as orch:
            orch._wm = None
            task = TaskContract(
                goal="must not touch root",
                task_type=TaskType.CODE_FIX,
                allowed_paths=["src/**"],
            )
            result = await orch.run(task)

        assert result["outcome"] == "failed"
        assert "workspace unavailable" in result["detail"]
        assert not adapter._runs

    @pytest.mark.asyncio
    async def test_generic_runtime_approval_denial_cancels_single_run(self, tmp_path):
        class CancelTrackingAdapter(MockHermesAdapter):
            def __init__(self):
                super().__init__()
                self.cancelled_run_ids = []

            async def cancel(self, handle):
                self.cancelled_run_ids.append(handle.run_id)
                await super().cancel(handle)

        adapter = CancelTrackingAdapter()
        adapter.enqueue_scenario("approval_required")
        orch = await build_orch(tmp_path, adapter)
        self._deny_approval(orch)

        result = await orch.run(TaskContract(goal="request approval", task_type=TaskType.CODE_FIX))
        await orch.close()

        assert result["outcome"] == "failed"
        assert len(adapter._runs) == 1
        assert adapter.cancelled_run_ids == list(adapter._runs)
        assert next(iter(adapter._runs.values())).status.value in {"cancelled", "completed"}

    @pytest.mark.asyncio
    async def test_single_call_threshold_blocks_before_submit(self, tmp_path):
        from orchestrator.execution_policy import ExecutionPolicy

        adapter = MockHermesAdapter()
        orch = await build_orch(tmp_path, adapter)
        self._deny_approval(orch)
        orch._execution_policy = ExecutionPolicy(
            require_approval_above_calls=0, max_total_calls=8
        )

        result = await orch.run(TaskContract(goal="protected call", task_type=TaskType.CODE_FIX))
        await orch.close()

        assert result["outcome"] in {"failed", "abandoned"}
        assert not adapter._runs

    @pytest.mark.asyncio
    async def test_delegated_call_threshold_blocks_children_before_submit(self, tmp_path):
        from orchestrator.execution_policy import ExecutionPolicy

        adapter = MockHermesAdapter()
        orch = await build_orch(tmp_path, adapter)
        self._deny_approval(orch)
        policy = ExecutionPolicy(require_approval_above_calls=0, max_total_calls=8)
        orch._execution_policy = policy
        orch._delegation_executor._execution_policy = policy
        task = TaskContract(
            goal="protected delegation",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "abandoned"
        assert not adapter._runs

    @pytest.mark.asyncio
    async def test_delegated_threshold_denial_blocks_entire_batch_before_submit(
        self, tmp_path
    ):
        from orchestrator.execution_policy import ExecutionPolicy

        adapter = MockHermesAdapter()
        orch = await build_orch(tmp_path, adapter)
        self._deny_approval(orch)
        policy = ExecutionPolicy(require_approval_above_calls=1, max_total_calls=8)
        orch._execution_policy = policy
        orch._delegation_executor._execution_policy = policy
        task = TaskContract(
            goal="atomically protect delegated batch",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] in {"failed", "abandoned"}
        assert not adapter._runs

    @pytest.mark.asyncio
    async def test_delegated_threshold_approval_authorizes_protected_batch(
        self, tmp_path
    ):
        from orchestrator.execution_policy import ExecutionPolicy

        class ApprovingGate:
            @staticmethod
            def requires_approval(_task):
                return False, ""

            @staticmethod
            def prompt_user(_reason, _task):
                return True

        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        orch._approval = ApprovingGate()
        policy = ExecutionPolicy(require_approval_above_calls=1, max_total_calls=8)
        orch._execution_policy = policy
        orch._delegation_executor._execution_policy = policy
        task = TaskContract(
            goal="approve delegated batch",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "completed"
        assert len(adapter._runs) == 2

    @pytest.mark.asyncio
    async def test_generic_runtime_approval_denial_cancels_delegated_child(self, tmp_path):
        class CancelTrackingAdapter(MockHermesAdapter):
            def __init__(self):
                super().__init__()
                self.cancelled_run_ids = []

            async def cancel(self, handle):
                self.cancelled_run_ids.append(handle.run_id)
                await super().cancel(handle)

        adapter = CancelTrackingAdapter()
        adapter.enqueue_scenario(
            "pass",
            extra_events=[{"type": "approval_request", "payload": {}}],
        )
        adapter.enqueue_scenario(
            "pass",
            extra_events=[{"type": "approval_request", "payload": {}}],
        )
        orch = await build_orch(tmp_path, adapter)
        task = TaskContract(
            goal="delegated approval",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "failed"
        assert len(adapter._runs) == 2
        assert set(adapter.cancelled_run_ids) == set(adapter._runs)
        assert all(
            run.status.value in {"cancelled", "completed"}
            for run in adapter._runs.values()
        )

    @pytest.mark.asyncio
    async def test_delegated_write_fails_closed_without_workspace_manager(self, tmp_path):
        import subprocess

        adapter = MockHermesAdapter()
        async with await build_orch(tmp_path, adapter) as orch:
            orch._wm = None
            base = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=tmp_path,
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            task = TaskContract(
                goal="delegate writes safely",
                task_type=TaskType.MULTI_FILE_REFACTOR,
                complexity=4,
                subtasks=[
                    SubtaskSpec(id="one", goal="write one"),
                    SubtaskSpec(id="two", goal="write two"),
                ],
            )
            result = await orch.run(task)

        assert result["outcome"] == "failed"
        assert "workspace unavailable" in result["detail"]
        assert not adapter._runs
        assert subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path,
            check=True, capture_output=True, text=True,
        ).stdout.strip() == base

    @pytest.mark.asyncio
    async def test_partial_delegation_allocation_cleans_before_any_submit(self, tmp_path, monkeypatch):
        import subprocess

        adapter = MockHermesAdapter()
        async with await build_orch(tmp_path, adapter) as orch:
            assert orch._wm is not None
            real_allocate = orch._wm.allocate
            allocations = 0

            def fail_second_allocation(task_id, child_id):
                nonlocal allocations
                allocations += 1
                if allocations == 2:
                    raise RuntimeError("injected allocation failure")
                return real_allocate(task_id, child_id)

            monkeypatch.setattr(orch._wm, "allocate", fail_second_allocation)
            task = TaskContract(
                goal="delegate writes atomically",
                task_type=TaskType.MULTI_FILE_REFACTOR,
                complexity=4,
                subtasks=[
                    SubtaskSpec(id="one", goal="write one"),
                    SubtaskSpec(id="two", goal="write two"),
                ],
            )
            result = await orch.run(task)
            records = list(orch._wm.list_records())

        assert result["outcome"] == "failed"
        assert "allocation" in result["detail"]
        assert not adapter._runs
        assert records and all(item.status.value == "cleaned" for item in records)
        assert not list((tmp_path / ".worktrees").glob("**/child-*"))
        assert subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        ).stdout.strip() == ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_stage", ["events", "wait", "usage"])
    async def test_single_runtime_lifecycle_exception_rolls_back_workspace(
        self, tmp_path, failure_stage
    ):
        import subprocess

        class ExplodingAdapter(MockHermesAdapter):
            async def submit(self, task):
                (Path(task.workspace.path) / "unsafe.txt").write_text("unsafe")
                return await super().submit(task)

            async def events(self, handle, *, after=None):
                if failure_stage == "events":
                    raise RuntimeError("events exploded")
                async for event in super().events(handle, after=after):
                    yield event

            async def wait(self, handle):
                if failure_stage == "wait":
                    raise RuntimeError("wait exploded")
                return await super().wait(handle)

            async def usage(self, handle):
                if failure_stage == "usage":
                    raise RuntimeError("usage exploded")
                return await super().usage(handle)

        adapter = ExplodingAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        task = TaskContract(goal="do not leak", task_type=TaskType.CODE_FIX)

        with pytest.raises(RuntimeError, match=f"{failure_stage} exploded"):
            await orch.run(task)
        await orch.close()

        assert not (tmp_path / "unsafe.txt").exists()
        remaining = list((tmp_path / ".worktrees").glob("**/single-*"))
        if failure_stage in {"events", "wait"}:
            assert remaining, "unconfirmed runtime must retain its quarantined workspace"
        else:
            assert not remaining
        assert subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        ).stdout.strip() == ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mutation",
        [
            "tracked",
            "untracked",
            "commit",
            "new_ignored",
            "ref",
            "object",
            "extra_worktree",
            "internal_sibling",
            "assume_unchanged",
            "skip_worktree",
        ],
    )
    async def test_read_only_execution_isolated_and_mutation_is_rejected(self, tmp_path, mutation):
        import os
        import subprocess

        class MutatingReadOnlyAdapter(MockHermesAdapter):
            async def submit(self, task):
                root = Path(task.workspace.path) if task.workspace else tmp_path
                if mutation == "tracked":
                    (root / ".gitignore").write_text("runtime changed tracked state\n")
                elif mutation == "untracked":
                    (root / "untracked.txt").write_text("mutation")
                elif mutation == "commit":
                    (root / "committed.txt").write_text("mutation")
                    subprocess.run(["git", "add", "committed.txt"], cwd=root, check=True)
                    subprocess.run(
                        ["git", "commit", "-m", "runtime mutation"], cwd=root,
                        check=True, capture_output=True,
                        env={
                            **os.environ,
                            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
                            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com",
                        },
                    )
                elif mutation == "ref":
                    subprocess.run(
                        ["git", "branch", "runtime-side-effect"],
                        cwd=root,
                        check=True,
                        capture_output=True,
                    )
                elif mutation == "object":
                    subprocess.run(
                        ["git", "hash-object", "-w", "--stdin"],
                        cwd=root,
                        input="runtime-only object\n",
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                elif mutation == "extra_worktree":
                    subprocess.run(
                        [
                            "git",
                            "worktree",
                            "add",
                            "--detach",
                            str(tmp_path.parent / f"{tmp_path.name}-runtime-extra"),
                            "HEAD",
                        ],
                        cwd=root,
                        check=True,
                        capture_output=True,
                    )
                elif mutation == "internal_sibling":
                    sibling = tmp_path / ".worktrees" / "unrelated-sibling"
                    sibling.mkdir(parents=True)
                    (sibling / "runtime.txt").write_text("persistent mutation")
                elif mutation in {"assume_unchanged", "skip_worktree"}:
                    flag = (
                        "--assume-unchanged"
                        if mutation == "assume_unchanged"
                        else "--skip-worktree"
                    )
                    subprocess.run(
                        ["git", "update-index", flag, ".gitignore"],
                        cwd=root,
                        check=True,
                        capture_output=True,
                    )
                else:
                    (root / "runtime.db").write_text("ignored mutation")
                return await super().submit(task)

        adapter = MutatingReadOnlyAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        result = await orch.run(TaskContract(
            goal="inspect only", task_type=TaskType.CODE_REVIEW,
        ))
        await orch.close()

        if mutation == "extra_worktree":
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "remove",
                    "--force",
                    str(tmp_path.parent / f"{tmp_path.name}-runtime-extra"),
                ],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )

        assert result["outcome"] == "failed"
        assert result["verification_status"] == "fail"
        assert subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path,
            check=True, capture_output=True, text=True,
        ).stdout.strip() == base
        assert not (tmp_path / "committed.txt").exists()
        assert not (tmp_path / "runtime.db").exists()
        assert not (tmp_path / "untracked.txt").exists()

    @pytest.mark.asyncio
    async def test_read_only_unchanged_preexisting_ignored_file_is_not_attributed(
        self, tmp_path
    ):
        (tmp_path / "existing.db").write_text("baseline")
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)

        result = await orch.run(
            TaskContract(goal="inspect existing state", task_type=TaskType.CODE_REVIEW)
        )
        await orch.close()

        assert result["outcome"] == "completed"
        assert (tmp_path / "existing.db").read_text() == "baseline"

    @pytest.mark.asyncio
    async def test_read_only_unchanged_preexisting_internal_worktree_state_passes(
        self, tmp_path
    ):
        existing = tmp_path / ".worktrees" / "pre-existing" / "state.txt"
        existing.parent.mkdir(parents=True)
        existing.write_text("baseline")
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)

        result = await orch.run(
            TaskContract(
                goal="inspect existing internal state",
                task_type=TaskType.CODE_REVIEW,
            )
        )
        await orch.close()

        assert result["outcome"] == "completed"
        assert existing.read_text() == "baseline"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mutation", ["root_untracked", "existing_ignored"])
    async def test_read_only_isolated_execution_rejects_root_mutation(
        self, tmp_path, mutation
    ):
        if mutation == "existing_ignored":
            (tmp_path / "existing.db").write_text("baseline")

        class RootMutatingAdapter(MockHermesAdapter):
            async def submit(self, task):
                target = (
                    tmp_path / "existing.db"
                    if mutation == "existing_ignored"
                    else tmp_path / "root-leak.txt"
                )
                target.write_text("runtime mutation")
                return await super().submit(task)

        adapter = RootMutatingAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)

        result = await orch.run(
            TaskContract(goal="inspect without root writes", task_type=TaskType.CODE_REVIEW)
        )
        await orch.close()

        assert result["outcome"] == "failed"
        assert result["verification_status"] == "fail"

    @pytest.mark.asyncio
    async def test_read_only_rejects_mutation_after_eval_before_success(self, tmp_path):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        real_gate = orch._eval_gate

        class PostEvalMutatingGate:
            async def run(self, *args):
                result = await real_gate.run(*args)
                (tmp_path / "post-eval.txt").write_text("late mutation")
                return result

        orch._eval_gate = PostEvalMutatingGate()
        result = await orch.run(
            TaskContract(goal="inspect without late writes", task_type=TaskType.CODE_REVIEW)
        )
        await orch.close()

        assert result["outcome"] == "failed"
        assert result["verification_status"] == "fail"

    @pytest.mark.asyncio
    async def test_delegated_parent_post_eval_mutation_fails_final_baseline(
        self, tmp_path
    ):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        real_gate = orch._eval_gate

        class PostEvalChildMutationGate:
            def __init__(self):
                self.calls = 0

            async def run(self, task, result, budget):
                self.calls += 1
                evaluation = await real_gate.run(task, result, budget)
                if self.calls == 3:
                    (tmp_path / "post-eval.db").write_text("late ignored mutation")
                return evaluation

        gate = PostEvalChildMutationGate()
        orch._eval_gate = gate
        orch._delegation_executor._eval_gate = gate
        task = TaskContract(
            goal="parallel read-only inspection",
            task_type=TaskType.PARALLEL_RESEARCH,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="left", goal="inspect left"),
                SubtaskSpec(id="right", goal="inspect right"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "failed"
        assert "final baseline mismatch" in result["detail"]
        assert result["verification_status"] == "fail"

    @pytest.mark.asyncio
    async def test_single_eval_exception_cleans_workspace_and_preserves_root(self, tmp_path):
        import os
        import subprocess

        repo = tmp_path / "eval-exception-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com",
        }
        (repo / "README.md").write_text("base")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
        subprocess.run(
            ["git", "commit", "-m", "base"], cwd=repo,
            check=True, capture_output=True, env=env,
        )
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        bootstrap = MockHermesAdapter()
        first = await build_orch(tmp_path, bootstrap)
        await first.close()

        class WritingAdapter(MockHermesAdapter):
            async def submit(self, task):
                (Path(task.workspace.path) / "change.txt").write_text("staged")
                return await super().submit(task)

        class ExplodingGate:
            async def run(self, *_args):
                raise RuntimeError("evaluation exploded")

        adapter = WritingAdapter()
        adapter.enqueue_scenario("pass")
        orch = await Orchestrator.build(
            runtime=adapter,
            db_path=str(tmp_path / "eval-exception.db"),
            repo_path=str(repo),
            policy_path=str(tmp_path / "default.yaml"),
        )
        orch._eval_gate = ExplodingGate()
        task = TaskContract(goal="staged change", task_type=TaskType.CODE_FIX)

        with pytest.raises(RuntimeError, match="evaluation exploded"):
            await orch.run(task)
        await orch.close()

        assert subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            check=True, capture_output=True, text=True,
        ).stdout.strip() == base
        assert not (repo / "change.txt").exists()
        assert not list((repo / ".worktrees").glob("**/single-*"))

    @pytest.mark.asyncio
    async def test_single_write_integrates_only_after_pass(self, tmp_path):
        import os
        import subprocess

        repo = tmp_path / "single-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com",
        }
        subprocess.run(["git", "commit", "--allow-empty", "-m", "base"], cwd=repo,
                       check=True, capture_output=True, env=env)
        bootstrap = MockHermesAdapter()
        first = await build_orch(tmp_path, bootstrap)
        await first.close()

        class WritingAdapter(MockHermesAdapter):
            async def submit(self, task):
                assert Path(task.workspace.path) != repo
                (Path(task.workspace.path) / "delivered.txt").write_text("ok")
                assert not (repo / "delivered.txt").exists()
                return await super().submit(task)

        adapter = WritingAdapter()
        adapter.enqueue_scenario("pass")
        orch = await Orchestrator.build(
            runtime=adapter, db_path=str(tmp_path / "single.db"),
            repo_path=str(repo), policy_path=str(tmp_path / "default.yaml"),
        )
        result = await orch.run(TaskContract(
            goal="write one file", task_type=TaskType.CODE_FIX,
            allowed_paths=["delivered.txt"],
        ))
        await orch.close()
        assert result["outcome"] == "completed"
        assert (repo / "delivered.txt").read_text() == "ok"

    @pytest.mark.asyncio
    async def test_single_completed_wrong_rolls_back_root(self, tmp_path):
        import os
        import subprocess

        repo = tmp_path / "wrong-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com",
        }
        subprocess.run(["git", "commit", "--allow-empty", "-m", "base"], cwd=repo,
                       check=True, capture_output=True, env=env)
        base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()
        bootstrap = MockHermesAdapter()
        first = await build_orch(tmp_path, bootstrap)
        await first.close()

        class WrongAdapter(MockHermesAdapter):
            async def submit(self, task):
                (Path(task.workspace.path) / "wrong.txt").write_text("wrong")
                return await super().submit(task)

        adapter = WrongAdapter()
        adapter.enqueue_scenario("pass")
        adapter.enqueue_scenario("pass")
        orch = await Orchestrator.build(
            runtime=adapter, db_path=str(tmp_path / "wrong.db"),
            repo_path=str(repo), policy_path=str(tmp_path / "default.yaml"),
        )
        result = await orch.run(TaskContract(
            goal="write only allowed", task_type=TaskType.CODE_FIX,
            allowed_paths=["allowed.txt"],
        ))
        await orch.close()
        final = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                               capture_output=True, text=True).stdout.strip()
        assert result["outcome"] == "failed"
        assert final == base
        assert not (repo / "wrong.txt").exists()

    @pytest.mark.asyncio
    async def test_delegation_delivers_worktree_artifacts_to_root_before_completed(self, tmp_path):
        import os
        import subprocess

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com",
        }
        (repo / "README.md").write_text("base")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
        subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True, env=env)

        policy = tmp_path / "delivery.yaml"
        policy.write_text((tmp_path / "default.yaml").read_text() if (tmp_path / "default.yaml").exists() else "")
        # build_orch writes its policy into tmp_path; use it once, then build a
        # separate orchestrator whose repository root stays clean.
        bootstrap = MockHermesAdapter()
        bootstrap.enqueue_scenario("pass")
        first = await build_orch(tmp_path, bootstrap)
        await first.close()
        policy.write_text((tmp_path / "default.yaml").read_text())

        class WritingAdapter(MockHermesAdapter):
            async def submit(self, task):
                subtask_id = task.context["_subtask_id"]
                artifact_dir = Path(task.workspace.path) / "artifacts"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / f"{subtask_id}.txt").write_text(subtask_id)
                return await super().submit(task)

        adapter = WritingAdapter()
        adapter.enqueue_scenario("pass", summary="auth")
        adapter.enqueue_scenario("pass", summary="db")
        orch = await Orchestrator.build(
            runtime=adapter,
            db_path=str(tmp_path / "delivery.db"),
            repo_path=str(repo),
            policy_path=str(policy),
        )
        task = TaskContract(
            goal="deliver explicit work",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            allowed_paths=["artifacts/**"],
            subtasks=[
                SubtaskSpec(id="auth", goal="write auth", allowed_paths=["artifacts/**"]),
                SubtaskSpec(id="db", goal="write db", allowed_paths=["artifacts/**"]),
            ],
        )

        result = await orch.run(task)
        records = list(orch._wm.list_records())
        await orch.close()

        assert result["outcome"] == "completed"
        assert (repo / "artifacts" / "auth.txt").read_text() == "auth"
        assert (repo / "artifacts" / "db.txt").read_text() == "db"
        assert all(record.status.value == "cleaned" for record in records)

    @pytest.mark.asyncio
    async def test_integrated_eval_failure_rolls_root_back(self, tmp_path):
        import os
        import subprocess

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com",
        }
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "base"],
            cwd=repo, check=True, capture_output=True, env=env,
        )
        base_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()

        bootstrap = MockHermesAdapter()
        first = await build_orch(tmp_path, bootstrap)
        await first.close()
        policy = tmp_path / "default.yaml"

        class WritingAdapter(MockHermesAdapter):
            async def submit(self, task):
                target = Path(task.workspace.path) / "outside" / "bad.txt"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("child-valid, parent-invalid")
                return await super().submit(task)

        adapter = WritingAdapter()
        adapter.enqueue_scenario("pass")
        orch = await Orchestrator.build(
            runtime=adapter,
            db_path=str(tmp_path / "rollback.db"),
            repo_path=str(repo),
            policy_path=str(policy),
        )
        task = TaskContract(
            goal="attempt invalid integrated delivery",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            allowed_paths=["allowed/**"],
            subtasks=[
                SubtaskSpec(id="bad", goal="write outside parent scope", allowed_paths=["outside/**"]),
            ],
        )

        result = await orch.run(task)
        records = list(orch._wm.list_records())
        await orch.close()
        final_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()

        assert result["outcome"] == "failed"
        assert final_sha == base_sha
        assert not (repo / "outside" / "bad.txt").exists()
        assert all(record.status.value == "cleaned" for record in records)
        assert not list((repo / ".worktrees").glob("**/child-*"))

    @pytest.mark.asyncio
    async def test_simple_task_completes(self, tmp_path):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", files_changed=["src/auth/login.py"], summary="fixed login")

        async with await build_orch(tmp_path, adapter) as orch:
            task = TaskContract(
                goal="fix the login timeout bug",
                task_type=TaskType.CODE_FIX,
                complexity=1,
                allowed_paths=["src/auth/**"],
            )
            result = await orch.run(task)

        assert result["outcome"] == "completed"
        assert result["route"] == "single"
        assert result["retry_count"] == 0
        assert result["eval"]["overall"] == "pass"

    @pytest.mark.asyncio
    async def test_multi_file_task_routes_delegation(self, tmp_path):
        # Need a real git repo so WorkspaceManager can create worktrees
        import subprocess
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        (tmp_path / ".gitignore").write_text("default.yaml\ntest.db*\n.worktrees/\n")
        subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"],
                       check=True, capture_output=True,
                       env={**__import__("os").environ,
                            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t.com",
                            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t.com"})
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario(
            "pass",
            files_changed=["src/auth/login.py", "src/payments/charge.py"],
            summary="refactored both modules",
        )

        async with await build_orch(tmp_path, adapter) as orch:
            task = TaskContract(
                goal="refactor auth and payments independently",
                task_type=TaskType.MULTI_FILE_REFACTOR,
                complexity=4,
                allowed_paths=["src/auth/**", "src/payments/**"],
            )
            result = await orch.run(task)

        assert result["outcome"] == "completed"
        assert result["route"] == "delegation"

    @pytest.mark.asyncio
    async def test_token_usage_recorded(self, tmp_path):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass", input_tokens=2000, output_tokens=800)

        async with await build_orch(tmp_path, adapter) as orch:
            task = TaskContract(goal="summarise the codebase", complexity=1)
            result = await orch.run(task)

        assert result["usage"]["input_tokens"] == 2000
        assert result["usage"]["output_tokens"] == 800
        assert result["usage"]["total_tokens"] == 2800

    @pytest.mark.asyncio
    async def test_path_violation_fails_eval(self, tmp_path):
        """Agent writes outside allowed_paths → eval FAIL → retry → fail again."""
        adapter = MockHermesAdapter()
        # Both submit attempts return a path violation
        adapter.enqueue_scenario("pass", files_changed=["src/payments/charge.py"])  # 1st attempt
        adapter.enqueue_scenario("pass", files_changed=["src/payments/charge.py"])  # retry

        async with await build_orch(tmp_path, adapter) as orch:
            task = TaskContract(
                goal="fix auth only",
                complexity=1,
                allowed_paths=["src/auth/**"],
            )
            result = await orch.run(task)

        # eval fails (path violation) → retry → fails again → abandoned
        assert result["outcome"] == "failed"
        assert result["eval"]["overall"] == "fail"
        assert "paths" in result["eval"]["failed_checks"]

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, tmp_path):
        adapter = MockHermesAdapter()
        # First attempt: path violation
        adapter.enqueue_scenario("pass", files_changed=["src/payments/charge.py"])
        # Retry: correct paths
        adapter.enqueue_scenario("pass", files_changed=["src/auth/login.py"])

        async with await build_orch(tmp_path, adapter) as orch:
            task = TaskContract(
                goal="fix auth login",
                complexity=1,
                allowed_paths=["src/auth/**"],
            )
            result = await orch.run(task)

        assert result["outcome"] == "completed"
        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_agent_failure_propagates(self, tmp_path):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("fail", error_message="tool crashed")
        adapter.enqueue_scenario("fail", error_message="tool crashed again")  # retry also fails

        async with await build_orch(tmp_path, adapter) as orch:
            task = TaskContract(goal="do something risky", complexity=1)
            result = await orch.run(task)

        assert result["outcome"] == "failed"

    @pytest.mark.asyncio
    async def test_sequential_dep_forces_single(self, tmp_path):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")

        async with await build_orch(tmp_path, adapter) as orch:
            # "then" keyword → sequential dependency → forced single
            task = TaskContract(
                goal="first fix the bug then deploy to staging",
                task_type=TaskType.MULTI_FILE_REFACTOR,
                complexity=4,
                allowed_paths=["src/**"],
            )
            result = await orch.run(task)

        assert result["route"] == "single"

    @pytest.mark.asyncio
    async def test_telemetry_written_to_db(self, tmp_path):
        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")

        orch = await build_orch(tmp_path, adapter)
        task = TaskContract(goal="fix the bug", complexity=1)
        await orch.run(task)

        # Verify routing_decisions table was written
        rows = await orch._db.get_routing_history()
        assert len(rows) >= 1
        row = rows[0]
        assert row["task_id"] == task.id
        assert row["route"] in ("single", "delegation")
        assert row["policy_version"] == "routing-v1.0"

        await orch.close()

    @pytest.mark.asyncio
    async def test_budget_exceeded_abandons_task(self, tmp_path):
        """If calls_used already exceeds max at check time, task is abandoned."""
        adapter = MockHermesAdapter()
        # Not even consumed — budget check fires before submit

        policy = tmp_path / "tight.yaml"
        policy.write_text(
            """
policy_version: "routing-v1.0"
routing:
  delegation:
    min_independent_subtasks: 2
    min_estimated_input_tokens: 8000
    allowed_task_types: []
  single:
    max_complexity: 5
    max_affected_modules: 10
  constraints:
    sequential_dependency_forces_single: false
budget:
  max_children: 2
  max_depth: 1
  max_retries: 1
  max_total_calls: 0   # zero budget → immediate block
  require_approval_above_calls: 1
approval:
  always_require: []
  require_for_risk_levels: [4]
worktree:
  base_path: ".worktrees"
  readonly_task_types: []
"""
        )
        orch = await Orchestrator.build(
            runtime=adapter,
            db_path=str(tmp_path / "budget_test.db"),
            repo_path=str(tmp_path),
            policy_path=str(policy),
        )
        task = TaskContract(goal="expensive task", complexity=3)
        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "abandoned"

    @pytest.mark.asyncio
    async def test_post_submit_failure_quiesces_before_workspace_cleanup(
        self, tmp_path, monkeypatch
    ):
        from contracts.result import AgentResult, RunStatus

        class TrackingAdapter(MockHermesAdapter):
            def __init__(self):
                super().__init__()
                self.order = []
                self.workspace_path = None

            async def submit(self, task):
                self.workspace_path = Path(task.workspace.path)
                return await super().submit(task)

            async def cancel(self, handle):
                assert self.workspace_path.exists()
                self.order.append("cancel")
                await super().cancel(handle)

            async def wait(self, handle):
                assert self.workspace_path.exists()
                self.order.append("wait")
                return AgentResult(
                    run_id=handle.run_id,
                    task_id=handle.task_id,
                    status=RunStatus.CANCELLED,
                )

            async def quiesce(self, handle):
                assert self.workspace_path.exists()
                self.order.append("quiesce")

        adapter = TrackingAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        real_record = orch._tel.record

        async def fail_after_submit(task_id, event_type, payload, run_id=None):
            if event_type == "task_submitted":
                raise RuntimeError("injected post-submit failure")
            return await real_record(task_id, event_type, payload, run_id)

        monkeypatch.setattr(orch._tel, "record", fail_after_submit)
        task = TaskContract(goal="quiesce before cleanup", task_type=TaskType.CODE_FIX)

        with pytest.raises(RuntimeError, match="post-submit failure"):
            await orch.run(task)
        record = orch._sm.get(task.id)
        await orch.close()

        assert adapter.order == ["cancel", "wait", "quiesce"]
        assert not adapter.workspace_path.exists()
        assert record is not None
        assert record.run_id in adapter._runs

    @pytest.mark.asyncio
    async def test_event_exception_quiesces_before_workspace_cleanup(self, tmp_path):
        from contracts.result import AgentResult, RunStatus

        class EventFailureAdapter(MockHermesAdapter):
            def __init__(self):
                super().__init__()
                self.order = []
                self.workspace_path = None

            async def submit(self, task):
                self.workspace_path = Path(task.workspace.path)
                return await super().submit(task)

            async def events(self, handle, *, after=None):
                self.order.append("events")
                raise RuntimeError("injected event failure")
                if False:
                    yield

            async def cancel(self, handle):
                assert self.workspace_path.exists()
                self.order.append("cancel")
                await super().cancel(handle)

            async def wait(self, handle):
                assert self.workspace_path.exists()
                self.order.append("wait")
                return AgentResult(
                    run_id=handle.run_id,
                    task_id=handle.task_id,
                    status=RunStatus.CANCELLED,
                )

            async def quiesce(self, handle):
                assert self.workspace_path.exists()
                self.order.append("quiesce")

        adapter = EventFailureAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        task = TaskContract(goal="event failure", task_type=TaskType.CODE_FIX)

        with pytest.raises(RuntimeError, match="event failure"):
            await orch.run(task)
        record = orch._sm.get(task.id)
        await orch.close()

        assert adapter.order == ["events", "cancel", "wait", "quiesce"]
        assert not adapter.workspace_path.exists()
        assert record is not None
        assert record.run_id in adapter._runs

    @pytest.mark.asyncio
    async def test_unconfirmed_quiescence_quarantines_workspace(self, tmp_path):
        class UnconfirmedAdapter(MockHermesAdapter):
            def __init__(self):
                super().__init__()
                self.order = []
                self.workspace_path = None

            async def submit(self, task):
                self.workspace_path = Path(task.workspace.path)
                return await super().submit(task)

            async def events(self, handle, *, after=None):
                raise RuntimeError("events unavailable")
                if False:
                    yield

            async def cancel(self, handle):
                self.order.append("cancel")
                raise RuntimeError("cancel unavailable")

            async def wait(self, handle):
                self.order.append("wait")
                raise RuntimeError("terminal confirmation unavailable")

        adapter = UnconfirmedAdapter()
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        task = TaskContract(goal="quarantine active run", task_type=TaskType.CODE_FIX)

        with pytest.raises(RuntimeError, match="quiescence"):
            await orch.run(task)
        record = orch._sm.get(task.id)
        await orch.close()

        assert adapter.order == ["cancel", "wait"]
        assert adapter.workspace_path.exists()
        assert record is not None
        assert record.run_id in adapter._runs

    @pytest.mark.asyncio
    async def test_delegated_event_failures_quiesce_before_cleanup(self, tmp_path):
        from contracts.result import AgentResult, RunStatus
        from orchestrator.budget import BudgetConfig

        class DelegatedFailureAdapter(MockHermesAdapter):
            def __init__(self):
                super().__init__()
                self.order = []
                self.workspace_paths = {}

            async def submit(self, task):
                handle = await super().submit(task)
                self.workspace_paths[handle.run_id] = Path(task.workspace.path)
                return handle

            async def events(self, handle, *, after=None):
                self.order.append((handle.run_id, "events"))
                raise RuntimeError("delegated events failed")
                if False:
                    yield

            async def cancel(self, handle):
                assert self.workspace_paths[handle.run_id].exists()
                self.order.append((handle.run_id, "cancel"))
                await super().cancel(handle)

            async def wait(self, handle):
                assert self.workspace_paths[handle.run_id].exists()
                self.order.append((handle.run_id, "wait"))
                return AgentResult(
                    run_id=handle.run_id,
                    task_id=handle.task_id,
                    status=RunStatus.CANCELLED,
                )

            async def quiesce(self, handle):
                assert self.workspace_paths[handle.run_id].exists()
                self.order.append((handle.run_id, "quiesce"))

        adapter = DelegatedFailureAdapter()
        adapter.enqueue_scenario("pass")
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        orch._budget_config = BudgetConfig(max_children=2, max_retries=0)
        task = TaskContract(
            goal="delegated lifecycle failure",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "failed"
        for run_id in adapter._runs:
            assert [step for observed, step in adapter.order if observed == run_id] == [
                "events",
                "cancel",
                "wait",
                "quiesce",
            ]
            assert not adapter.workspace_paths[run_id].exists()

    @pytest.mark.asyncio
    async def test_delegated_unconfirmed_quiescence_quarantines_worktrees(self, tmp_path):
        from orchestrator.budget import BudgetConfig

        class UnconfirmedDelegatedAdapter(MockHermesAdapter):
            def __init__(self):
                super().__init__()
                self.order = []
                self.workspace_paths = {}

            async def submit(self, task):
                handle = await super().submit(task)
                self.workspace_paths[handle.run_id] = Path(task.workspace.path)
                return handle

            async def events(self, handle, *, after=None):
                raise RuntimeError("delegated events unavailable")
                if False:
                    yield

            async def cancel(self, handle):
                self.order.append((handle.run_id, "cancel"))
                raise RuntimeError("delegated cancel unavailable")

            async def wait(self, handle):
                self.order.append((handle.run_id, "wait"))
                raise RuntimeError("delegated terminal unavailable")

        adapter = UnconfirmedDelegatedAdapter()
        adapter.enqueue_scenario("pass")
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        orch._budget_config = BudgetConfig(max_children=2, max_retries=0)
        task = TaskContract(
            goal="quarantine delegated runs",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "failed"
        assert "quarantined" in result["detail"]
        assert len(adapter._runs) == 2
        for run_id in adapter._runs:
            assert [step for observed, step in adapter.order if observed == run_id] == [
                "cancel",
                "wait",
            ]
            assert adapter.workspace_paths[run_id].exists()

    @pytest.mark.asyncio
    async def test_delegated_known_run_ids_survive_later_usage_failures(self, tmp_path):
        from orchestrator.budget import BudgetConfig

        class UsageFailureAdapter(MockHermesAdapter):
            async def usage(self, handle):
                raise RuntimeError(f"usage unavailable for {handle.run_id}")

        adapter = UsageFailureAdapter()
        adapter.enqueue_scenario("pass")
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        orch._budget_config = BudgetConfig(max_children=2, max_retries=0)
        task = TaskContract(
            goal="preserve submitted child identities",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "failed"
        assert {item["run_id"] for item in result["child_runs"]} == set(adapter._runs)

    @pytest.mark.asyncio
    async def test_delegated_usage_persistence_failure_preserves_child_attempt_ids(
        self, tmp_path
    ):
        from orchestrator.budget import BudgetConfig

        adapter = MockHermesAdapter()
        adapter.enqueue_scenario("pass")
        adapter.enqueue_scenario("pass")
        orch = await build_orch(tmp_path, adapter)
        orch._budget_config = BudgetConfig(max_children=2, max_retries=0)

        async def fail_usage_persistence(*_args, **_kwargs):
            raise RuntimeError("usage persistence unavailable")

        orch._db.append_usage = fail_usage_persistence
        task = TaskContract(
            goal="retain child identities after parent persistence failure",
            task_type=TaskType.MULTI_FILE_REFACTOR,
            complexity=4,
            subtasks=[
                SubtaskSpec(id="one", goal="child one"),
                SubtaskSpec(id="two", goal="child two"),
            ],
        )

        result = await orch.run(task)
        await orch.close()

        assert result["outcome"] == "failed"
        assert result["run_id"] is None
        assert {item["run_id"] for item in result["child_runs"]} == set(adapter._runs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_status", "expected_outcome"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("timeout", "timeout"),
    ],
)
async def test_runtime_quiesces_before_workspace_cleanup_for_all_terminal_paths(
    tmp_path, monkeypatch, terminal_status, expected_outcome
):
    from contracts.result import AgentEvent, AgentResult, RunStatus, Usage
    from orchestrator.workspace import ExecutionWorkspace

    order = []

    class TerminalAdapter(MockHermesAdapter):
        async def events(self, handle, *, after=None):  # noqa: ANN001
            event_type = "completed" if terminal_status == "completed" else "error"
            yield AgentEvent(
                id="terminal",
                run_id=handle.run_id,
                type=event_type,
                payload={},
            )

        async def wait(self, handle):  # noqa: ANN001
            return AgentResult(
                run_id=handle.run_id,
                task_id=handle.task_id,
                status=RunStatus(terminal_status),
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                files_changed=[],
                summary="OK" if terminal_status == "completed" else None,
                error=None if terminal_status == "completed" else terminal_status,
            )

        async def quiesce(self, handle):  # noqa: ANN001
            order.append(("quiesce", handle.run_id))

    def rollback(workspace):  # noqa: ANN001
        order.append(("rollback", None))
        workspace.cleaned = True

    def cleanup(workspace):  # noqa: ANN001
        order.append(("cleanup", None))
        workspace.cleaned = True

    monkeypatch.setattr(ExecutionWorkspace, "rollback", rollback)
    monkeypatch.setattr(ExecutionWorkspace, "cleanup", cleanup)
    adapter = TerminalAdapter()
    orch = await build_orch(tmp_path, adapter)
    try:
        result = await orch.run(TaskContract(goal="terminal lifecycle"))
    finally:
        await orch.close()

    cleanup_index = next(
        index for index, (name, _) in enumerate(order) if name in {"rollback", "cleanup"}
    )
    quiesce_index = next(index for index, (name, _) in enumerate(order) if name == "quiesce")
    assert quiesce_index < cleanup_index
    assert result["outcome"] == expected_outcome
    assert result["observed"]["runtime_status"] == terminal_status


@pytest.mark.asyncio
async def test_quiescence_failure_preserves_runtime_terminal_truth_and_skips_cleanup(
    tmp_path, monkeypatch
):
    from contracts.result import AgentEvent, AgentResult, RunStatus, Usage
    from orchestrator.workspace import ExecutionWorkspace

    cleanup_calls = []

    class QuiescenceFailureAdapter(MockHermesAdapter):
        async def events(self, handle, *, after=None):  # noqa: ANN001
            yield AgentEvent(id="terminal", run_id=handle.run_id, type="completed", payload={})

        async def wait(self, handle):  # noqa: ANN001
            return AgentResult(
                run_id=handle.run_id,
                task_id=handle.task_id,
                status=RunStatus.COMPLETED,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                files_changed=[],
                summary="OK",
            )

        async def quiesce(self, handle):  # noqa: ANN001
            raise RuntimeError("run resources still own workspace")

    def forbidden_cleanup(_workspace):  # noqa: ANN001
        cleanup_calls.append("called")

    monkeypatch.setattr(ExecutionWorkspace, "rollback", forbidden_cleanup)
    monkeypatch.setattr(ExecutionWorkspace, "cleanup", forbidden_cleanup)
    adapter = QuiescenceFailureAdapter()
    orch = await build_orch(tmp_path, adapter)
    try:
        result = await orch.run(TaskContract(goal="quiescence failure"))
    finally:
        await orch.close()

    assert result["outcome"] == "failed"
    assert result["observed"]["runtime_status"] == "completed"
    assert "quiescence" in result["detail"]
    assert cleanup_calls == []

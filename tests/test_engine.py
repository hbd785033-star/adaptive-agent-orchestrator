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
        assert all(record.status.value == "abandoned" for record in records)

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

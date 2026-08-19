from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from model_council.inventory import ModelSpec

from adapters.mock import MockHermesAdapter
from adapters.runtime import RuntimeCapabilities
from contracts.runtime_health import RuntimeHealth
from contracts.runtime_selection import RuntimeSelectionPolicy
from contracts.task import RiskLevel, TaskContract, TaskType
from orchestrator.engine import Orchestrator
from orchestrator.runtime_registry import RuntimeRegistry


class NoLifecycleRuntime(MockHermesAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.submit_calls = 0

    async def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    async def submit(self, task):  # noqa: ANN001
        self.submit_calls += 1
        return await super().submit(task)


class TrackingRuntime(MockHermesAdapter):
    def __init__(
        self,
        *,
        connect_error: Exception | None = None,
        disconnect_error: Exception | None = None,
        submit_error: Exception | None = None,
        capabilities: RuntimeCapabilities | None = None,
    ) -> None:
        super().__init__()
        self.connect_error = connect_error
        self.disconnect_error = disconnect_error
        self.submit_error = submit_error
        self.reported_capabilities = capabilities or RuntimeCapabilities()
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.submit_calls = 0
        self.event_calls = 0
        self.wait_calls = 0
        self.usage_calls = 0
        self.cancel_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error

    async def capabilities(self) -> RuntimeCapabilities:
        return self.reported_capabilities

    async def submit(self, task):  # noqa: ANN001
        self.submit_calls += 1
        if self.submit_error is not None:
            raise self.submit_error
        return await super().submit(task)

    async def events(self, handle, *, after=None):  # noqa: ANN001
        self.event_calls += 1
        async for event in super().events(handle, after=after):
            yield event

    async def wait(self, handle):  # noqa: ANN001
        self.wait_calls += 1
        return await super().wait(handle)

    async def usage(self, handle):  # noqa: ANN001
        self.usage_calls += 1
        return await super().usage(handle)

    async def cancel(self, handle):  # noqa: ANN001
        self.cancel_calls += 1
        await super().cancel(handle)


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    policy = tmp_path / "policy.yaml"
    policy.write_text(
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
    return repo, policy


@pytest.fixture
def inventory() -> list[ModelSpec]:
    return [
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


def task() -> TaskContract:
    return TaskContract(
        task_type=TaskType.GENERAL,
        goal="return deterministic output",
        complexity=1,
        risk=RiskLevel.LOW,
    )


def registry(hermes: TrackingRuntime, runtime_b: TrackingRuntime) -> RuntimeRegistry:
    return RuntimeRegistry(entries=[("hermes", hermes), ("runtime_b", runtime_b)])


def policy(*priority: str) -> RuntimeSelectionPolicy:
    return RuntimeSelectionPolicy(
        policy_version="runtime-selection-e2e3a-v1",
        runtime_priority=priority,
        allow_degraded_fallback=False,
    )


async def build_multi(
    git_repo: tuple[Path, Path],
    tmp_path: Path,
    inventory: list[ModelSpec],
    hermes: TrackingRuntime,
    runtime_b: TrackingRuntime,
    *,
    selection_policy: RuntimeSelectionPolicy,
) -> Orchestrator:
    repo, policy_path = git_repo
    return await Orchestrator.build(
        runtime_registry=registry(hermes, runtime_b),
        runtime_selection_policy=selection_policy,
        model_discoverer=lambda: inventory,
        db_path=str(tmp_path / "multi.db"),
        repo_path=str(repo),
        policy_path=str(policy_path),
    )


@pytest.mark.asyncio
async def test_protocol_only_registry_runtime_cannot_bypass_planning(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    repo, policy_path = git_repo
    runtime_b = NoLifecycleRuntime()
    orchestrator = await Orchestrator.build(
        runtime_registry=RuntimeRegistry(entries=[("runtime_b", runtime_b)]),
        runtime_selection_policy=policy("runtime_b"),
        model_discoverer=lambda: inventory,
        db_path=str(tmp_path / "protocol-only.db"),
        repo_path=str(repo),
        policy_path=str(policy_path),
    )
    try:
        result = await orchestrator.run(task())
    finally:
        await orchestrator.close()
    assert result["outcome"] == "failed"
    assert "runtime health is unavailable" in result["detail"].lower()
    assert runtime_b.submit_calls == 0


@pytest.mark.asyncio
async def test_health_identity_mismatch_fails_closed(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    repo, policy_path = git_repo
    with pytest.raises(ValueError, match="runtime health identity must match"):
        await Orchestrator.build(
            runtime_registry=RuntimeRegistry(
                entries=[("hermes", TrackingRuntime()), ("runtime_b", TrackingRuntime())]
            ),
            runtime_selection_policy=policy("hermes", "runtime_b"),
            runtime_health_by_runtime={
                "runtime_b": RuntimeHealth(runtime="wrong")
            },
            model_discoverer=lambda: inventory,
            db_path=str(tmp_path / "health-mismatch.db"),
            repo_path=str(repo),
            policy_path=str(policy_path),
        )


@pytest.mark.asyncio
async def test_unknown_plan_executor_fails_before_any_submit(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    hermes = TrackingRuntime()
    runtime_b = TrackingRuntime()
    orchestrator = await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy("runtime_b", "hermes"),
    )

    original = orchestrator._prepare_hmc_planning

    async def forged_plan(task_contract):  # noqa: ANN001
        evidence, result = await original(task_contract)
        forged = result.plan.model_copy(update={"executor": "unknown"})
        selection = result.selection.model_copy(update={"selected_runtime": "unknown"})
        return evidence, result.model_copy(update={"plan": forged, "selection": selection})

    orchestrator._prepare_hmc_planning = forged_plan
    with pytest.raises(KeyError, match="unknown runtime identity: unknown"):
        await orchestrator.run(task())
    await orchestrator.close()

    assert hermes.submit_calls == runtime_b.submit_calls == 0


@pytest.mark.asyncio
async def test_plan_selection_mismatch_fails_before_any_submit(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    hermes = TrackingRuntime()
    runtime_b = TrackingRuntime()
    orchestrator = await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy("runtime_b", "hermes"),
    )
    original = orchestrator._prepare_hmc_planning

    async def mismatched_plan(task_contract):  # noqa: ANN001
        evidence, result = await original(task_contract)
        forged = result.plan.model_copy(update={"executor": "hermes"})
        return evidence, result.model_copy(update={"plan": forged})

    orchestrator._prepare_hmc_planning = mismatched_plan
    with pytest.raises(RuntimeError, match="must match runtime selection"):
        await orchestrator.run(task())
    await orchestrator.close()

    assert hermes.submit_calls == runtime_b.submit_calls == 0


@pytest.mark.asyncio
async def test_multi_runtime_requires_explicit_selection_policy(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    repo, policy_path = git_repo
    with pytest.raises(ValueError, match="multiple runtimes require explicit selection policy"):
        await Orchestrator.build(
            runtime_registry=registry(TrackingRuntime(), TrackingRuntime()),
            model_discoverer=lambda: inventory,
            db_path=str(tmp_path / "missing-policy.db"),
            repo_path=str(repo),
            policy_path=str(policy_path),
        )


@pytest.mark.asyncio
async def test_runtime_and_registry_are_mutually_exclusive(
    git_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    repo, policy_path = git_repo
    with pytest.raises(ValueError, match="runtime and runtime_registry are mutually exclusive"):
        await Orchestrator.build(
            runtime=TrackingRuntime(),
            runtime_registry=RuntimeRegistry(entries=[("runtime_b", TrackingRuntime())]),
            db_path=str(tmp_path / "conflict.db"),
            repo_path=str(repo),
            policy_path=str(policy_path),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hermes_error", "runtime_b_error", "priority", "selected"),
    [
        (None, RuntimeError("b unavailable"), ("runtime_b", "hermes"), "hermes"),
        (RuntimeError("hermes unavailable"), None, ("hermes", "runtime_b"), "runtime_b"),
        (None, None, ("runtime_b", "hermes"), "runtime_b"),
        (None, None, ("hermes", "runtime_b"), "hermes"),
    ],
)
async def test_per_runtime_health_and_policy_select_exact_runtime(
    git_repo: tuple[Path, Path],
    tmp_path: Path,
    inventory: list[ModelSpec],
    hermes_error: Exception | None,
    runtime_b_error: Exception | None,
    priority: tuple[str, str],
    selected: str,
) -> None:
    hermes = TrackingRuntime(connect_error=hermes_error)
    runtime_b = TrackingRuntime(connect_error=runtime_b_error)
    hermes.enqueue_scenario("pass", summary="hermes")
    runtime_b.enqueue_scenario("pass", summary="runtime-b")
    async with await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy(*priority),
    ) as orchestrator:
        result = await orchestrator.run(task())

    assessments = {
        item["runtime"]: item for item in result["planned"]["candidate_assessments"]
    }
    assert set(assessments) == {"hermes", "runtime_b"}
    assert result["planned"]["runtime_selection"]["selected_runtime"] == selected
    assert result["planned"]["runtime_plan"]["executor"] == selected
    expected_runtime = runtime_b if selected == "runtime_b" else hermes
    other_runtime = hermes if selected == "runtime_b" else runtime_b
    assert expected_runtime.submit_calls == 1
    assert other_runtime.submit_calls == 0
    assert result["observed"]["runtime_adapter"] == selected
    assert result["observed"]["runtime_adapter_invoked"] is True


@pytest.mark.asyncio
async def test_all_unavailable_produces_no_plan_and_no_submit(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    hermes = TrackingRuntime(connect_error=RuntimeError("hermes down"))
    runtime_b = TrackingRuntime(connect_error=RuntimeError("b down"))
    async with await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy("hermes", "runtime_b"),
    ) as orchestrator:
        result = await orchestrator.run(task())

    assert result["outcome"] == "failed"
    assert result["run_id"] is None
    assert result["planned"]["runtime_selection"]["selected_runtime"] is None
    assert result["planned"]["runtime_plan"] is None
    assert hermes.submit_calls == runtime_b.submit_calls == 0


@pytest.mark.asyncio
async def test_selected_submit_failure_never_falls_back(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    hermes = TrackingRuntime()
    runtime_b = TrackingRuntime(submit_error=RuntimeError("runtime-b submit failed"))
    async with await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy("runtime_b", "hermes"),
    ) as orchestrator:
        with pytest.raises(RuntimeError, match="runtime-b submit failed"):
            await orchestrator.run(task())

    assert runtime_b.submit_calls == 1
    assert hermes.submit_calls == 0


@pytest.mark.asyncio
async def test_cancellation_and_terminal_confirmation_use_selected_runtime(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    hermes = TrackingRuntime()
    runtime_b = TrackingRuntime()
    runtime_b.enqueue_scenario("approval_required")
    async with await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy("runtime_b", "hermes"),
    ) as orchestrator:
        result = await orchestrator.run(task())

    assert result["outcome"] == "failed"
    assert runtime_b.submit_calls == 1
    assert runtime_b.cancel_calls == 1
    assert runtime_b.wait_calls >= 1
    assert hermes.submit_calls == hermes.cancel_calls == hermes.wait_calls == 0


@pytest.mark.asyncio
async def test_close_attempts_all_disconnects_and_isolates_failure(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    hermes = TrackingRuntime(disconnect_error=RuntimeError("hermes close failed"))
    runtime_b = TrackingRuntime()
    orchestrator = await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy("hermes", "runtime_b"),
    )

    await orchestrator.close()

    assert hermes.disconnect_calls == 1
    assert runtime_b.disconnect_calls == 1


@pytest.mark.asyncio
async def test_explicit_inventory_does_not_call_runtime_inventory_provider(
    git_repo: tuple[Path, Path], tmp_path: Path, inventory: list[ModelSpec]
) -> None:
    class InventoryRuntime(TrackingRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.inventory_calls = 0

        async def model_inventory_payload(self):  # noqa: ANN001
            self.inventory_calls += 1
            raise AssertionError("explicit inventory must take precedence")

    hermes = InventoryRuntime()
    runtime_b = TrackingRuntime()
    hermes.enqueue_scenario("pass")
    async with await build_multi(
        git_repo,
        tmp_path,
        inventory,
        hermes,
        runtime_b,
        selection_policy=policy("hermes", "runtime_b"),
    ) as orchestrator:
        result = await orchestrator.run(task())

    assert result["outcome"] == "completed"
    assert hermes.inventory_calls == 0

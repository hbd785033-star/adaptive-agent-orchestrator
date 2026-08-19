from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from model_council.inventory import ModelSpec
from model_council.inventory import discover_models as hmc_discover_models

from adapters.mock import MockHermesAdapter
from adapters.runtime import RuntimeCapabilities
from contracts.task import RiskLevel, SubtaskSpec, TaskContract, TaskType
from evals.gate import DeterministicEvalGate
from orchestrator.budget import ApprovalGate, BudgetConfig
from orchestrator.engine import Orchestrator
from orchestrator.profiler import TaskProfiler
from orchestrator.router import RuleRouter
from orchestrator.state_machine import StateMachine
from orchestrator.workspace import WorkspaceManager
from storage.database import Database
from telemetry.events import TelemetryRecorder


class ControlledRuntime(MockHermesAdapter):
    """Deterministic runtime for production-wiring mechanics only."""

    def __init__(self) -> None:
        super().__init__()
        self.connected = False
        self.submit_calls = 0

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def submit(self, task):  # noqa: ANN001
        assert self.connected
        self.submit_calls += 1
        return await super().submit(task)


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


class InventoryRuntime(ControlledRuntime):
    """Deterministic runtime exposing the Hermes picker inventory seam."""

    def __init__(self, payload) -> None:  # noqa: ANN001
        super().__init__()
        self.payload = payload
        self.inventory_calls = 0

    async def model_inventory_payload(self):  # noqa: ANN001
        self.inventory_calls += 1
        return self.payload

    async def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            streaming_events=True,
            mid_run_steer=False,
            native_delegation=False,
            cancellation=True,
            session_resume=False,
            max_concurrent_runs=8,
            filesystem_read=True,
        )


class NoDirectDiscoveryRuntime(InventoryRuntime):
    """Payload runtime whose test fails if default HMC discovery is invoked."""


@pytest.fixture
def filesystem_read_task() -> TaskContract:
    return TaskContract(
        task_type=TaskType.CODE_REVIEW,
        goal=(
            "Read README.md from the permitted workspace and return exactly the "
            "first non-empty line after the Markdown H1 heading, verbatim."
        ),
        allowed_paths=["README.md"],
        output_schema=[],
        risk=RiskLevel.LOW,
        complexity=1,
        subtasks=[],
    )


def picker_payload() -> dict:
    return {
        "provider": "openai",
        "model": "gpt-test",
        "providers": [
            {
                "slug": "openai",
                "authenticated": True,
                "models": ["gpt-test"],
                "capabilities": {"gpt-test": {"reasoning": True, "fast": True}},
            }
        ],
    }


@pytest.mark.asyncio
async def test_runtime_picker_payload_drives_real_hmc_planning_without_hermes_import(
    git_repo: tuple[Path, Path], tmp_path: Path, filesystem_read_task: TaskContract, monkeypatch
) -> None:
    repo, policy = git_repo
    runtime = InventoryRuntime(picker_payload())
    runtime.enqueue_scenario("pass", summary="planned")

    def forbidden_default_discovery(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("default Hermes-module discovery must not be invoked")

    monkeypatch.setattr(
        "model_council.inventory._hermes_payload", forbidden_default_discovery
    )
    async with await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "inventory.db"),
        repo_path=str(repo),
        policy_path=str(policy),
    ) as orchestrator:
        result = await orchestrator.run(filesystem_read_task)

    assert runtime.inventory_calls == 1
    assert result["outcome"] == "completed"
    assert result["planned"]["hmc"]["request_type"] == "PlannerRequest"
    assert result["planned"]["hmc"]["recommendation_type"] == "PlannerRecommendation"
    assert result["planned"]["runtime_plan"]["executor"] == "hermes"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"providers": []}, {"providers": None}])
async def test_inventory_failure_is_before_submission(
    git_repo: tuple[Path, Path], tmp_path: Path, filesystem_read_task: TaskContract, payload
) -> None:
    repo, policy = git_repo
    runtime = InventoryRuntime(payload)
    runtime.enqueue_scenario("pass", summary="must not execute")
    async with await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "inventory-fail.db"),
        repo_path=str(repo),
        policy_path=str(policy),
    ) as orchestrator:
        result = await orchestrator.run(filesystem_read_task)

    assert runtime.inventory_calls == 1
    assert runtime.submit_calls == 0
    assert result["outcome"] == "failed"
    assert result["run_id"] is None
    assert "planning" in result["detail"].lower()


@pytest.mark.asyncio
async def test_explicit_model_discoverer_precedes_runtime_inventory(
    git_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    repo, policy = git_repo
    runtime = InventoryRuntime({"providers": []})
    runtime.enqueue_scenario("pass", summary="injected")
    inventory = [
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
    task = TaskContract(goal="injected planning", complexity=1, risk=RiskLevel.LOW)
    async with await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "injected.db"),
        repo_path=str(repo),
        policy_path=str(policy),
        model_discoverer=lambda: inventory,
    ) as orchestrator:
        result = await orchestrator.run(task)

    assert runtime.inventory_calls == 0
    assert result["outcome"] == "completed"


@pytest.mark.asyncio
async def test_payload_normalization_uses_hmc_discover_models_payload_keyword(
    git_repo: tuple[Path, Path], tmp_path: Path, filesystem_read_task: TaskContract, monkeypatch
) -> None:
    repo, policy = git_repo
    runtime = InventoryRuntime(picker_payload())
    runtime.enqueue_scenario("pass", summary="normalized")
    seen = []

    def capture(payload=None):  # noqa: ANN001
        seen.append(payload)
        return hmc_discover_models(payload=payload)

    monkeypatch.setattr("orchestrator.engine.discover_models", capture)
    async with await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "payload.db"),
        repo_path=str(repo),
        policy_path=str(policy),
    ) as orchestrator:
        result = await orchestrator.run(filesystem_read_task)

    assert result["outcome"] == "completed"
    assert seen == [picker_payload()]


@pytest.mark.asyncio
async def test_production_path_preserves_hmc_planning_before_observed_execution(
    git_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    from orchestrator.engine import Orchestrator

    repo, policy = git_repo
    runtime = ControlledRuntime()
    runtime.enqueue_scenario("pass", summary="AAO_E2E_1_OK")
    inventory = [
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
    task = TaskContract(
        task_type=TaskType.GENERAL,
        goal="Return exactly this token and nothing else: AAO_E2E_1_OK",
        complexity=1,
        risk=RiskLevel.LOW,
        allowed_paths=[],
        forbidden_actions=[
            "filesystem access",
            "shell",
            "tests",
            "web",
            "modification",
            "background work",
        ],
    )

    async with await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "run.db"),
        repo_path=str(repo),
        policy_path=str(policy),
        model_discoverer=lambda: inventory,
    ) as orchestrator:
        result = await orchestrator.run(task)

    assert runtime.submit_calls == 1
    assert result["outcome"] == "completed"
    assert result["planned"]["task_profile"]["execution_complexity"] == "low"
    assert result["planned"]["requirements"] == {
        "filesystem_read": False,
        "filesystem_write": False,
        "shell": False,
        "tests": False,
        "web": False,
        "background_execution": False,
        "persistent_tasks": False,
        "human_in_loop": False,
    }
    assert result["planned"]["hmc"]["request_type"] == "PlannerRequest"
    assert result["planned"]["hmc"]["recommendation_type"] == "PlannerRecommendation"
    assert result["planned"]["hmc"]["recommendation"]["planned_call_count"] >= 1
    assert result["planned"]["runtime_selection"]["selected_runtime"] == "hermes"
    assert result["planned"]["runtime_plan"]["executor"] == "hermes"
    assert result["planned"]["runtime_plan"]["execution_mode"] == "direct"
    assert result["observed"]["runtime_adapter_invoked"] is True
    assert result["observed"]["run_id"]
    assert result["observed"]["runtime_status"] == "completed"
    assert "AAO_E2E_1_OK" in result["observed"]["output"]
    assert "runtime_call_count" not in result["planned"]["hmc"]["recommendation"]
    assert "judgment" not in result


class UnobservedRuntime(MockHermesAdapter):
    """Runtime without an observed health connection for the safety regression."""

    def __init__(self) -> None:
        super().__init__()
        self.submit_calls = 0

    async def connect(self) -> None:
        """Expose the production runtime connection seam without observing it."""
        return None

    async def submit(self, task):  # noqa: ANN001
        self.submit_calls += 1
        return await super().submit(task)


@pytest.mark.asyncio
async def test_single_execution_without_runtime_health_never_submits(
    git_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    repo, policy = git_repo
    runtime = UnobservedRuntime()
    runtime.enqueue_scenario("pass", summary="must not execute")
    db = Database(tmp_path / "unobserved.db")
    await db.connect()
    orchestrator = Orchestrator(
        runtime=runtime,
        db=db,
        state_machine=StateMachine(db),
        profiler=TaskProfiler(),
        router=RuleRouter(policy),
        budget_config=BudgetConfig(),
        approval_gate=ApprovalGate(policy),
        eval_gate=DeterministicEvalGate(repo),
        workspace_manager=WorkspaceManager(repo, policy),
        telemetry=TelemetryRecorder(db),
        runtime_health=None,
    )
    task = TaskContract(
        task_type=TaskType.GENERAL,
        goal="Return exactly this token and nothing else: AAO_E2E_1_OK",
        complexity=1,
        risk=RiskLevel.LOW,
    )

    try:
        result = await orchestrator.run(task)
    finally:
        await orchestrator.close()

    assert runtime.submit_calls == 0
    assert result["outcome"] in {"failed", "abandoned"}
    assert "runtime health" in result["detail"].lower()


@pytest.mark.asyncio
async def test_explicit_subtasks_without_planned_delegation_never_execute(
    git_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    from orchestrator.engine import Orchestrator

    repo, policy = git_repo
    runtime = ControlledRuntime()
    runtime.enqueue_scenario("pass", summary="must not execute")
    inventory = [
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
    task = TaskContract(
        task_type=TaskType.GENERAL,
        goal="execute explicit subtasks",
        subtasks=[SubtaskSpec(id="one", goal="perform one bounded step")],
    )

    async with await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "delegation-blocked.db"),
        repo_path=str(repo),
        policy_path=str(policy),
        model_discoverer=lambda: inventory,
    ) as orchestrator:
        delegation_calls = 0

        async def forbidden_delegation(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal delegation_calls
            delegation_calls += 1
            raise AssertionError("delegation must fail closed before invocation")

        orchestrator._delegation_executor.execute = forbidden_delegation
        result = await orchestrator.run(task)

    assert delegation_calls == 0
    assert runtime.submit_calls == 0
    assert result["run_id"] is None
    assert result["outcome"] == "failed"
    assert "planned delegation execution is not available" in result["detail"]


@pytest.mark.asyncio
async def test_direct_runtime_plan_overrides_legacy_delegation_route(
    git_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    from orchestrator.engine import Orchestrator

    repo, policy = git_repo
    runtime = ControlledRuntime()
    runtime.enqueue_scenario("pass", summary="AAO_E2E_1_OK")
    inventory = [
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
    task = TaskContract(
        task_type=TaskType.GENERAL,
        goal="legacy delegation route " + ("x" * 30000),
        complexity=1,
        risk=RiskLevel.LOW,
    )

    async with await Orchestrator.build(
        runtime=runtime,
        db_path=str(tmp_path / "direct-overrides-route.db"),
        repo_path=str(repo),
        policy_path=str(policy),
        model_discoverer=lambda: inventory,
    ) as orchestrator:
        delegation_calls = 0

        async def forbidden_delegation(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal delegation_calls
            delegation_calls += 1
            raise AssertionError("direct RuntimePlan must govern dispatch")

        orchestrator._delegation_executor.execute = forbidden_delegation
        result = await orchestrator.run(task)

    assert delegation_calls == 0
    assert runtime.submit_calls == 1
    assert result["planned"]["runtime_plan"]["execution_mode"] == "direct"
    assert result["planned"]["runtime_plan"]["executor"] == "hermes"
    assert result["observed"]["runtime_adapter_invoked"] is True

"""
Main orchestration engine — ties all layers together.

Flow per task:
    receive → profile → route → budget/approval check
    → allocate workspace
    → single:     inject_constraints → submit → stream → wait → eval → outcome
    → delegation: split_for_delegation → DelegationExecutor.execute → aggregate → outcome
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import structlog
from model_council.inventory import ModelSpec, discover_models

from adapters.runtime import AgentRuntime, RuntimeCapabilities
from contracts.delegation import DelegationResult
from contracts.evaluation import EvalStatus
from contracts.execution_mode import ExecutionModePolicy
from contracts.result import AgentResult, RunStatus
from contracts.runtime_health import HealthStatus, RuntimeHealth
from contracts.runtime_selection import RuntimeSelectionPolicy
from contracts.task import TaskContract
from evals.gate import DeterministicEvalGate, trusted_changed_files
from orchestrator.budget import ApprovalGate, BudgetConfig, BudgetState
from orchestrator.candidate_filter import RuntimeCandidate
from orchestrator.delegation_executor import DelegationExecutor
from orchestrator.execution_policy import ExecutionPolicy
from orchestrator.hmc_planning import HMCPlanningEvidence, build_hmc_planning
from orchestrator.planning_pipeline import PlanningResult, plan_runtime
from orchestrator.profiler import TaskProfiler
from orchestrator.prompt_guard import inject_constraints, split_for_delegation
from orchestrator.router import RuleRouter
from orchestrator.runtime_registry import RuntimeRegistry
from orchestrator.state_machine import StateMachine, TaskRecord, TaskStatus
from orchestrator.workspace import (
    ExecutionWorkspace,
    RepositoryBaseline,
    StagedDelivery,
    WorkspaceManager,
)
from storage.database import Database
from telemetry.events import TelemetryRecorder

log = structlog.get_logger(__name__)


def _runtime_terminal_outcome(status: RunStatus) -> str | None:
    """Return the public execution outcome for terminal runtime evidence."""
    return {
        RunStatus.COMPLETED: "completed",
        RunStatus.FAILED: "failed",
        RunStatus.CANCELLED: "cancelled",
        RunStatus.TIMEOUT: "timeout",
    }.get(status)


def _observed_runtime_identity(result: AgentResult | None) -> str | None:
    """Return only runtime identity evidence produced by the invoked adapter."""
    if result is None:
        return None
    runtime = result.provenance.get("runtime")
    return runtime if isinstance(runtime, str) and runtime.strip() else None


def _planning_payload(
    evidence: HMCPlanningEvidence,
    planning_result: PlanningResult,
) -> dict:
    context = evidence.context
    return {
        "task_profile": evidence.intake.profile.model_dump(mode="json"),
        "requirements": evidence.intake.requirements.model_dump(mode="json"),
        "hmc": {
            "request_type": type(context.request).__name__,
            "recommendation_type": type(context.recommendation).__name__,
            "request_contract_version": context.request.contract_version,
            "request_task_profile": asdict(context.request.task_profile),
            "mapping_policy_version": context.mapping_policy_version,
            "planner_id": context.planner_id,
            "recommendation": asdict(context.recommendation),
        },
        "candidate_assessments": [
            assessment.model_dump(mode="json")
            for assessment in planning_result.assessments
        ],
        "runtime_selection": planning_result.selection.model_dump(mode="json"),
        "execution_mode": (
            planning_result.mode.model_dump(mode="json")
            if planning_result.mode is not None
            else None
        ),
        "runtime_plan": (
            planning_result.plan.model_dump(mode="json")
            if planning_result.plan is not None
            else None
        ),
    }


class Orchestrator:
    """
    Top-level controller. One instance per process.

    Usage::

        async with await Orchestrator.build(runtime=adapter) as orch:
            result = await orch.run(task)
    """

    def __init__(
        self,
        runtime: AgentRuntime | None,
        db: Database,
        state_machine: StateMachine,
        profiler: TaskProfiler,
        router: RuleRouter,
        budget_config: BudgetConfig,
        approval_gate: ApprovalGate,
        eval_gate: DeterministicEvalGate,
        workspace_manager: WorkspaceManager | None,
        telemetry: TelemetryRecorder,
        runtime_health: RuntimeHealth | None = None,
        runtime_health_by_runtime: dict[str, RuntimeHealth] | None = None,
        runtime_capabilities_by_runtime: dict[str, RuntimeCapabilities] | None = None,
        model_discoverer: Callable[[], list[ModelSpec]] | None = None,
        runtime_registry: RuntimeRegistry | None = None,
        runtime_selection_policy: RuntimeSelectionPolicy | None = None,
        planning_required: bool | None = None,
    ) -> None:
        self._runtime_registry = runtime_registry or (
            RuntimeRegistry(entries=[("hermes", runtime)])
            if runtime is not None
            else None
        )
        if self._runtime_registry is None:
            raise ValueError("runtime or runtime_registry is required")
        if runtime is not None and runtime_registry is not None:
            raise ValueError("runtime and runtime_registry are mutually exclusive")
        identities = self._runtime_registry.identities()
        if runtime_selection_policy is None:
            if len(identities) != 1:
                raise ValueError("multiple runtimes require explicit selection policy")
            runtime_selection_policy = RuntimeSelectionPolicy(
                policy_version="runtime-selection-v1",
                runtime_priority=identities,
                allow_degraded_fallback=False,
            )
        self._runtime_selection_policy = runtime_selection_policy
        self._runtime_health_by_runtime: dict[str, RuntimeHealth] = {
            identity: (runtime_health_by_runtime or {}).get(
                identity, RuntimeHealth(runtime=identity)
            )
            for identity in self._runtime_registry.identities()
        }
        for identity, health in self._runtime_health_by_runtime.items():
            if health.runtime != identity:
                raise ValueError("runtime health identity must match registry identity")
        if runtime_health is not None:
            self._runtime_health_by_runtime["hermes"] = runtime_health
        self._runtime_capabilities_by_runtime: dict[str, RuntimeCapabilities] = dict(
            runtime_capabilities_by_runtime or {}
        )
        self._db = db
        self._sm = state_machine
        self._profiler = profiler
        self._router = router
        self._budget_config = budget_config
        self._approval = approval_gate
        self._execution_policy = ExecutionPolicy(
            always_require_actions=set(getattr(approval_gate, "_always_require", set())),
            require_approval_above_calls=budget_config.require_approval_above_calls,
            max_total_calls=budget_config.max_total_calls,
        )
        self._eval_gate = eval_gate
        self._wm = workspace_manager
        self._tel = telemetry
        self._planning_required = (
            planning_required
            if planning_required is not None
            else runtime is None or any(
                callable(getattr(registered_runtime, "connect", None))
                for _, registered_runtime in self._runtime_registry.items()
            )
        )
        self._model_discoverer = model_discoverer
        self._explicit_model_discoverer = model_discoverer is not None
        self._delegation_executor = DelegationExecutor(
            runtime=runtime or self._runtime_registry.resolve(self._runtime_selection_policy.runtime_priority[0]),
            eval_gate=eval_gate,
            telemetry=telemetry,
            execution_policy=self._execution_policy,
        )

    @classmethod
    async def build(
        cls,
        runtime: AgentRuntime | None = None,
        db_path: str = "data/orchestrator.db",
        repo_path: str = ".",
        policy_path: str = "policies/default.yaml",
        model_discoverer: Callable[[], list[ModelSpec]] | None = None,
        runtime_registry: RuntimeRegistry | None = None,
        runtime_selection_policy: RuntimeSelectionPolicy | None = None,
        runtime_health_by_runtime: dict[str, RuntimeHealth] | None = None,
        planning_required: bool | None = None,
    ) -> Orchestrator:
        import yaml

        db = Database(Path(db_path))
        await db.connect()

        raw = yaml.safe_load(Path(policy_path).read_text())
        budget_cfg_raw = raw.get("budget", )
        budget_config = BudgetConfig(
            max_children=budget_cfg_raw.get("max_children", 2),
            max_depth=budget_cfg_raw.get("max_depth", 1),
            max_retries=budget_cfg_raw.get("max_retries", 1),
            max_total_calls=budget_cfg_raw.get("max_total_calls", 8),
            require_approval_above_calls=budget_cfg_raw.get("require_approval_above_calls", 5),
        )

        workspace_manager: WorkspaceManager | None = None
        if Path(repo_path).exists():
            try:
                import subprocess

                probe = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if probe.returncode == 0 and probe.stdout.strip() == "true":
                    workspace_manager = WorkspaceManager(
                        repo_path=repo_path, policy_path=policy_path
                    )
            except Exception:
                log.warning("workspace_manager_init_failed", repo_path=repo_path)

        if runtime is not None and runtime_registry is not None:
            await db.close()
            raise ValueError("runtime and runtime_registry are mutually exclusive")
        normalized_registry = runtime_registry or (
            RuntimeRegistry(entries=[("hermes", runtime)])
            if runtime is not None
            else None
        )
        if normalized_registry is None:
            await db.close()
            raise ValueError("runtime or runtime_registry is required")
        if runtime_selection_policy is None:
            identities = normalized_registry.identities()
            if len(identities) != 1:
                await db.close()
                raise ValueError("multiple runtimes require explicit selection policy")
            runtime_selection_policy = RuntimeSelectionPolicy(
                policy_version="runtime-selection-v1",
                runtime_priority=identities,
                allow_degraded_fallback=False,
            )

        supplied_health = dict(runtime_health_by_runtime or {})
        for identity, health in supplied_health.items():
            if identity not in normalized_registry.identities():
                await db.close()
                raise ValueError("runtime health identity is not registered")
            if health.runtime != identity:
                await db.close()
                raise ValueError("runtime health identity must match registry identity")
        observed_health_by_runtime: dict[str, RuntimeHealth] = {}
        runtime_capabilities_by_runtime: dict[str, RuntimeCapabilities] = {}
        for identity, registered_runtime in normalized_registry.items():
            checked_at = datetime.now(UTC)
            connect = getattr(registered_runtime, "connect", None)
            if callable(connect):
                try:
                    await connect()
                except Exception as exc:
                    observed_health_by_runtime[identity] = RuntimeHealth(
                        runtime=identity,
                        status=HealthStatus.UNAVAILABLE,
                        reasons=[f"runtime connection failed: {exc}"],
                        checked_at=checked_at,
                    )
                else:
                    observed_health_by_runtime[identity] = RuntimeHealth(
                        runtime=identity,
                        status=HealthStatus.AVAILABLE,
                        reasons=["runtime connect succeeded"],
                        checked_at=checked_at,
                    )
            else:
                observed_health_by_runtime[identity] = supplied_health.get(
                    identity, RuntimeHealth(runtime=identity)
                )
            try:
                runtime_capabilities_by_runtime[identity] = await registered_runtime.capabilities()
            except Exception as exc:
                runtime_capabilities_by_runtime[identity] = RuntimeCapabilities()
                observed_health_by_runtime[identity] = RuntimeHealth(
                    runtime=identity,
                    status=HealthStatus.UNAVAILABLE,
                    reasons=[f"runtime capabilities failed: {exc}"],
                    checked_at=checked_at,
                )

        return cls(
            runtime=None,
            db=db,
            state_machine=StateMachine(db),
            profiler=TaskProfiler(),
            router=RuleRouter(policy_path),
            budget_config=budget_config,
            approval_gate=ApprovalGate(policy_path),
            eval_gate=DeterministicEvalGate(repo_path),
            workspace_manager=workspace_manager,
            telemetry=TelemetryRecorder(db),
            runtime_health_by_runtime=observed_health_by_runtime,
            runtime_capabilities_by_runtime=runtime_capabilities_by_runtime,
            model_discoverer=model_discoverer,
            runtime_registry=normalized_registry,
            runtime_selection_policy=runtime_selection_policy,
            planning_required=(
                planning_required
                if planning_required is not None
                else runtime is None
                or runtime_registry is not None
                or model_discoverer is not None
                or any(
                    callable(getattr(registered_runtime, "model_inventory_payload", None))
                    for _, registered_runtime in normalized_registry.items()
                )
            ),
        )

    async def close(self) -> None:
        try:
            for _identity, runtime in self._runtime_registry.items():
                disconnect = getattr(runtime, "disconnect", None)
                if callable(disconnect):
                    with contextlib.suppress(Exception):
                        await disconnect()
        finally:
            await self._db.close()

    def _planning_health_available(self) -> bool:
        return any(
            health.status == HealthStatus.AVAILABLE
            for health in self._runtime_health_by_runtime.values()
        )

    def _planning_health_observed(self) -> bool:
        return all(
            health.status != HealthStatus.UNKNOWN
            for health in self._runtime_health_by_runtime.values()
        )

    async def _resolve_planning_models(self) -> list[ModelSpec]:
        if not self._explicit_model_discoverer:
            inventory = next(
                (
                    registered_runtime.model_inventory_payload
                    for _, registered_runtime in self._runtime_registry.items()
                    if callable(getattr(registered_runtime, "model_inventory_payload", None))
                ),
                None,
            )
            if inventory is not None:
                payload = await inventory()
                models = discover_models(payload=payload)
            else:
                models = discover_models()
        else:
            assert self._model_discoverer is not None
            models = self._model_discoverer()
        if not models:
            raise RuntimeError("HMC model inventory contains no usable configured models")
        return models

    async def _prepare_hmc_planning(
        self,
        task: TaskContract,
    ) -> tuple[HMCPlanningEvidence, PlanningResult] | None:
        """Prepare HMC evidence and an AAO plan before runtime submission."""
        if not self._planning_health_observed():
            raise RuntimeError(
                "runtime health is unavailable; executable planning is required before runtime submission"
            )

        models = await self._resolve_planning_models()
        evidence = build_hmc_planning(
            task,
            model_discoverer=lambda: models,
        )
        candidates = []
        for identity, registered_runtime in self._runtime_registry.items():
            capabilities = self._runtime_capabilities_by_runtime.get(identity)
            if capabilities is None:
                try:
                    capabilities = await registered_runtime.capabilities()
                except Exception as exc:
                    self._runtime_health_by_runtime[identity] = RuntimeHealth(
                        runtime=identity,
                        status=HealthStatus.UNAVAILABLE,
                        reasons=[f"runtime capabilities failed: {exc}"],
                        checked_at=datetime.now(UTC),
                    )
                    capabilities = RuntimeCapabilities()
                self._runtime_capabilities_by_runtime[identity] = capabilities
            candidates.append(
                RuntimeCandidate(
                    runtime=identity,
                    capabilities=capabilities,
                    health=self._runtime_health_by_runtime[identity],
                )
            )
        planning_result = plan_runtime(
            evidence.intake.profile,
            evidence.intake.requirements,
            candidates,
            self._runtime_selection_policy,
            ExecutionModePolicy(policy_version="execution-mode-v1"),
            plan_policy_version="runtime-plan-v1",
            hmc_context=evidence.context,
        )
        return evidence, planning_result

    def _capture_root_baseline(self) -> RepositoryBaseline:
        if self._wm is None:
            raise RuntimeError("repository baseline requires a workspace manager")
        root = self._wm.repo_path.resolve()
        candidates = [self._wm._base.resolve(), self._db._path.resolve()]
        candidates.extend(
            Path(str(self._db._path.resolve()) + suffix)
            for suffix in ("-wal", "-shm", "-journal")
        )
        exclusions: list[str] = []
        for candidate in candidates:
            try:
                exclusions.append(candidate.relative_to(root).as_posix())
            except ValueError:
                continue
        return RepositoryBaseline.capture(
            root,
            excluded_paths=tuple(exclusions),
            protected_paths=(self._wm._base.resolve(),),
        )

    async def _cancel_and_confirm_terminal(self, runtime: AgentRuntime, handle) -> None:
        """Cancel, prove terminal truth, then release run workspace authority."""
        cancel_error: Exception | None = None
        try:
            await runtime.cancel(handle)
        except Exception as exc:
            cancel_error = exc
        try:
            result = await runtime.wait(handle)
        except Exception as exc:
            detail = f"terminal confirmation failed: {exc}"
            if cancel_error is not None:
                detail = f"cancel failed: {cancel_error}; {detail}"
            raise RuntimeError(f"runtime quiescence could not be established: {detail}") from exc
        if result.status not in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMEOUT,
        }:
            raise RuntimeError(
                "runtime quiescence could not be established: "
                f"wait returned non-terminal status {result.status}"
            )
        try:
            await runtime.quiesce(handle)
        except Exception as exc:
            raise RuntimeError(
                "runtime quiescence could not be established after "
                f"terminal status {result.status.value}: {exc}"
            ) from exc

    async def __aenter__(self) -> Orchestrator:
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self, task: TaskContract) -> dict:
        """
        Execute a task end-to-end. Returns a summary dict.
        Raises only on unrecoverable internal errors; task-level failures
        are recorded as FAILED/ABANDONED.
        """
        record = await self._sm.create(task)

        try:
            return await self._execute(record)
        except Exception as exc:
            log.exception("orchestrator_internal_error", task_id=task.id)
            await record.mark_failed(str(exc))
            raise

    async def _execute(
        self,
        record: TaskRecord,
        budget: BudgetState | None = None,
        is_retry: bool = False,
    ) -> dict:
        task = record.task
        if budget is None:
            budget = BudgetState(task_id=task.id, config=self._budget_config)

        # 1. Profile — skip on retry (record already went through PROFILED once)
        if not is_retry:
            profile = self._profiler.profile(task)
            await record.transition(TaskStatus.PROFILED)
            await self._tel.record(task.id, "task_profiled", profile.as_dict())
        else:
            profile = self._profiler.profile(task)  # re-profile but don't transition

        # 2. Route
        decision = self._router.route(
            task,
            independent_subtask_count=profile.independent_subtask_count,
            has_sequential_dependency=profile.has_sequential_dependency,
            affected_module_count=profile.affected_module_count,
        )
        if not is_retry:
            await record.mark_routed(decision.route)
            await self._db.append_routing_decision(
                task.id, decision.policy_version, decision.route, decision.reasons
            )
            await self._tel.record(task.id, "task_routed", decision.to_dict())
            log.info("task_routed", task_id=task.id, **decision.to_dict())

        # 3. Budget + approval check
        violation = budget.check_calls()
        if violation:
            await record.mark_abandoned(f"budget: {violation.detail}")
            return self._summary(record, "abandoned", violation.detail)

        needs_approval, reason = self._approval.requires_approval(task)
        if needs_approval:
            approved = self._approval.prompt_user(reason, task)
            if not approved:
                await record.mark_abandoned("user declined approval")
                return self._summary(record, "abandoned", "declined by user")

        planning_evidence: HMCPlanningEvidence | None = None
        planning_result: PlanningResult | None = None
        if self._planning_required and not task.subtasks:
            try:
                prepared = await self._prepare_hmc_planning(task)
                if prepared is not None:
                    planning_evidence, planning_result = prepared
            except Exception as exc:
                await record.mark_failed("planning_failed")
                return self._summary(
                    record,
                    "failed",
                    f"HMC/AAO planning failed: {exc}",
                )
            if planning_result is None or planning_result.plan is None:
                await record.mark_failed("planning_failed")
                return self._summary(
                    record,
                    "failed",
                    "HMC/AAO planning produced no executable RuntimePlan",
                    planning_evidence=planning_evidence,
                    planning_result=planning_result,
                )

        if self._planning_required and task.subtasks:
            await record.mark_failed("planning_failed")
            return self._summary(
                record,
                "failed",
                "planned delegation execution is not available",
            )

        # 4. Pre-flight: subtask count vs budget
        if (
            decision.route == "delegation"
            and profile.independent_subtask_count > budget.config.max_children
        ):
            if task.subtasks:
                detail = (
                    f"explicit subtasks={len(task.subtasks)} exceed "
                    f"max_children={budget.config.max_children}"
                )
                await record.mark_abandoned(detail)
                return self._summary(record, "abandoned", detail)
            log.warning(
                "subtask_count_exceeds_budget",
                task_id=task.id,
                subtask_count=profile.independent_subtask_count,
                max_children=budget.config.max_children,
            )
            await self._tel.record(
                task.id,
                "subtask_clamped",
                {
                    "requested": profile.independent_subtask_count,
                    "clamped_to": budget.config.max_children,
                },
            )

        # ── Branch: DELEGATION ─────────────────────────────────────────────────
        if decision.route == "delegation":
            if planning_result is not None:
                if (
                    planning_result.plan is not None
                    and planning_result.plan.execution_mode.value == "direct"
                ):
                    return await self._execute_single(
                        record,
                        task,
                        budget,
                        decision,
                        is_retry,
                        planning_evidence=planning_evidence,
                        planning_result=planning_result,
                    )
                await record.mark_failed("planning_failed")
                return self._summary(
                    record,
                    "failed",
                    "planned execution mode is not supported by E2E-1B",
                    planning_evidence=planning_evidence,
                    planning_result=planning_result,
                )
            return await self._execute_delegation(record, task, budget, decision)

        # ── Branch: SINGLE ─────────────────────────────────────────────────────
        return await self._execute_single(
            record,
            task,
            budget,
            decision,
            is_retry,
            planning_evidence=planning_evidence,
            planning_result=planning_result,
        )

    # ── Single-agent execution ─────────────────────────────────────────────────

    async def _execute_single(
        self,
        record: TaskRecord,
        task: TaskContract,
        budget: BudgetState,
        decision,
        is_retry: bool,
        *,
        planning_evidence: HMCPlanningEvidence | None = None,
        planning_result: PlanningResult | None = None,
    ) -> dict:
        runtime_adapter_invoked = False
        observed_events: list[str] = []
        agent_result = None
        handle = None
        planned_runtime = planning_result.plan.executor if planning_result and planning_result.plan else None
        if (
            planning_result is not None
            and planning_result.selection.selected_runtime != planned_runtime
        ):
            raise RuntimeError("RuntimePlan executor must match runtime selection")
        runtime_identity = planned_runtime or self._runtime_selection_policy.runtime_priority[0]
        runtime = self._runtime_registry.resolve(runtime_identity)

        def summary(outcome: str, detail: str, **extra) -> dict:
            return self._summary(
                record,
                outcome,
                detail,
                planning_evidence=planning_evidence,
                planning_result=planning_result,
                runtime_adapter_invoked=runtime_adapter_invoked,
                observed_events=observed_events,
                agent_result=agent_result,
                run_handle=handle,
                **extra,
            )

        # Every repository execution, including read-only work, is isolated.
        if self._wm is None:
            await record.mark_failed("workspace unavailable for repository task")
            return summary("failed", "workspace unavailable for repository task")
        write_task = self._wm.needs_worktree(task.task_type.value)
        workspace = ExecutionWorkspace.create(
            self._wm.repo_path,
            task_id=task.id,
            execution_id=f"single-{record.retry_count}",
        )
        task = task.model_copy(
            update={"context": {**task.context, "_eval_base_sha": workspace.base_sha}}
        )
        guarded_task = inject_constraints(task, workspace.path, workspace.execution_id)

        # 6. Submit
        submission_decision = self._execution_policy.authorize_submission(
            task, calls_used=budget.calls_used
        )
        if not submission_decision.allowed:
            approved = self._approval.prompt_user(
                "runtime call threshold requires approval", task
            )
            submission_decision = self._execution_policy.authorize_submission(
                task, calls_used=budget.calls_used, approval=approved
            )
        if not submission_decision.allowed:
            workspace.rollback()
            await record.mark_abandoned(submission_decision.reason)
            return summary("abandoned", submission_decision.reason)

        violation = budget.reserve_calls(1)
        if violation:
            workspace.rollback()
            await record.mark_abandoned(f"budget: {violation.detail}")
            return summary("abandoned", violation.detail)
        root_baseline: RepositoryBaseline | None = None
        workspace_baseline: RepositoryBaseline | None = None
        try:
            if not write_task:
                root_baseline = self._capture_root_baseline()
                workspace_baseline = RepositoryBaseline.capture(workspace.path)
            if self._planning_required:
                if runtime is None or planned_runtime is None:
                    raise RuntimeError(
                        "runtime submission requires an executable RuntimePlan"
                    )
                if self._runtime_health_by_runtime[planned_runtime].status != HealthStatus.AVAILABLE:
                    raise RuntimeError(
                        "runtime submission requires available observed runtime health"
                    )
                if planning_result is None or planning_result.plan is None or planning_result.plan.execution_mode is None:
                    raise RuntimeError(
                        "runtime submission requires an executable RuntimePlan"
                    )
            assert runtime is not None
            handle = await runtime.submit(guarded_task)
            runtime_adapter_invoked = True
        except Exception:
            budget.release_reserved_call()
            if workspace:
                workspace.rollback()
            raise
        try:
            budget.commit_reserved_call()
            await record.mark_running(handle.run_id)
            await self._tel.record(
                task.id, "task_submitted", {"run_id": handle.run_id}, handle.run_id
            )
        except Exception:
            await self._cancel_and_confirm_terminal(runtime, handle)
            workspace.rollback()
            raise

        terminal_confirmed = False
        try:
            # 7. Stream events — watch for approval / usage / errors
            async for event in runtime.events(handle):
                observed_events.append(event.type)
                await self._tel.record(
                    task.id, f"event_{event.type}", event.payload, handle.run_id
                )

                if event.type == "approval_request":
                    await self._cancel_and_confirm_terminal(runtime, handle)
                    terminal_confirmed = True
                    workspace.rollback()
                    await record.mark_failed("runtime approval request cannot be resumed")
                    return summary("failed", "runtime approval request failed closed")

                elif event.type in ("completed", "error"):
                    break

            agent_result = await runtime.wait(handle)
            terminal_confirmed = agent_result.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.TIMEOUT,
            }
            if not terminal_confirmed:
                raise RuntimeError(
                    f"runtime wait returned non-terminal status {agent_result.status}"
                )
            terminal_outcome = _runtime_terminal_outcome(agent_result.status)
            try:
                await runtime.quiesce(handle)
            except Exception as exc:
                detail = (
                    "runtime quiescence failed after observed terminal status "
                    f"{agent_result.status.value}: {exc}; workspace retained"
                )
                await record.mark_failed(detail)
                return summary("failed", detail)
            if terminal_outcome != "completed":
                workspace.rollback()
                detail = agent_result.error or f"runtime ended {terminal_outcome}"
                await record.mark_failed(detail)
                return summary(terminal_outcome or "failed", detail)
            usage = await runtime.usage(handle)
        except Exception as exc:
            if not terminal_confirmed:
                try:
                    await self._cancel_and_confirm_terminal(runtime, handle)
                    terminal_confirmed = True
                except Exception as quiescence_error:
                    raise RuntimeError(str(quiescence_error)) from exc
            if not workspace.cleaned:
                workspace.rollback()
            raise
        try:
            if usage.input_tokens is not None and usage.output_tokens is not None:
                await self._db.append_usage(
                    task.id,
                    handle.run_id,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.estimated_cost_usd,
                )

            # 9. Eval gate
            await record.mark_evaluating()
            eval_result = await self._eval_gate.run(guarded_task, agent_result, budget)
            await self._db.append_eval_result(
                task.id,
                handle.run_id,
                eval_result.overall.value,
                [c.model_dump() for c in eval_result.checks],
            )
            await self._tel.record(
                task.id,
                "eval_completed",
                {
                    "overall": eval_result.overall.value,
                    "failed_checks": [c.name for c in eval_result.failed_checks()],
                },
                handle.run_id,
            )
        except Exception:
            if not workspace.cleaned:
                workspace.rollback()
            raise

        # 10. Outcome
        if eval_result.overall == EvalStatus.PASS:
            if not write_task:
                root_changes = root_baseline.changed() if root_baseline else ["<root>"]
                workspace_changes = (
                    workspace_baseline.changed()
                    if workspace_baseline
                    else ["<workspace>"]
                )
                if root_changes or workspace_changes:
                    cleanup_error = ""
                    try:
                        workspace.cleanup()
                    except Exception as exc:
                        cleanup_error = f"; cleanup failed: {exc}"
                    detail = (
                        "read-only final baseline mismatch: "
                        f"root={root_changes}, workspace={workspace_changes}"
                        f"{cleanup_error}"
                    )
                    await record.mark_failed("read_only_mutation")
                    return summary(
                        "failed",
                        detail,
                        eval_result=eval_result,
                        usage=usage,
                        files_changed=workspace_changes,
                        workspace_root=str(workspace.path),
                        isolation_level="workspace",
                        verification_status="fail",
                    )
            try:
                trusted_files = workspace.changed_files()
                if write_task:
                    workspace.integrate()
                else:
                    workspace.cleanup()
                await record.mark_completed()
            except Exception as exc:
                if workspace.cleaned:
                    self._wm.rollback(workspace.base_sha)
                else:
                    workspace.rollback()
                await record.mark_failed("integration_failed")
                return summary("failed", str(exc), eval_result=eval_result)
            return summary(
                "completed",
                "",
                eval_result=eval_result,
                usage=usage,
                files_changed=trusted_files,
                workspace_root=str(workspace.path),
                isolation_level="workspace",
            )

        # Eval failed — retry once if budget allows
        workspace.rollback()
        retry_violation = budget.check_retries()
        if not retry_violation:
            await record.mark_failed("eval_failed")
            await record.mark_retry()
            log.info("task_retrying", task_id=task.id, retry=record.retry_count)
            budget.retries_used += 1
            await record.mark_routed(decision.route)
            return await self._execute(record, budget=budget, is_retry=True)

        await record.mark_failed("eval_failed_no_retries")
        return summary(
            "failed",
            "eval failed and retries exhausted",
            eval_result=eval_result,
        )

    # ── Delegation execution ───────────────────────────────────────────────────

    async def _execute_delegation(
        self,
        record: TaskRecord,
        task: TaskContract,
        budget: BudgetState,
        decision,
    ) -> dict:
        # 5. Workspace allocation is all-or-nothing for every delegated child.
        if self._wm is None:
            await record.mark_failed("workspace unavailable for delegated task")
            return self._summary(
                record, "failed", "workspace unavailable for delegated task"
            )
        child_count = len(task.subtasks) if task.subtasks else budget.config.max_children
        batch_approval: bool | None = None
        batch_decisions = [
            self._execution_policy.authorize_submission(
                task, calls_used=budget.calls_used + offset
            )
            for offset in range(child_count)
        ]
        if any(not item.allowed for item in batch_decisions):
            approved = self._approval.prompt_user(
                "delegated runtime call threshold requires approval", task
            )
            batch_approval = approved
            batch_decisions = [
                self._execution_policy.authorize_submission(
                    task,
                    calls_used=budget.calls_used + offset,
                    approval=approved,
                )
                for offset in range(child_count)
            ]
        if any(not item.allowed for item in batch_decisions):
            reason = next(item.reason for item in batch_decisions if not item.allowed)
            await record.mark_abandoned(reason)
            return self._summary(record, "abandoned", reason)
        worktrees: list[tuple[str, Path | None]] = []
        integration_base: str | None = None
        read_only = not self._wm.needs_worktree(task.task_type.value)
        root_baseline: RepositoryBaseline | None = None
        worktree_baselines: dict[str, RepositoryBaseline] = {}
        try:
            self._wm.ensure_root_clean()
            integration_base = self._wm.current_head()
            task = task.model_copy(
                update={
                    "context": {**task.context, "_eval_base_sha": integration_base}
                }
            )
            for i in range(child_count):
                child_id = f"child-{i+1}"
                wt = self._wm.allocate(task.id, child_id)
                self._wm.activate(task.id, child_id)
                worktrees.append((child_id, wt.worktree_path))
                log.info(
                    "worktree_allocated",
                    task_id=task.id,
                    child_id=child_id,
                    worktree_path=str(wt.worktree_path),
                )
        except Exception as exc:
            cleanup_errors = []
            for child_id, _ in worktrees:
                try:
                    self._wm.clean(task.id, child_id)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"{child_id}: {cleanup_exc}")
            detail = f"workspace allocation failed: {exc}"
            if cleanup_errors:
                detail += f"; cleanup failed: {'; '.join(cleanup_errors)}"
            await record.mark_failed("workspace_allocation_failed")
            return self._summary(record, "failed", detail)

        quarantined_ids: set[str] = set()
        delegation_result: DelegationResult | None = None
        try:
            # 6. Prompt Guard — preserve explicit subtask goals in child contracts.
            child_contracts = split_for_delegation(task, worktrees)

            if read_only:
                root_baseline = self._capture_root_baseline()
                worktree_baselines = {
                    child_id: RepositoryBaseline.capture(path)
                    for child_id, path in worktrees
                    if path is not None
                }

            # 7. A delegated parent is running, but has no native runtime run.
            await record.transition(TaskStatus.RUNNING)
            # 8. Execute children concurrently via DelegationExecutor
            delegation_result: DelegationResult = await self._delegation_executor.execute(
                parent_task_id=task.id,
                children=child_contracts,
                budget=budget,
                submission_approval=batch_approval,
            )
            quarantined_ids = {
                child.child_id
                for child in delegation_result.children
                if not child.workspace_safe_to_cleanup
            }

            # Persist only child usage tied to an observed native runtime run.
            for child in delegation_result.children:
                if (
                    child.run_id is None
                    or child.input_tokens is None
                    or child.output_tokens is None
                ):
                    continue
                await self._db.append_usage(
                    task.id,
                    child.run_id,
                    child.input_tokens,
                    child.output_tokens,
                    child.estimated_cost_usd,
                )
        except Exception as exc:
            cleanup_errors = []
            for child_id, _ in worktrees:
                if child_id in quarantined_ids:
                    continue
                try:
                    self._wm.clean(task.id, child_id)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"{child_id}: {cleanup_exc}")
            detail = f"delegation execution failed: {exc}"
            if cleanup_errors:
                detail += f"; cleanup failed: {'; '.join(cleanup_errors)}"
            if quarantined_ids:
                detail += f"; quarantined workspaces: {sorted(quarantined_ids)}"
            await record.mark_failed("delegation_execution_failed")
            if delegation_result is not None:
                return self._summary_delegation(
                    record, delegation_result, detail=detail, outcome="failed"
                )
            return self._summary(record, "failed", detail)

        status = delegation_result.overall_status
        try:
            # 9. Evaluate overall outcome
            await record.mark_evaluating()
            await self._tel.record(
                task.id,
                "delegation_outcome",
                {
                    "status": status,
                    "successful": delegation_result.successful,
                    "failed": delegation_result.failed,
                    "children": len(delegation_result.children),
                },
            )
        except Exception as exc:
            cleanup_errors = []
            for child_id, _ in worktrees:
                if child_id in quarantined_ids:
                    continue
                try:
                    self._wm.clean(task.id, child_id)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"{child_id}: {cleanup_exc}")
            detail = f"delegation state transition failed: {exc}"
            if cleanup_errors:
                detail += f"; cleanup failed: {'; '.join(cleanup_errors)}"
            if quarantined_ids:
                detail += f"; quarantined workspaces: {sorted(quarantined_ids)}"
            await record.mark_failed("delegation_state_failed")
            return self._summary_delegation(
                record, delegation_result, detail=detail, outcome="failed"
            )

        if status == "completed":
            records = [
                wt_record for wt_record in self._wm.list_records()
                if wt_record.task_id == task.id
            ]
            if read_only:
                aggregate = delegation_result.aggregate_result
                try:
                    if aggregate is None:
                        raise RuntimeError("delegation produced no aggregate result")
                    integrated_eval = await self._eval_gate.run(task, aggregate, budget)
                    if integrated_eval.overall != EvalStatus.PASS:
                        failed = ", ".join(
                            check.name for check in integrated_eval.failed_checks()
                        )
                        raise RuntimeError(f"integrated eval failed: {failed}")

                    root_changes = (
                        root_baseline.changed() if root_baseline else ["<root>"]
                    )
                    child_changes = {
                        child_id: baseline.changed()
                        for child_id, baseline in worktree_baselines.items()
                    }
                    child_changes = {
                        child_id: changes
                        for child_id, changes in child_changes.items()
                        if changes
                    }
                    if root_changes or child_changes:
                        raise RuntimeError(
                            "read-only final baseline mismatch: "
                            f"root={root_changes}, children={child_changes}"
                        )

                    for wt_record in records:
                        self._wm.clean(task.id, wt_record.child_id)
                    await record.mark_completed()
                except Exception as exc:
                    cleanup_errors = []
                    for wt_record in records:
                        try:
                            self._wm.clean(task.id, wt_record.child_id)
                        except Exception as cleanup_exc:
                            cleanup_errors.append(
                                f"{wt_record.child_id}: {cleanup_exc}"
                            )
                    detail = str(exc)
                    if cleanup_errors:
                        detail += f"; cleanup failed: {'; '.join(cleanup_errors)}"
                    await record.mark_failed("read_only_mutation")
                    return self._summary_delegation(
                        record,
                        delegation_result,
                        detail=detail,
                        outcome="failed",
                        verification_status="fail",
                    )
                return self._summary_delegation(
                    record,
                    delegation_result,
                    files_changed=[],
                    verification_status="pass",
                )
            try:
                trusted_files: list[str] = []
                staged_delivery: StagedDelivery | None = None
                for wt_record in records:
                    child_files = trusted_changed_files(
                        wt_record.worktree_path, base_sha=integration_base
                    )
                    if child_files is None:
                        raise RuntimeError(
                            f"could not derive trusted files for {wt_record.child_id}"
                        )
                    trusted_files.extend(child_files)
                trusted_files = list(dict.fromkeys(trusted_files))

                for wt_record in records:
                    self._wm.commit_changes(task.id, wt_record.child_id)

                staged_delivery = self._wm.stage_batch(
                    task.id, [wt_record.child_id for wt_record in records]
                )

                aggregate = delegation_result.aggregate_result
                if aggregate is None:
                    raise RuntimeError("delegation produced no aggregate result")
                integrated_task = inject_constraints(task, staged_delivery.path)
                integrated_task.context["_eval_base_sha"] = integration_base
                integrated_eval = await self._eval_gate.run(
                    integrated_task, aggregate, budget
                )
                if integrated_eval.overall != EvalStatus.PASS:
                    failed = ", ".join(
                        check.name for check in integrated_eval.failed_checks()
                    )
                    raise RuntimeError(f"integrated eval failed: {failed}")
                self._wm.deliver_batch(staged_delivery)
                staged_delivery = None
                for wt_record in records:
                    self._wm.clean(task.id, wt_record.child_id)
                await record.mark_completed()
            except Exception as exc:
                failures = []
                if staged_delivery is not None:
                    try:
                        self._wm.discard_batch(staged_delivery)
                    except Exception as discard_exc:
                        failures.append(f"staging cleanup failed: {discard_exc}")
                try:
                    self._wm.rollback(integration_base)
                except Exception as rollback_exc:
                    failures.append(f"rollback failed: {rollback_exc}")
                for wt_record in records:
                    try:
                        self._wm.clean(task.id, wt_record.child_id)
                    except Exception as cleanup_exc:
                        failures.append(
                            f"cleanup failed for {wt_record.child_id}: {cleanup_exc}"
                        )
                detail = str(exc)
                if failures:
                    detail += "; " + "; ".join(failures)
                await record.mark_failed("integration_failed")
                return self._summary_delegation(
                    record,
                    delegation_result,
                    detail=detail,
                    outcome="failed",
                    verification_status="fail",
                )
            return self._summary_delegation(
                record,
                delegation_result,
                files_changed=trusted_files,
                verification_status="pass",
            )

        if status == "partial_failed":
            cleanup_errors = []
            for child_id, _ in worktrees:
                if child_id in quarantined_ids:
                    continue
                try:
                    self._wm.clean(task.id, child_id)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"{child_id}: {cleanup_exc}")
            await record.mark_failed("partial_children_failed")
            detail = (
                f"{delegation_result.failed} of {len(delegation_result.children)} "
                "children failed; nothing integrated"
            )
            if cleanup_errors:
                detail += f"; cleanup failed: {'; '.join(cleanup_errors)}"
            if quarantined_ids:
                detail += f"; quarantined workspaces: {sorted(quarantined_ids)}"
            return self._summary_delegation(
                record,
                delegation_result,
                detail=detail,
                outcome="failed",
            )

        # All failed
        cleanup_errors = []
        for child_id, _ in worktrees:
            if child_id in quarantined_ids:
                continue
            try:
                self._wm.clean(task.id, child_id)
            except Exception as cleanup_exc:
                cleanup_errors.append(f"{child_id}: {cleanup_exc}")
        detail = "all delegation children failed"
        if cleanup_errors:
            detail += f"; cleanup failed: {'; '.join(cleanup_errors)}"
        if quarantined_ids:
            detail += f"; quarantined workspaces: {sorted(quarantined_ids)}"
        await record.mark_failed("all_children_failed")
        return self._summary_delegation(
            record, delegation_result, detail=detail, outcome="failed"
        )

    # ── Summary helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _summary(
        record: TaskRecord, outcome: str, detail: str, **extra
    ) -> dict:
        d: dict = {
            "task_id": record.task.id,
            "run_id": record.run_id,
            "outcome": outcome,
            "route": record.route,
            "retry_count": record.retry_count,
            "detail": detail,
        }
        if "eval_result" in extra:
            er = extra["eval_result"]
            d["eval"] = {
                "overall": er.overall.value,
                "failed_checks": [c.name for c in er.failed_checks()],
            }
            d["verification_status"] = er.overall.value
        if "usage" in extra:
            u = extra["usage"]
            d["usage"] = {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cached_tokens": u.cached_tokens,
                "total_tokens": u.total_tokens,
                "estimated_cost_usd": u.estimated_cost_usd,
            }
        for name in (
            "files_changed",
            "workspace_root",
            "isolation_level",
            "verification_status",
        ):
            if name in extra:
                d[name] = extra[name]
        planning_evidence = extra.get("planning_evidence")
        planning_result = extra.get("planning_result")
        planning_present = planning_evidence is not None and planning_result is not None
        if planning_present:
            d["planned"] = _planning_payload(planning_evidence, planning_result)
        observed_result = extra.get("agent_result")
        run_handle = extra.get("run_handle")
        runtime_invoked = bool(extra.get("runtime_adapter_invoked", False))
        if planning_present or runtime_invoked or observed_result is not None:
            if observed_result is not None:
                d["tool_calls"] = observed_result.tool_calls
            observed = {
                "runtime_adapter": (
                    _observed_runtime_identity(observed_result)
                    if runtime_invoked
                    else None
                ),
                "runtime_adapter_invoked": runtime_invoked,
                "runtime_version": (
                    observed_result.runtime_version if observed_result is not None else None
                ),
                "run_id": (
                    observed_result.run_id
                    if observed_result is not None
                    else record.run_id
                ),
                "session_id": (
                    run_handle.session_id if run_handle is not None else None
                ),
                "runtime_status": (
                    observed_result.status.value
                    if observed_result is not None
                    else None
                ),
                "model": observed_result.model if observed_result is not None else None,
                "provider": observed_result.provider if observed_result is not None else None,
                "events": list(extra.get("observed_events", [])),
                "output": observed_result.summary if observed_result is not None else None,
                "error": observed_result.error if observed_result is not None else None,
                "provenance": (
                    dict(observed_result.provenance) if observed_result is not None else {}
                ),
            }
            if "usage" in extra:
                observed["usage"] = extra["usage"].model_dump(mode="json")
            d["observed"] = observed
        return d

    @staticmethod
    def _summary_delegation(
        record: TaskRecord,
        dr: DelegationResult,
        detail: str = "",
        outcome: str | None = None,
        **extra,
    ) -> dict:
        child_runs = [
            {"child_id": child.child_id, "run_id": run_id}
            for child in dr.children
            for run_id in child.attempt_run_ids
        ]
        evaluation_statuses = [
            child.eval_result.overall.value
            for child in dr.children
            if child.eval_result is not None
        ]
        verification_status = extra.get("verification_status")
        if verification_status is None:
            if any(status == "fail" for status in evaluation_statuses):
                verification_status = "fail"
            elif dr.children and len(evaluation_statuses) == len(dr.children):
                verification_status = "pass"
        usage = None
        total_input = dr.total_input_tokens
        total_output = dr.total_output_tokens
        if dr.children and total_input is not None and total_output is not None:
            usage = {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "estimated_cost_usd": dr.total_cost_usd,
            }
        return {
            "task_id": record.task.id,
            "run_id": None,
            "outcome": outcome or dr.overall_status,
            "route": "delegation",
            "retry_count": record.retry_count,
            "detail": detail,
            "child_runs": child_runs,
            "files_changed": extra.get("files_changed"),
            "workspace_root": None,
            "isolation_level": "workspace",
            "verification_status": verification_status,
            "delegation": {
                "children": len(dr.children),
                "successful": dr.successful,
                "failed": dr.failed,
                "duration_ms": dr.duration_ms,
            },
            "usage": usage,
        }

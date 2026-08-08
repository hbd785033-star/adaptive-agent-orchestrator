"""
Main orchestration engine — ties all layers together.

Flow per task:
    receive → profile → route → budget/approval check
    → allocate workspace → submit to runtime → stream events
    → eval gate → record telemetry → transition to COMPLETED or FAILED/RETRY
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import structlog

from adapters.runtime import AgentRuntime
from contracts.evaluation import EvalStatus
from contracts.result import RunStatus
from contracts.task import TaskContract, TaskType
from evals.gate import DeterministicEvalGate
from orchestrator.budget import ApprovalGate, BudgetConfig, BudgetState
from orchestrator.profiler import TaskProfiler
from orchestrator.router import RuleRouter
from orchestrator.state_machine import StateMachine, TaskRecord, TaskStatus
from orchestrator.workspace import WorkspaceManager
from storage.database import Database
from telemetry.events import TelemetryRecorder

log = structlog.get_logger(__name__)


class Orchestrator:
    """
    Top-level controller. One instance per process.

    Usage:
        async with Orchestrator.build(runtime=adapter) as orch:
            result = await orch.run(task)
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        db: Database,
        state_machine: StateMachine,
        profiler: TaskProfiler,
        router: RuleRouter,
        budget_config: BudgetConfig,
        approval_gate: ApprovalGate,
        eval_gate: DeterministicEvalGate,
        workspace_manager: WorkspaceManager | None,
        telemetry: TelemetryRecorder,
    ) -> None:
        self._runtime = runtime
        self._db = db
        self._sm = state_machine
        self._profiler = profiler
        self._router = router
        self._budget_config = budget_config
        self._approval = approval_gate
        self._eval_gate = eval_gate
        self._wm = workspace_manager
        self._tel = telemetry

    @classmethod
    async def build(
        cls,
        runtime: AgentRuntime,
        db_path: str = "data/orchestrator.db",
        repo_path: str = ".",
        policy_path: str = "policies/default.yaml",
    ) -> "Orchestrator":
        import yaml
        db = Database(Path(db_path))
        await db.connect()

        raw = yaml.safe_load(Path(policy_path).read_text())
        budget_cfg_raw = raw.get("budget", {})
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
                workspace_manager = WorkspaceManager(repo_path=repo_path, policy_path=policy_path)
            except Exception:
                log.warning("workspace_manager_init_failed", repo_path=repo_path)

        return cls(
            runtime=runtime,
            db=db,
            state_machine=StateMachine(db),
            profiler=TaskProfiler(),
            router=RuleRouter(policy_path),
            budget_config=budget_config,
            approval_gate=ApprovalGate(policy_path),
            eval_gate=DeterministicEvalGate(repo_path),
            workspace_manager=workspace_manager,
            telemetry=TelemetryRecorder(db),
        )

    async def close(self) -> None:
        await self._db.close()

    async def __aenter__(self) -> "Orchestrator":
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

    async def _execute(self, record: TaskRecord, budget: BudgetState | None = None, is_retry: bool = False) -> dict:
        task = record.task
        if budget is None:
            budget = BudgetState(task_id=task.id, config=self._budget_config)

        # 1. Profile — skip on retry (record already went through PROFILED once)
        if not is_retry:
            profile = self._profiler.profile(task)
            await record.transition(record.status.__class__.PROFILED)
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

        # 4. Workspace allocation (write tasks only)
        if self._wm and self._wm.needs_worktree(task.task_type.value):
            if decision.route == "delegation":
                for i in range(budget.config.max_children):
                    child_id = f"child-{i+1}"
                    wt = self._wm.allocate(task.id, child_id)
                    log.info("worktree_allocated", task_id=task.id, child_id=child_id,
                             worktree_path=str(wt.worktree_path))

        # 5. Submit + stream events
        handle = await self._runtime.submit(task)
        await record.mark_running(handle.run_id)
        budget.calls_used += 1
        await self._tel.record(task.id, "task_submitted", {"run_id": handle.run_id}, handle.run_id)

        # Stream events → collect tool calls, watch for approval requests
        last_event_id: str | None = None
        async for event in self._runtime.events(handle.run_id):
            last_event_id = event.id
            await self._tel.record(task.id, f"event_{event.type}", event.payload, handle.run_id)

            if event.type == "approval_request":
                approved = self._approval.prompt_user(
                    event.payload.get("reason", "agent requested approval"), task
                )
                if not approved:
                    await self._runtime.cancel(handle.run_id)
                    await record.mark_failed("approval denied mid-run")
                    return self._summary(record, "failed", "approval denied")

            elif event.type == "usage":
                budget.calls_used += 1

            elif event.type in ("completed", "error"):
                break

        # 6. Get final result
        agent_result = await self._runtime.result(handle.run_id)

        # If agent itself failed (error event), skip eval and mark failed
        if agent_result.status == RunStatus.FAILED:
            await record.mark_failed(agent_result.error or "agent reported failure")
            return self._summary(record, "failed", agent_result.error or "agent error")
        usage = agent_result.usage
        await self._db.append_usage(
            task.id, handle.run_id,
            usage.input_tokens, usage.output_tokens, usage.estimated_cost_usd
        )

        # 7. Eval gate
        await record.mark_evaluating()
        eval_result = await self._eval_gate.run(task, agent_result, budget)
        await self._db.append_eval_result(
            task.id, handle.run_id, eval_result.overall.value,
            [c.model_dump() for c in eval_result.checks]
        )
        await self._tel.record(task.id, "eval_completed", {
            "overall": eval_result.overall.value,
            "failed_checks": [c.name for c in eval_result.failed_checks()],
        }, handle.run_id)

        # 8. Outcome
        if eval_result.overall == EvalStatus.PASS:
            await record.mark_completed()
            if self._wm:
                for wt_record in list(self._wm.list_records()):
                    if wt_record.task_id == task.id:
                        self._wm.mark_merging(task.id, wt_record.child_id)
            return self._summary(record, "completed", "", eval_result=eval_result, usage=usage)

        # Eval failed — retry if budget allows
        retry_violation = budget.check_retries()
        if not retry_violation:
            await record.mark_failed("eval_failed")
            await record.mark_retry()
            log.info("task_retrying", task_id=task.id, retry=record.retry_count)
            budget.retries_used += 1
            # On retry: transition back to ROUTED so _execute can re-enter RUNNING
            await record.mark_routed(decision.route)
            return await self._execute(record, budget=budget, is_retry=True)

        # Exhausted retries
        await record.mark_failed("eval_failed_no_retries")
        if self._wm:
            for wt_record in list(self._wm.list_records()):
                if wt_record.task_id == task.id:
                    self._wm.abandon(task.id, wt_record.child_id)

        return self._summary(record, "failed", "eval failed and retries exhausted", eval_result=eval_result)

    @staticmethod
    def _summary(record: TaskRecord, outcome: str, detail: str, **extra) -> dict:
        d: dict = {
            "task_id": record.task.id,
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
        if "usage" in extra:
            u = extra["usage"]
            d["usage"] = {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "total_tokens": u.total_tokens,
            }
        return d

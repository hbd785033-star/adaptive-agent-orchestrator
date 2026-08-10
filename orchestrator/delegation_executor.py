"""
DelegationExecutor — true parallel multi-agent execution.

Responsibilities
----------------
- Submit N child TaskContracts concurrently via asyncio.gather
- Stream events per child independently
- Run Eval Gate per child after completion
- Handle partial failure: only retry failed children, not successful ones
- Produce a DelegationResult with per-child ChildExecution records

This module owns all multi-agent concurrency logic so engine.py stays
focused on the single-task flow. The engine calls:

    result = await executor.execute(parent, children, plan, budget)

and gets back a DelegationResult it can eval-aggregate and record.
"""
from __future__ import annotations

import asyncio
import time

import structlog

from adapters.runtime import AgentRuntime
from contracts.delegation import ChildExecution, DelegationResult
from contracts.evaluation import EvalResult, EvalStatus
from contracts.result import AgentResult, RunHandle, RunStatus
from contracts.task import TaskContract
from evals.gate import DeterministicEvalGate
from orchestrator.budget import BudgetState
from orchestrator.execution_policy import ExecutionPolicy
from telemetry.events import TelemetryRecorder

log = structlog.get_logger(__name__)


class DelegationExecutor:
    """
    Executes a list of child TaskContracts in true parallel.

    Each child:
      1. submit() → RunHandle
      2. stream events() → watch for approval / usage / error
      3. wait() for terminal result
      4. run DeterministicEvalGate
      5. retry once if eval fails and budget allows

    Failed children are retried independently; successful ones are not re-run.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        eval_gate: DeterministicEvalGate,
        telemetry: TelemetryRecorder,
        execution_policy: ExecutionPolicy | None = None,
    ) -> None:
        self._runtime = runtime
        self._eval_gate = eval_gate
        self._tel = telemetry
        self._execution_policy = execution_policy

    async def execute(
        self,
        parent_task_id: str,
        children: list[TaskContract],
        budget: BudgetState,
    ) -> DelegationResult:
        """
        Run all children concurrently. Return DelegationResult.

        Parameters
        ----------
        parent_task_id:
            ID of the parent task (for telemetry correlation).
        children:
            Pre-built, Prompt-Guard-augmented child TaskContracts.
        budget:
            Shared budget — children decrement calls_used together.
        """
        started = int(time.monotonic() * 1000)
        result = DelegationResult(
            parent_task_id=parent_task_id,
            started_at_ms=started,
        )

        await self._tel.record(
            parent_task_id,
            "delegation_started",
            {
                "child_count": len(children),
                "decomposition_mode": (
                    "explicit"
                    if children and all(
                        child.context.get("_decomposition_mode") == "explicit"
                        for child in children
                    )
                    else "replicated_goal"
                ),
            },
        )

        has_dependencies = any(child.context.get("_dependencies") for child in children)
        if has_dependencies:
            result.children = await self._execute_dag(parent_task_id, children, budget)
        else:
            # ── Round 1: run all independent children concurrently ────────────
            tasks = [
                self._run_child(parent_task_id, child, budget)
                for child in children
            ]
            executions: list[ChildExecution] = await asyncio.gather(*tasks)
            result.children = list(executions)

            # ── Round 2: retry failed children (once, independently) ─────────
            failed = result.failed_children()
            if (
                failed
                and budget.check_retries() is None
                and budget.can_submit_calls(1) is None
            ):
                budget.retries_used += 1
                retry_tasks = [
                    self._run_child(parent_task_id, _child_for(c, children), budget, retry=True)
                    for c in failed
                ]
                retried: list[ChildExecution] = await asyncio.gather(*retry_tasks)

                id_map = {c.child_id: c for c in retried}
                result.children = [id_map.get(c.child_id, c) for c in result.children]

        # ── Aggregate ─────────────────────────────────────────────────────────
        result.finished_at_ms = int(time.monotonic() * 1000)
        result.aggregate_result = self._aggregate(result)

        await self._tel.record(
            parent_task_id,
            "delegation_completed",
            {
                "status": result.overall_status,
                "successful": result.successful,
                "failed": result.failed,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
                "duration_ms": result.duration_ms,
            },
        )

        log.info(
            "delegation_completed",
            parent_task_id=parent_task_id,
            status=result.overall_status,
            successful=result.successful,
            failed=result.failed,
            duration_ms=result.duration_ms,
        )
        return result

    async def _execute_dag(
        self,
        parent_task_id: str,
        children: list[TaskContract],
        budget: BudgetState,
    ) -> list[ChildExecution]:
        """Execute explicit subtasks in dependency-respecting parallel waves."""
        by_subtask = {
            child.context.get("_subtask_id", child.context.get("_child_id", child.id)): child
            for child in children
        }
        order = list(by_subtask)
        pending = dict(by_subtask)
        outcomes: dict[str, ChildExecution] = {}

        while pending:
            blocked = [
                subtask_id
                for subtask_id, child in pending.items()
                if any(
                    dependency in outcomes and not outcomes[dependency].succeeded
                    for dependency in child.context.get("_dependencies", [])
                )
            ]
            for subtask_id in blocked:
                child = pending.pop(subtask_id)
                outcomes[subtask_id] = ChildExecution(
                    child_id=child.context.get("_child_id", child.id[:8]),
                    run_id="<dependency>",
                    status="failed",
                )

            ready_ids = [
                subtask_id
                for subtask_id, child in pending.items()
                if all(
                    dependency in outcomes and outcomes[dependency].succeeded
                    for dependency in child.context.get("_dependencies", [])
                )
            ]
            if not ready_ids:
                # Contract validation prevents cycles; this is a fail-closed
                # runtime guard for malformed externally constructed children.
                for subtask_id, child in list(pending.items()):
                    outcomes[subtask_id] = ChildExecution(
                        child_id=child.context.get("_child_id", child.id[:8]),
                        run_id="<dependency>",
                        status="failed",
                    )
                    del pending[subtask_id]
                break

            wave_children = [pending.pop(subtask_id) for subtask_id in ready_ids]
            wave_results = list(await asyncio.gather(*[
                self._run_child(parent_task_id, child, budget)
                for child in wave_children
            ]))

            failed_indexes = [index for index, execution in enumerate(wave_results) if not execution.succeeded]
            if (
                failed_indexes
                and budget.check_retries() is None
                and budget.can_submit_calls(1) is None
            ):
                budget.retries_used += 1
                retried = await asyncio.gather(*[
                    self._run_child(parent_task_id, wave_children[index], budget, retry=True)
                    for index in failed_indexes
                ])
                for index, execution in zip(failed_indexes, retried, strict=True):
                    wave_results[index] = execution

            for subtask_id, execution in zip(ready_ids, wave_results, strict=True):
                outcomes[subtask_id] = execution

        return [outcomes[subtask_id] for subtask_id in order]

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _run_child(
        self,
        parent_task_id: str,
        task: TaskContract,
        budget: BudgetState,
        retry: bool = False,
    ) -> ChildExecution:
        child_id = task.context.get("_child_id", task.id[:8])
        t0 = time.monotonic()

        log.info(
            "child_starting",
            parent_task_id=parent_task_id,
            child_id=child_id,
            retry=retry,
        )

        violation = budget.reserve_calls(1)
        if violation:
            return ChildExecution(
                child_id=child_id,
                run_id="<budget>",
                status="failed",
                duration_ms=int((time.monotonic() - t0) * 1000),
                retry_count=1 if retry else 0,
            )

        try:
            try:
                handle: RunHandle = await self._runtime.submit(task)
            except Exception:
                budget.release_reserved_call()
                raise
            budget.commit_reserved_call()

            await self._tel.record(
                parent_task_id,
                "child_submitted",
                {"child_id": child_id, "run_id": handle.run_id, "retry": retry},
                handle.run_id,
            )

            # Stream events — watch for errors / usage
            async for event in self._runtime.events(handle):
                await self._tel.record(
                    parent_task_id,
                    f"child_event_{event.type}",
                    event.payload,
                    handle.run_id,
                )
                if event.type == "approval_request" and self._execution_policy:
                    decision = self._execution_policy.authorize_event(
                        task,
                        event.payload,
                        calls_used=max(0, budget.calls_used - 1),
                        approval=False,
                    )
                    if not decision.allowed:
                        await self._runtime.cancel(handle)
                        return ChildExecution(
                            child_id=child_id,
                            run_id=handle.run_id,
                            status="failed",
                            duration_ms=int((time.monotonic() - t0) * 1000),
                            retry_count=1 if retry else 0,
                        )
                if event.type in ("completed", "error"):
                    break

            agent_result: AgentResult = await self._runtime.wait(handle)
            usage = await self._runtime.usage(handle)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if agent_result.status == RunStatus.FAILED:
                log.warning(
                    "child_agent_failed",
                    child_id=child_id,
                    error=agent_result.error,
                )
                return ChildExecution(
                    child_id=child_id,
                    run_id=handle.run_id,
                    status="failed",
                    result=agent_result,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    estimated_cost_usd=usage.estimated_cost_usd or 0.0,
                    retry_count=1 if retry else 0,
                    duration_ms=duration_ms,
                )

            # Eval gate
            eval_result: EvalResult = await self._eval_gate.run(
                task, agent_result, budget
            )
            status = "completed" if eval_result.overall == EvalStatus.PASS else "failed"

            log.info(
                "child_completed",
                child_id=child_id,
                eval=eval_result.overall.value,
                duration_ms=duration_ms,
            )

            return ChildExecution(
                child_id=child_id,
                run_id=handle.run_id,
                status=status,
                result=agent_result,
                eval_result=eval_result,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                estimated_cost_usd=usage.estimated_cost_usd or 0.0,
                retry_count=1 if retry else 0,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log.exception("child_exception", child_id=child_id, exc=str(exc))
            return ChildExecution(
                child_id=child_id,
                run_id="<error>",
                status="failed",
                duration_ms=duration_ms,
                retry_count=1 if retry else 0,
            )

    @staticmethod
    def _aggregate(result: DelegationResult) -> AgentResult | None:
        """
        Merge successful children into a single parent AgentResult.
        Currently: concatenate summaries, union files_changed.
        Replace with smarter merge logic in v2.
        """
        from contracts.result import AgentResult, RunStatus, Usage

        successful = result.successful_children()
        if not successful:
            return None

        all_files: list[str] = []
        summaries: list[str] = []
        for child in successful:
            if child.result:
                all_files.extend(child.result.files_changed)
                if child.result.summary:
                    summaries.append(f"[{child.child_id}] {child.result.summary}")

        return AgentResult(
            run_id="<aggregated>",
            task_id=result.parent_task_id,
            status=RunStatus.COMPLETED,
            usage=Usage(
                input_tokens=result.total_input_tokens,
                output_tokens=result.total_output_tokens,
                total_tokens=result.total_input_tokens + result.total_output_tokens,
                estimated_cost_usd=result.total_cost_usd,
            ),
            files_changed=list(dict.fromkeys(all_files)),  # dedup, preserve order
            summary="\n".join(summaries),
        )


def _child_for(execution: ChildExecution, children: list[TaskContract]) -> TaskContract:
    """Find the TaskContract matching a ChildExecution by child_id."""
    for c in children:
        if c.context.get("_child_id") == execution.child_id:
            return c
    raise KeyError(f"No child contract found for child_id={execution.child_id}")

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
        submission_approval: bool | None = None,
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
            result.children = await self._execute_dag(
                parent_task_id, children, budget, submission_approval
            )
        else:
            # ── Round 1: run all independent children concurrently ────────────
            tasks = [
                self._run_child(
                    parent_task_id,
                    child,
                    budget,
                    submission_approval=submission_approval,
                )
                for child in children
            ]
            executions: list[ChildExecution] = await asyncio.gather(*tasks)
            result.children = list(executions)

            # ── Round 2: retry failed children (once, independently) ─────────
            failed = [child for child in result.failed_children() if child.status == "failed"]
            if (
                failed
                and budget.check_retries() is None
                and budget.can_submit_calls(1) is None
            ):
                budget.retries_used += 1
                retry_tasks = [
                    self._run_child(
                        parent_task_id,
                        _child_for(c, children),
                        budget,
                        retry=True,
                        submission_approval=submission_approval,
                    )
                    for c in failed
                ]
                retried: list[ChildExecution] = await asyncio.gather(*retry_tasks)

                id_map = {c.child_id: c for c in retried}
                result.children = [
                    _merge_retry_attempts(c, id_map[c.child_id])
                    if c.child_id in id_map
                    else c
                    for c in result.children
                ]

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
        submission_approval: bool | None,
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
                    run_id=None,
                    status="failed",
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_usd=0.0,
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
                        run_id=None,
                        status="failed",
                        input_tokens=0,
                        output_tokens=0,
                        estimated_cost_usd=0.0,
                    )
                    del pending[subtask_id]
                break

            wave_children = [pending.pop(subtask_id) for subtask_id in ready_ids]
            wave_results = list(await asyncio.gather(*[
                self._run_child(
                    parent_task_id,
                    child,
                    budget,
                    submission_approval=submission_approval,
                )
                for child in wave_children
            ]))

            failed_indexes = [
                index
                for index, execution in enumerate(wave_results)
                if execution.status == "failed"
            ]
            if (
                failed_indexes
                and budget.check_retries() is None
                and budget.can_submit_calls(1) is None
            ):
                budget.retries_used += 1
                retried = await asyncio.gather(*[
                    self._run_child(
                        parent_task_id,
                        wave_children[index],
                        budget,
                        retry=True,
                        submission_approval=submission_approval,
                    )
                    for index in failed_indexes
                ])
                for index, execution in zip(failed_indexes, retried, strict=True):
                    wave_results[index] = _merge_retry_attempts(
                        wave_results[index], execution
                    )

            for subtask_id, execution in zip(ready_ids, wave_results, strict=True):
                outcomes[subtask_id] = execution

        return [outcomes[subtask_id] for subtask_id in order]

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _cancel_and_confirm_terminal(self, handle: RunHandle) -> AgentResult:
        """Cancel one child and prove it is terminal before cleanup is permitted."""
        cancel_error: Exception | None = None
        try:
            await self._runtime.cancel(handle)
        except Exception as exc:
            cancel_error = exc
        try:
            result = await self._runtime.wait(handle)
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
        return result

    async def _run_child(
        self,
        parent_task_id: str,
        task: TaskContract,
        budget: BudgetState,
        retry: bool = False,
        submission_approval: bool | None = None,
    ) -> ChildExecution:
        child_id = task.context.get("_child_id", task.id[:8])
        t0 = time.monotonic()
        handle: RunHandle | None = None
        terminal_confirmed = False
        agent_result: AgentResult | None = None
        usage = None

        log.info(
            "child_starting",
            parent_task_id=parent_task_id,
            child_id=child_id,
            retry=retry,
        )

        if self._execution_policy:
            submission_decision = self._execution_policy.authorize_submission(
                task,
                calls_used=budget.calls_used + budget.calls_reserved,
                approval=submission_approval,
            )
            if not submission_decision.allowed:
                return ChildExecution(
                    child_id=child_id,
                    run_id=None,
                    status="cancelled",
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_usd=0.0,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    retry_count=1 if retry else 0,
                )

        violation = budget.reserve_calls(1)
        if violation:
            return ChildExecution(
                child_id=child_id,
                run_id=None,
                status="failed",
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                duration_ms=int((time.monotonic() - t0) * 1000),
                retry_count=1 if retry else 0,
            )

        try:
            try:
                handle = await self._runtime.submit(task)
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
                    await self._cancel_and_confirm_terminal(handle)
                    terminal_confirmed = True
                    return ChildExecution(
                        child_id=child_id,
                        run_id=handle.run_id,
                        status="cancelled",
                        duration_ms=int((time.monotonic() - t0) * 1000),
                        retry_count=1 if retry else 0,
                    )
                if event.type in ("completed", "error"):
                    break

            agent_result = await self._runtime.wait(handle)
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
            duration_ms = int((time.monotonic() - t0) * 1000)

            if agent_result.status != RunStatus.COMPLETED:
                log.warning(
                    "child_agent_terminal_non_completed",
                    child_id=child_id,
                    status=agent_result.status.value,
                    error=agent_result.error,
                )
                usage = agent_result.usage
                return ChildExecution(
                    child_id=child_id,
                    run_id=handle.run_id,
                    status=agent_result.status.value,
                    result=agent_result,
                    input_tokens=usage.input_tokens if usage is not None else None,
                    output_tokens=usage.output_tokens if usage is not None else None,
                    estimated_cost_usd=(
                        usage.estimated_cost_usd if usage is not None else None
                    ),
                    retry_count=1 if retry else 0,
                    duration_ms=duration_ms,
                )

            usage = await self._runtime.usage(handle)
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
                estimated_cost_usd=usage.estimated_cost_usd,
                retry_count=1 if retry else 0,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log.exception("child_exception", child_id=child_id, exc=str(exc))
            if handle is not None and not terminal_confirmed:
                try:
                    await self._cancel_and_confirm_terminal(handle)
                    terminal_confirmed = True
                except Exception as quiescence_error:
                    log.error(
                        "child_workspace_quarantined",
                        child_id=child_id,
                        run_id=handle.run_id,
                        reason=str(quiescence_error),
                    )
                    return ChildExecution(
                        child_id=child_id,
                        run_id=handle.run_id,
                        status="timeout",
                        duration_ms=duration_ms,
                        retry_count=1 if retry else 0,
                    )
            return ChildExecution(
                child_id=child_id,
                run_id=handle.run_id if handle is not None else None,
                status="failed",
                result=agent_result,
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                estimated_cost_usd=(
                    usage.estimated_cost_usd if usage is not None else None
                ),
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
                if child.result.files_changed is not None:
                    all_files.extend(child.result.files_changed)
                if child.result.summary:
                    summaries.append(f"[{child.child_id}] {child.result.summary}")

        total_input = result.total_input_tokens
        total_output = result.total_output_tokens
        aggregate_usage = None
        if total_input is not None and total_output is not None:
            aggregate_usage = Usage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                estimated_cost_usd=result.total_cost_usd,
            )

        return AgentResult(
            run_id=None,
            task_id=result.parent_task_id,
            status=RunStatus.COMPLETED,
            usage=aggregate_usage,
            files_changed=list(dict.fromkeys(all_files)),  # dedup, preserve order
            summary="\n".join(summaries),
        )


def _child_for(execution: ChildExecution, children: list[TaskContract]) -> TaskContract:
    """Find the TaskContract matching a ChildExecution by child_id."""
    for c in children:
        if c.context.get("_child_id") == execution.child_id:
            return c
    raise KeyError(f"No child contract found for child_id={execution.child_id}")


def _merge_retry_attempts(
    previous: ChildExecution, current: ChildExecution
) -> ChildExecution:
    """Keep the current attempt while monotonically retaining observed run IDs."""
    if previous.child_id != current.child_id:
        raise ValueError("cannot merge retry provenance for different children")
    attempt_run_ids = list(
        dict.fromkeys(previous.attempt_run_ids + current.attempt_run_ids)
    )
    return current.model_copy(update={"attempt_run_ids": attempt_run_ids})

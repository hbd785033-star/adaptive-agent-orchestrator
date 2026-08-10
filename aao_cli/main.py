"""CLI entry point — typer-based."""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="aao", help="Adaptive Agent Orchestrator — control plane for Hermes")
console = Console()


@app.command()
def run(
    goal: str = typer.Argument(..., help="Task goal"),
    task_type: str = typer.Option("general", "--type", "-t", help="Task type"),
    risk: int = typer.Option(1, "--risk", "-r", help="Risk level 1-4"),
    complexity: int = typer.Option(1, "--complexity", "-c", help="Complexity 1-5"),
    allowed_paths: list[str] | None = typer.Option(None, "--allow", "-a", help="Allowed file paths (glob)"),  # noqa: B008
    forbidden: list[str] | None = typer.Option(None, "--forbid", "-f", help="Forbidden actions"),  # noqa: B008
    criteria: list[str] | None = typer.Option(None, "--criterion", "-x", help="Success criteria"),  # noqa: B008
    hermes_url: str = typer.Option("ws://localhost:4999", "--hermes", help="Hermes Gateway URL"),
    hermes_key: str | None = typer.Option(None, "--key", help="Hermes API key"),
    policy: str = typer.Option("policies/default.yaml", "--policy", help="Policy YAML path"),
    repo: str = typer.Option(".", "--repo", help="Repo path for worktree and evals"),
    mock: bool = typer.Option(False, "--mock", help="Use mock adapter (no live Hermes)"),
    record_out: Path | None = typer.Option(None, "--record-out", help="Write ExecutionRecord 0.1 JSON"),  # noqa: B008
) -> None:
    """Submit a task to the orchestrator."""
    asyncio.run(_run_task(
        goal=goal,
        task_type=task_type,
        risk=risk,
        complexity=complexity,
        allowed_paths=allowed_paths or [],
        forbidden=forbidden or [],
        criteria=criteria or [],
        hermes_url=hermes_url,
        hermes_key=hermes_key,
        policy=policy,
        repo=repo,
        mock=mock,
        record_out=record_out,
    ))


async def _run_task(**kwargs) -> None:  # noqa: ANN003
    from contracts.task import RiskLevel, TaskContract, TaskType
    from orchestrator.engine import Orchestrator

    try:
        task_type_enum = TaskType(kwargs["task_type"])
    except ValueError:
        task_type_enum = TaskType.GENERAL

    try:
        risk_enum = RiskLevel(kwargs["risk"])
    except ValueError:
        risk_enum = RiskLevel.LOW

    task = TaskContract(
        goal=kwargs["goal"],
        task_type=task_type_enum,
        risk=risk_enum,
        complexity=kwargs["complexity"],
        allowed_paths=kwargs["allowed_paths"],
        forbidden_actions=kwargs["forbidden"],
        success_criteria=kwargs["criteria"],
    )

    if kwargs["mock"]:
        from adapters.mock import MockHermesAdapter
        runtime: object = MockHermesAdapter()
        runtime.enqueue_scenario("pass", summary="mock run completed")  # type: ignore[attr-defined]
    else:
        from adapters.hermes.gateway import HermesAdapter
        runtime = HermesAdapter(url=kwargs["hermes_url"], api_key=kwargs["hermes_key"])
        await runtime.connect()  # type: ignore[attr-defined]

    started_at = datetime.now(UTC)
    async with await Orchestrator.build(
        runtime=runtime,  # type: ignore[arg-type]
        policy_path=kwargs["policy"],
        repo_path=kwargs["repo"],
    ) as orch:
        result = await orch.run(task)

    if kwargs.get("record_out"):
        from contracts.execution import ExecutionRecord

        finished_at = datetime.now(UTC)
        usage = result.get("usage", {})
        status = "completed" if result.get("outcome") == "completed" else "failed"
        record = ExecutionRecord(
            task_id=task.id,
            run_id=str(result.get("run_id") or f"aao-{task.id}"),
            model="mock" if kwargs["mock"] else "unknown",
            provider="fixture" if kwargs["mock"] else "unknown",
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            latency_seconds=(finished_at - started_at).total_seconds(),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cached_tokens=0,
            cost_usd=float(usage.get("estimated_cost_usd") or 0.0),
            tool_calls=[],
            files_changed=list(result.get("files_changed", [])),
            output=str(result.get("detail") or result.get("summary") or ""),
            workspace_root=result.get("workspace_root"),
            isolation_level=str(result.get("isolation_level", "none")),
            metadata={
                "route": result.get("route"),
                "retries": result.get("retry_count", 0),
                "verification_status": result.get("eval", {}).get("overall", "not_run"),
                "trial": 1,
                "identity_observed": kwargs["mock"],
            },
        )
        record.export(kwargs["record_out"])

    _print_result(result)


def _print_result(result: dict) -> None:
    outcome = result.get("outcome", "unknown")
    icon = "✅" if outcome == "completed" else "❌"
    console.print(
        f"\n{icon}  Outcome: [bold]{outcome}[/bold]"
        f"   Route: {result.get('route')}"
        f"   Retries: {result.get('retry_count', 0)}"
    )
    if result.get("detail"):
        console.print(f"   Detail: {result['detail']}")
    if u := result.get("usage"):
        console.print(
            f"   Tokens: in={u['input_tokens']} out={u['output_tokens']} total={u['total_tokens']}"
        )
    if e := result.get("eval"):
        overall = e.get("overall", "?")
        failed = e.get("failed_checks", [])
        color = "green" if overall == "pass" else "red"
        console.print(f"   Eval: [{color}]{overall}[/{color}]", end="")
        if failed:
            console.print(f"   Failed checks: {failed}")
        else:
            console.print()


@app.command(name="export-record")
def export_record(
    source: Path = typer.Argument(..., exists=True, readable=True),  # noqa: B008
    out: Path = typer.Option(..., "--out", "-o"),  # noqa: B008
) -> None:
    """Validate and export an ExecutionRecord schema 0.1 JSON document."""
    from contracts.execution import ExecutionRecord

    record = ExecutionRecord.model_validate_json(source.read_text(encoding="utf-8"))
    destination = record.export(out)
    console.print(str(destination))


@app.command()
def spike(
    url: str = typer.Option("ws://localhost:4999", "--url", help="Hermes Gateway WebSocket URL"),
    api_key: str | None = typer.Option(None, "--key", help="Optional bearer token"),
    out: str = typer.Option("spike/phase0_report.yaml", "--out", help="Output report path"),
) -> None:
    """Run Phase 0 Hermes Gateway compatibility check."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "spike/phase0_gateway.py", "--url", url, "--out", out]
        + (["--api-key", api_key] if api_key else []),
    )
    raise SystemExit(result.returncode)


@app.command()
def history(
    policy_version: str | None = typer.Option(None, "--policy-version", "-p"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Show routing decision history."""
    asyncio.run(_show_history(policy_version, limit))


async def _show_history(policy_version: str | None, limit: int) -> None:
    from storage.database import Database
    db = Database()
    await db.connect()
    rows = await db.get_routing_history(policy_version)
    await db.close()

    table = Table("task_id", "route", "policy_version", "decided_at")
    for row in rows[-limit:]:
        table.add_row(row["task_id"][:12], row["route"], row["policy_version"], row["decided_at"][:19])
    console.print(table)


@app.command(name="eval-routing")
def eval_routing(
    dataset: str = typer.Option("datasets/routing_cases.yaml", "--dataset", "-d", help="Routing cases YAML"),
    policy: str = typer.Option("policies/default.yaml", "--policy", help="Policy YAML path"),
) -> None:
    """Score the current routing policy against labelled test cases."""
    asyncio.run(_eval_routing(dataset, policy))


async def _eval_routing(dataset_path: str, policy_path: str) -> None:
    import yaml

    from contracts.task import RiskLevel, TaskContract, TaskType
    from orchestrator.profiler import TaskProfiler
    from orchestrator.router import RuleRouter

    with open(dataset_path) as fh:
        cases = yaml.safe_load(fh.read()).get("cases", [])
    if not cases:
        console.print("[yellow]No cases found in dataset.[/yellow]")
        return

    router = RuleRouter(policy_path)
    profiler = TaskProfiler()

    passed = failed = skipped = 0
    table = Table("id", "expected", "actual", "match", "reasons", show_lines=False)

    for case in cases:
        task_raw = case.get("task", {})
        expected = case.get("expected_route", "")
        case_id = case.get("id", "?")

        try:
            task = TaskContract(
                goal=task_raw.get("goal", "test task"),
                task_type=TaskType(task_raw.get("task_type", "general")),
                risk=RiskLevel(task_raw.get("risk", 1)),
                complexity=task_raw.get("complexity", 1),
                success_criteria=task_raw.get("success_criteria", []),
                forbidden_actions=task_raw.get("forbidden_actions", []),
            )
            profile = profiler.profile(task)
            decision = router.route(
                task,
                independent_subtask_count=task_raw.get("independent_subtask_count",
                                                        profile.independent_subtask_count),
                has_sequential_dependency=task_raw.get("has_sequential_dependency",
                                                        profile.has_sequential_dependency),
                affected_module_count=task_raw.get("affected_module_count",
                                                    profile.affected_module_count),
            )
            match = decision.route == expected
            mark = "[green]✓[/green]" if match else "[red]✗[/red]"
            reasons_str = "; ".join(decision.reasons[:2])
            table.add_row(case_id, expected, decision.route, mark, reasons_str)
            if match:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            table.add_row(case_id, expected, "ERROR", "[yellow]?[/yellow]", str(exc))
            skipped += 1

    console.print(table)
    total = passed + failed + skipped
    color = "green" if failed == 0 else "red"
    console.print(
        f"\n[{color}]Results: {passed}/{total} passed[/{color}]"
        f"  ({failed} failed, {skipped} errored)"
        f"  Policy: {router.policy_version}"
    )
    if failed > 0:
        raise SystemExit(1)


@app.command()
def stats(
    db_path: str = typer.Option("data/orchestrator.db", "--db", help="Database path"),
    policy_version: str | None = typer.Option(None, "--policy-version", "-p", help="Filter by policy version"),
) -> None:
    """Show task outcome and routing statistics from telemetry."""
    if not Path(db_path).exists():
        console.print(f"[yellow]No database found at {db_path}. Run a task first.[/yellow]")
        return

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Routing breakdown
    if policy_version:
        routing_rows = con.execute(
            "SELECT route, policy_version, COUNT(*) as n FROM routing_decisions"
            " WHERE policy_version = ? GROUP BY policy_version, route ORDER BY route",
            (policy_version,),
        ).fetchall()
    else:
        routing_rows = con.execute(
            "SELECT route, policy_version, COUNT(*) as n FROM routing_decisions"
            " GROUP BY policy_version, route ORDER BY policy_version, route"
        ).fetchall()

    if routing_rows:
        rt = Table("policy_version", "route", "count", title="Routing Decisions")
        for r in routing_rows:
            rt.add_row(r["policy_version"], r["route"], str(r["n"]))
        console.print(rt)
    else:
        console.print("[dim]No routing decisions recorded yet.[/dim]")

    # Task outcomes
    outcomes = con.execute(
        "SELECT status, COUNT(*) as n FROM tasks GROUP BY status ORDER BY n DESC"
    ).fetchall()
    if outcomes:
        ot = Table("status", "count", title="Task Outcomes")
        for r in outcomes:
            ot.add_row(r["status"], str(r["n"]))
        console.print(ot)

    # Token usage
    usage = con.execute(
        "SELECT COUNT(*) as runs, SUM(input_tokens) as inp, SUM(output_tokens) as out,"
        " SUM(estimated_cost_usd) as cost FROM usage_records"
    ).fetchone()
    if usage and usage["runs"]:
        console.print(
            f"\nToken usage across {usage['runs']} run(s): "
            f"in={usage['inp'] or 0:,}  out={usage['out'] or 0:,}  "
            f"est_cost=${usage['cost'] or 0:.4f}"
        )

    # Eval pass rate
    evals = con.execute(
        "SELECT overall, COUNT(*) as n FROM eval_results GROUP BY overall"
    ).fetchall()
    if evals:
        total_evals = sum(r["n"] for r in evals)
        passed_n = next((r["n"] for r in evals if r["overall"] == "pass"), 0)
        pct = 100 * passed_n // total_evals if total_evals else 0
        color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
        console.print(f"Eval pass rate: [{color}]{passed_n}/{total_evals} ({pct}%)[/{color}]")

    con.close()


if __name__ == "__main__":
    app()

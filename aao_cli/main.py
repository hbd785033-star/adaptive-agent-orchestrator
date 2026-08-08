"""CLI entry point — typer-based."""
from __future__ import annotations

import asyncio
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
    allowed_paths: list[str] = typer.Option([], "--allow", "-a", help="Allowed file paths (glob)"),
    forbidden: list[str] = typer.Option([], "--forbid", "-f", help="Forbidden actions"),
    criteria: list[str] = typer.Option([], "--criterion", "-x", help="Success criteria"),
    hermes_url: str = typer.Option("ws://localhost:4999", "--hermes", help="Hermes Gateway URL"),
    hermes_key: str | None = typer.Option(None, "--key", help="Hermes API key"),
    policy: str = typer.Option("policies/default.yaml", "--policy", help="Policy YAML path"),
    repo: str = typer.Option(".", "--repo", help="Repo path for worktree and evals"),
    mock: bool = typer.Option(False, "--mock", help="Use mock adapter (no live Hermes)"),
) -> None:
    """Submit a task to the orchestrator."""
    asyncio.run(_run_task(
        goal=goal,
        task_type=task_type,
        risk=risk,
        complexity=complexity,
        allowed_paths=allowed_paths,
        forbidden=forbidden,
        criteria=criteria,
        hermes_url=hermes_url,
        hermes_key=hermes_key,
        policy=policy,
        repo=repo,
        mock=mock,
    ))


async def _run_task(**kwargs) -> None:
    from contracts.task import TaskContract, TaskType, RiskLevel
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
        runtime = MockHermesAdapter()
        runtime.enqueue_scenario("pass", summary="mock run completed")
    else:
        from adapters.hermes.gateway import HermesAdapter
        runtime = HermesAdapter(url=kwargs["hermes_url"], api_key=kwargs["hermes_key"])
        await runtime.connect()

    async with await Orchestrator.build(
        runtime=runtime,
        policy_path=kwargs["policy"],
        repo_path=kwargs["repo"],
    ) as orch:
        result = await orch.run(task)

    _print_result(result)


def _print_result(result: dict) -> None:
    outcome = result.get("outcome", "unknown")
    icon = "✅" if outcome == "completed" else "❌"
    console.print(f"\n{icon}  Outcome: [bold]{outcome}[/bold]   Route: {result.get('route')}   Retries: {result.get('retry_count', 0)}")

    if result.get("detail"):
        console.print(f"   Detail: {result['detail']}")

    if u := result.get("usage"):
        console.print(f"   Tokens: in={u['input_tokens']} out={u['output_tokens']} total={u['total_tokens']}")

    if e := result.get("eval"):
        overall = e.get("overall", "?")
        failed = e.get("failed_checks", [])
        status_color = "green" if overall == "pass" else "red"
        console.print(f"   Eval: [{status_color}]{overall}[/{status_color}]", end="")
        if failed:
            console.print(f"   Failed checks: {failed}")
        else:
            console.print()


@app.command()
def spike(
    url: str = typer.Option("ws://localhost:4999", "--url", help="Hermes Gateway WebSocket URL"),
    api_key: str | None = typer.Option(None, "--key", help="Optional bearer token"),
    out: str = typer.Option("spike/phase0_report.yaml", "--out", help="Output report path"),
) -> None:
    """Run Phase 0 Hermes Gateway compatibility check."""
    import subprocess, sys
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


if __name__ == "__main__":
    app()

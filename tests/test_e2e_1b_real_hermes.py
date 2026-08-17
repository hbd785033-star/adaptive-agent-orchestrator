from __future__ import annotations

import os
from pathlib import Path

import pytest

from adapters.hermes.gateway import HermesAdapter
from contracts.task import RiskLevel, TaskContract, TaskType
from orchestrator.engine import Orchestrator


@pytest.mark.asyncio
async def test_real_no_tool_task_traverses_hmc_aao_hermes(tmp_path: Path) -> None:
    if os.environ.get("AAO_RUN_LIVE_E2E") != "1":
        pytest.skip("LIVE E2E NOT OBSERVED: set AAO_RUN_LIVE_E2E=1 to opt in")

    repo = Path(__file__).resolve().parents[1]
    runtime = HermesAdapter()
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
        repo_path=str(repo),
        db_path=str(tmp_path / "live.db"),
        policy_path=str(repo / "policies" / "default.yaml"),
    ) as orchestrator:
        result = await orchestrator.run(task)

    assert "planned" in result, (
        "live E2E did not reach planned evidence; "
        f"keys={sorted(result.keys())!r}; "
        f"outcome={result.get('outcome')!r}; "
        f"detail={result.get('detail')!r}; "
        f"run_id={result.get('run_id')!r}"
    )
    assert result["planned"]["hmc"]["request_type"] == "PlannerRequest"
    assert result["planned"]["hmc"]["recommendation_type"] == "PlannerRecommendation"
    assert result["planned"]["runtime_selection"]["selected_runtime"] == "hermes"
    assert result["planned"]["runtime_plan"]["executor"] == "hermes"
    assert result["observed"]["runtime_adapter_invoked"] is True
    assert result["observed"]["run_id"]
    assert result["observed"]["runtime_status"] == "completed"
    assert "AAO_E2E_1_OK" in (result["observed"]["output"] or "")
    assert "judgment" not in result

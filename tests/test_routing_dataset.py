"""
Routing regression test — runs the full labelled dataset against the current
policy and fails CI if any case mismatches.

This is a lightweight programmatic equivalent of `aao eval-routing` that
integrates into the pytest suite so routing regressions are caught automatically.
"""
from __future__ import annotations

import pytest
import yaml

from contracts.task import RiskLevel, TaskContract, TaskType
from orchestrator.profiler import TaskProfiler
from orchestrator.router import RuleRouter

DATASET = "datasets/routing_cases.yaml"
POLICY  = "policies/default.yaml"


def load_cases() -> list[dict]:
    with open(DATASET) as fh:
        return yaml.safe_load(fh.read()).get("cases", [])


def _make_task(raw: dict) -> TaskContract:
    return TaskContract(
        goal=raw.get("goal", "test"),
        task_type=TaskType(raw.get("task_type", "general")),
        risk=RiskLevel(raw.get("risk", 1)),
        complexity=raw.get("complexity", 1),
        success_criteria=raw.get("success_criteria", []),
        forbidden_actions=raw.get("forbidden_actions", []),
    )


@pytest.fixture(scope="module")
def router() -> RuleRouter:
    return RuleRouter(POLICY)


@pytest.fixture(scope="module")
def profiler() -> TaskProfiler:
    return TaskProfiler()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise test_routing_case with every case in the dataset."""
    if "routing_case" in metafunc.fixturenames:
        cases = load_cases()
        ids = [c.get("id", f"case-{i}") for i, c in enumerate(cases)]
        metafunc.parametrize("routing_case", cases, ids=ids)


class TestRoutingDataset:
    def test_routing_case(
        self,
        routing_case: dict,
        router: RuleRouter,
        profiler: TaskProfiler,
    ) -> None:
        task_raw = routing_case.get("task", {})
        expected = routing_case.get("expected_route", "")
        case_id = routing_case.get("id", "?")

        task = _make_task(task_raw)
        profile = profiler.profile(task)

        decision = router.route(
            task,
            independent_subtask_count=task_raw.get(
                "independent_subtask_count", profile.independent_subtask_count
            ),
            has_sequential_dependency=task_raw.get(
                "has_sequential_dependency", profile.has_sequential_dependency
            ),
            affected_module_count=task_raw.get(
                "affected_module_count", profile.affected_module_count
            ),
        )

        assert decision.route == expected, (
            f"[{case_id}] expected route={expected!r} "
            f"but got route={decision.route!r}  "
            f"reasons={decision.reasons}"
        )

    def test_policy_version_is_set(self, router: RuleRouter) -> None:
        """Policy version must be a non-empty string for telemetry queries."""
        assert router.policy_version, "policy_version must not be empty"
        assert router.policy_version.startswith("routing-"), (
            f"policy_version should start with 'routing-', got {router.policy_version!r}"
        )

from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.execution_mode import (
    ExecutionModeDecision,
    ExecutionModeDecisionCode,
)
from contracts.runtime_plan import RuntimePlan
from contracts.runtime_selection import (
    RuntimeSelectionDecision,
    RuntimeSelectionDecisionCode,
)
from orchestrator.runtime_plan_composer import compose_runtime_plan


def selection(runtime: str | None = "codex", *, degraded: bool = False):
    return RuntimeSelectionDecision(
        selected_runtime=runtime,
        policy_version="selection-v1",
        decision_code=(
            RuntimeSelectionDecisionCode.SELECTED_DEGRADED_FALLBACK
            if degraded
            else RuntimeSelectionDecisionCode.SELECTED_HEALTHY
        )
        if runtime is not None
        else RuntimeSelectionDecisionCode.NO_ELIGIBLE_CANDIDATE,
        used_degraded_fallback=degraded,
        reasons=["selection reason"],
    )


def mode(runtime: str | None = "codex", value: str | None = "native"):
    code = {
        "native": ExecutionModeDecisionCode.SELECTED_NATIVE,
        "direct": ExecutionModeDecisionCode.SELECTED_DIRECT,
        "delegate": ExecutionModeDecisionCode.SELECTED_DELEGATE,
        "kanban": ExecutionModeDecisionCode.SELECTED_KANBAN,
        None: (
            ExecutionModeDecisionCode.NO_SELECTED_RUNTIME
            if runtime is None
            else ExecutionModeDecisionCode.UNSUPPORTED_MODE
        ),
    }[value]
    return ExecutionModeDecision(
        runtime=runtime,
        mode=value,
        policy_version="mode-v1",
        decision_code=code,
        reasons=["mode reason"],
    )


def test_runtime_selection_decision_rejects_contradictory_state():
    with pytest.raises(ValidationError):
        RuntimeSelectionDecision(
            selected_runtime=None,
            policy_version="selection-v1",
            decision_code=RuntimeSelectionDecisionCode.SELECTED_HEALTHY,
            used_degraded_fallback=False,
        )


def test_execution_mode_decision_rejects_contradictory_state():
    with pytest.raises(ValidationError):
        ExecutionModeDecision(
            runtime="codex",
            mode="direct",
            policy_version="mode-v1",
            decision_code=ExecutionModeDecisionCode.SELECTED_NATIVE,
        )


def test_composes_codex_native_with_exact_provenance_and_reasons():
    plan = compose_runtime_plan(selection(), mode(), plan_policy_version="plan-v1")
    assert plan.executor == "codex"
    assert plan.execution_mode == "native"
    assert plan.selection_policy_version == "selection-v1"
    assert plan.execution_mode_policy_version == "mode-v1"
    assert plan.reasons == ["selection: selection reason", "execution_mode: mode reason"]
    assert plan.fallback is None


@pytest.mark.parametrize("value", ["direct", "delegate", "kanban"])
def test_composes_all_explicit_hermes_modes(value):
    plan = compose_runtime_plan(
        selection("hermes"),
        mode("hermes", value),
        plan_policy_version="plan-v1",
    )
    assert plan.executor == "hermes"
    assert plan.execution_mode == value


def test_rejects_cross_decision_runtime_mismatch():
    with pytest.raises(ValueError):
        compose_runtime_plan(selection("codex"), mode("claude_code"), plan_policy_version="plan-v1")


def test_rejects_missing_selection():
    with pytest.raises(ValueError):
        compose_runtime_plan(selection(None), mode(None, None), plan_policy_version="plan-v1")


def test_rejects_unsupported_mode():
    with pytest.raises(ValueError):
        compose_runtime_plan(selection("future"), mode("future", None), plan_policy_version="plan-v1")


def test_planner_and_independent_reviewer_are_passed_through():
    plan = compose_runtime_plan(
        selection("codex"),
        mode("codex", "native"),
        plan_policy_version="plan-v1",
        planner="model_council",
        reviewer="claude_code",
    )
    assert plan.planner == "model_council"
    assert plan.reviewer == "claude_code"


def test_degraded_selection_does_not_create_execution_fallback():
    plan = compose_runtime_plan(
        selection("codex", degraded=True),
        mode("codex", "native"),
        plan_policy_version="plan-v1",
    )
    assert plan.fallback is None


def test_approval_required_is_passed_through():
    plan = compose_runtime_plan(
        selection("codex"),
        mode("codex", "native"),
        plan_policy_version="plan-v1",
        approval_required=True,
    )
    assert plan.approval_required is True


def test_runtime_plan_rejects_same_executor_and_reviewer_directly():
    with pytest.raises(ValidationError):
        RuntimePlan(
            executor="codex",
            execution_mode="native",
            reviewer="codex",
            policy_version="plan-v1",
        )

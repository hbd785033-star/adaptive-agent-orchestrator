"""Tests for deterministic execution mode policy decisions."""
from __future__ import annotations

import pytest

from adapters.runtime import RuntimeCapabilities
from contracts.execution_mode import ExecutionModePolicy
from contracts.runtime_selection import RuntimeSelectionDecision
from contracts.task_profile import TaskProfile
from orchestrator.execution_mode_policy import select_execution_mode


def selection(runtime: str | None) -> RuntimeSelectionDecision:
    return RuntimeSelectionDecision(
        selected_runtime=runtime,
        policy_version="runtime-selection-v1",
        decision_code="selected_healthy" if runtime else "no_eligible_candidate",
    )


def policy() -> ExecutionModePolicy:
    return ExecutionModePolicy(policy_version="execution-mode-v1")


def test_no_selected_runtime_returns_no_mode() -> None:
    decision = select_execution_mode(
        TaskProfile(),
        selection(None),
        RuntimeCapabilities(),
        policy(),
    )

    assert decision.runtime is None
    assert decision.mode is None
    assert decision.policy_version == "execution-mode-v1"
    assert decision.decision_code == "no_selected_runtime"
    assert decision.reasons == ["no runtime was selected"]


def test_codex_uses_native_mode_without_aao_orchestration() -> None:
    decision = select_execution_mode(
        TaskProfile(parallelizable=True, persistent_execution=True, long_running=True),
        selection("codex"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        policy(),
    )

    assert decision.runtime == "codex"
    assert decision.mode == "native"
    assert decision.decision_code == "selected_native"
    assert decision.reasons == ["selected runtime owns its native execution topology: codex"]


def test_claude_code_uses_native_mode_without_aao_orchestration() -> None:
    decision = select_execution_mode(
        TaskProfile(parallelizable=True),
        selection("claude_code"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        policy(),
    )

    assert decision.runtime == "claude_code"
    assert decision.mode == "native"
    assert decision.decision_code == "selected_native"


def test_hermes_simple_task_uses_direct_mode() -> None:
    decision = select_execution_mode(
        TaskProfile(reasoning_complexity="medium", execution_complexity="low"),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        policy(),
    )

    assert decision.runtime == "hermes"
    assert decision.mode == "direct"
    assert decision.decision_code == "selected_direct"
    assert decision.reasons == ["selected direct execution for runtime: hermes"]


def test_high_reasoning_low_execution_still_uses_direct_mode_for_hermes() -> None:
    decision = select_execution_mode(
        TaskProfile(
            reasoning_complexity="high",
            execution_complexity="low",
            persistent_execution=False,
            long_running=False,
            parallelizable=False,
            cross_role_dependencies=False,
        ),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        policy(),
    )

    assert decision.mode == "direct"
    assert decision.decision_code == "selected_direct"


def test_hermes_parallelizable_task_uses_delegate_when_supported() -> None:
    decision = select_execution_mode(
        TaskProfile(
            execution_complexity="medium",
            parallelizable=True,
            persistent_execution=False,
            long_running=False,
            cross_role_dependencies=False,
        ),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        policy(),
    )

    assert decision.mode == "delegate"
    assert decision.decision_code == "selected_delegate"
    assert decision.reasons == [
        "selected delegate execution because task is parallelizable and runtime supports native_delegation"
    ]


def test_delegate_signal_falls_back_to_direct_when_runtime_lacks_native_delegation() -> None:
    decision = select_execution_mode(
        TaskProfile(parallelizable=True),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=False, native_kanban=True),
        policy(),
    )

    assert decision.mode == "direct"
    assert decision.decision_code == "selected_direct"


def test_delegate_signal_falls_back_to_direct_when_policy_disables_delegate() -> None:
    decision = select_execution_mode(
        TaskProfile(parallelizable=True),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        ExecutionModePolicy(policy_version="execution-mode-v1", allow_delegate=False),
    )

    assert decision.mode == "direct"
    assert decision.decision_code == "selected_direct"


def test_persistent_execution_uses_kanban_when_supported() -> None:
    decision = select_execution_mode(
        TaskProfile(persistent_execution=True),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        policy(),
    )

    assert decision.mode == "kanban"
    assert decision.decision_code == "selected_kanban"
    assert decision.reasons == [
        "selected kanban execution because persistent_execution=True and runtime supports native_kanban"
    ]


def test_long_running_execution_uses_kanban_when_supported() -> None:
    decision = select_execution_mode(
        TaskProfile(long_running=True),
        selection("hermes"),
        RuntimeCapabilities(native_kanban=True),
        policy(),
    )

    assert decision.mode == "kanban"
    assert decision.decision_code == "selected_kanban"


def test_cross_role_dependencies_use_kanban_when_supported() -> None:
    decision = select_execution_mode(
        TaskProfile(cross_role_dependencies=True),
        selection("hermes"),
        RuntimeCapabilities(native_kanban=True),
        policy(),
    )

    assert decision.mode == "kanban"
    assert decision.decision_code == "selected_kanban"


def test_high_execution_and_parallelizable_use_kanban_when_supported() -> None:
    decision = select_execution_mode(
        TaskProfile(execution_complexity="high", parallelizable=True),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        policy(),
    )

    assert decision.mode == "kanban"
    assert decision.decision_code == "selected_kanban"


def test_kanban_signal_falls_back_without_native_kanban() -> None:
    decision = select_execution_mode(
        TaskProfile(persistent_execution=True, parallelizable=True),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=False),
        policy(),
    )

    assert decision.mode == "delegate"
    assert decision.decision_code == "selected_delegate"


def test_kanban_signal_falls_back_when_policy_disables_kanban() -> None:
    decision = select_execution_mode(
        TaskProfile(persistent_execution=True, parallelizable=True),
        selection("hermes"),
        RuntimeCapabilities(native_delegation=True, native_kanban=True),
        ExecutionModePolicy(policy_version="execution-mode-v1", allow_kanban=False),
    )

    assert decision.mode == "delegate"
    assert decision.decision_code == "selected_delegate"


def test_execution_mode_policy_is_strict_and_versioned() -> None:
    from pydantic import ValidationError

    ExecutionModePolicy(policy_version="execution-mode-v1")
    with pytest.raises(ValidationError):
        ExecutionModePolicy(policy_version="")
    with pytest.raises(ValidationError):
        ExecutionModePolicy(policy_version="   ")
    with pytest.raises(ValidationError):
        ExecutionModePolicy(policy_version="execution-mode-v1", unexpected=True)

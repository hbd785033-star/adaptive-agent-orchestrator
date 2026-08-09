"""Public orchestrator API with lazy imports to avoid package import cycles."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Orchestrator",
    "StateMachine",
    "TaskStatus",
    "RuleRouter",
    "BudgetConfig",
    "BudgetState",
    "ApprovalGate",
    "TaskProfiler",
    "WorkspaceManager",
]

_EXPORTS = {
    "Orchestrator": ("orchestrator.engine", "Orchestrator"),
    "StateMachine": ("orchestrator.state_machine", "StateMachine"),
    "TaskStatus": ("orchestrator.state_machine", "TaskStatus"),
    "RuleRouter": ("orchestrator.router", "RuleRouter"),
    "BudgetConfig": ("orchestrator.budget", "BudgetConfig"),
    "BudgetState": ("orchestrator.budget", "BudgetState"),
    "ApprovalGate": ("orchestrator.budget", "ApprovalGate"),
    "TaskProfiler": ("orchestrator.profiler", "TaskProfiler"),
    "WorkspaceManager": ("orchestrator.workspace", "WorkspaceManager"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

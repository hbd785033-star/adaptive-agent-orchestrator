from __future__ import annotations

import pytest

from adapters.mock import MockHermesAdapter
from orchestrator.runtime_registry import RuntimeRegistry


class PassiveRuntime(MockHermesAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def connect(self) -> None:
        self.calls.append("connect")

    async def submit(self, task):  # noqa: ANN001
        self.calls.append("submit")
        return await super().submit(task)


def test_single_runtime_registers_and_resolves_exact_object() -> None:
    hermes = PassiveRuntime()
    registry = RuntimeRegistry(entries=[("hermes", hermes)])

    assert registry.identities() == ("hermes",)
    assert registry.resolve("hermes") is hermes
    assert registry.items() == (("hermes", hermes),)
    assert hermes.calls == []


def test_multiple_runtime_order_is_deterministic_without_selection() -> None:
    hermes = PassiveRuntime()
    runtime_b = PassiveRuntime()
    registry = RuntimeRegistry(
        entries=[("hermes", hermes), ("runtime_b", runtime_b)]
    )

    assert registry.identities() == ("hermes", "runtime_b")
    assert registry.items() == (("hermes", hermes), ("runtime_b", runtime_b))
    assert registry.resolve("runtime_b") is runtime_b
    assert not hasattr(registry, "select")
    assert not hasattr(registry, "health")
    assert not hasattr(registry, "policy")
    assert hermes.calls == []
    assert runtime_b.calls == []


@pytest.mark.parametrize("identity", ["", "   "])
def test_blank_runtime_identity_fails_closed(identity: str) -> None:
    with pytest.raises(ValueError, match="identity must not be blank"):
        RuntimeRegistry(entries=[(identity, PassiveRuntime())])


def test_duplicate_runtime_identity_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate runtime identity: hermes"):
        RuntimeRegistry(
            entries=[
                ("hermes", PassiveRuntime()),
                ("hermes", PassiveRuntime()),
            ]
        )


def test_unknown_runtime_identity_fails_closed_without_side_effects() -> None:
    hermes = PassiveRuntime()
    registry = RuntimeRegistry(entries=[("hermes", hermes)])

    with pytest.raises(KeyError, match="unknown runtime identity: runtime_b"):
        registry.resolve("runtime_b")

    assert hermes.calls == []

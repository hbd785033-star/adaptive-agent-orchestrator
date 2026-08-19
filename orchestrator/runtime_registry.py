"""Immutable identity-to-runtime resolution without selection semantics."""
from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from adapters.runtime import AgentRuntime


class RuntimeRegistry:
    """Validate and resolve registered runtime instances by explicit identity."""

    def __init__(self, *, entries: Iterable[tuple[str, AgentRuntime]]) -> None:
        ordered: list[tuple[str, AgentRuntime]] = []
        by_identity: dict[str, AgentRuntime] = {}
        for identity, runtime in entries:
            if not identity.strip():
                raise ValueError("runtime identity must not be blank")
            if identity in by_identity:
                raise ValueError(f"duplicate runtime identity: {identity}")
            ordered.append((identity, runtime))
            by_identity[identity] = runtime
        self._items = tuple(ordered)
        self._by_identity = MappingProxyType(by_identity)

    def identities(self) -> tuple[str, ...]:
        return tuple(identity for identity, _ in self._items)

    def items(self) -> tuple[tuple[str, AgentRuntime], ...]:
        return self._items

    def resolve(self, identity: str) -> AgentRuntime:
        try:
            return self._by_identity[identity]
        except KeyError as exc:
            raise KeyError(f"unknown runtime identity: {identity}") from exc

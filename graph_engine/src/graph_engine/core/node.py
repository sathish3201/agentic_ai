"""Core Node interface.

Every node type in the engine must implement `execute` and `validate`.
`NodeMeta` enforces this at class-creation time so a malformed node type
fails fast at import/registration time rather than at graph-execution time.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any


class NodeMeta(ABCMeta):
    """Metaclass that enforces the Node interface on every subclass.

    Beyond ABCMeta's abstractmethod enforcement, this stamps a `node_type`
    class attribute (defaulting to the class name) so dynamically created
    node classes (see metaprogramming.factory) are addressable by name
    without requiring the author to set it manually.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if "node_type" not in namespace:
            cls.node_type = name
        return cls


class Node(metaclass=NodeMeta):
    """Base class for all executable graph nodes."""

    node_type: str = "Node"

    def __init__(self, node_id: str, **config: Any) -> None:
        self.node_id = node_id
        self.config = config

    @abstractmethod
    def execute(self, context: "ExecutionContext") -> Any:
        """Run this node's work, reading/writing shared state on context."""
        raise NotImplementedError

    def validate(self) -> None:
        """Override to raise on invalid configuration. No-op by default."""
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} node_id={self.node_id!r}>"


class ExecutionContext:
    """Shared mutable state passed between nodes during a graph run."""

    def __init__(self, initial_state: dict | None = None, user: Any = None) -> None:
        self.state: dict[str, Any] = dict(initial_state or {})
        self.user = user
        self.history: list[str] = []

    def record(self, node_id: str) -> None:
        self.history.append(node_id)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

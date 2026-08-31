"""Edge: a directed connection between two nodes, optionally conditional."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

EdgeCondition = Callable[["ExecutionContext"], bool]  # noqa: F821


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    condition: Optional[EdgeCondition] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_traversable(self, context: Any) -> bool:
        if self.condition is None:
            return True
        return bool(self.condition(context))

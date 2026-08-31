"""SubgraphNode: a Node whose execution delegates to an inner Graph.

Composite pattern — lets a workflow embed another workflow as a single
node, enabling nested/hierarchical graphs without special-casing anywhere
else in the engine (the Executor just calls `execute()` like any node).
"""
from __future__ import annotations

from typing import Any

from .graph import Graph
from .node import ExecutionContext, Node


class SubgraphNode(Node):
    node_type = "subgraph"

    def __init__(self, node_id: str, subgraph: Graph, **config: Any) -> None:
        super().__init__(node_id, **config)
        self.subgraph = subgraph

    def execute(self, context: ExecutionContext) -> Any:
        # Local import avoids a core.node <-> core.executor import cycle.
        from .executor import Executor

        Executor(self.subgraph).run(context)
        return context.state

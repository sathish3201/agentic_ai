"""Graph execution engine.

Day 1: minimal placeholder (topological, unconditional run) so
SubgraphNode and early tests are importable. Full conditional-edge
traversal, error handling/short-circuiting lands on Day 3.
"""
from __future__ import annotations

from .graph import Graph
from .node import ExecutionContext


class Executor:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def run(self, context: ExecutionContext) -> ExecutionContext:
        for node_id in self.graph.topological_order():
            node = self.graph.get_node(node_id)
            node.execute(context)
            context.record(node_id)
        return context

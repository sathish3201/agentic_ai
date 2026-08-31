"""Graph: adjacency-list graph of Nodes connected by Edges.

Optimized for O(1) node lookup and O(1) neighbor lookup, with a cached
topological order that is invalidated lazily (dirty-flag) on structural
mutation rather than recomputed on every access.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .edge import Edge
from .node import Node


class CycleError(ValueError):
    """Raised when a graph mutation would introduce a cycle where none is allowed."""


class Graph:
    def __init__(self, graph_id: str) -> None:
        self.graph_id = graph_id
        self._nodes: dict[str, Node] = {}
        self._adjacency: dict[str, list[Edge]] = {}
        self._topo_order: Optional[list[str]] = None
        self._dirty = True

    # -- node/edge mutation --------------------------------------------

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Node id already exists: {node.node_id}")
        node.validate()
        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, [])
        self._dirty = True

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"No such node: {node_id}")
        del self._nodes[node_id]
        self._adjacency.pop(node_id, None)
        for edges in self._adjacency.values():
            edges[:] = [e for e in edges if e.target != node_id]
        self._dirty = True

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self._nodes:
            raise KeyError(f"Unknown source node: {edge.source}")
        if edge.target not in self._nodes:
            raise KeyError(f"Unknown target node: {edge.target}")
        self._adjacency[edge.source].append(edge)
        self._dirty = True

    def remove_edge(self, source: str, target: str) -> None:
        edges = self._adjacency.get(source, [])
        before = len(edges)
        edges[:] = [e for e in edges if e.target != target]
        if len(edges) == before:
            raise KeyError(f"No edge {source} -> {target}")
        self._dirty = True

    # -- lookups ----------------------------------------------------------

    def get_node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def neighbors(self, node_id: str) -> list[Edge]:
        return list(self._adjacency.get(node_id, []))

    def nodes(self) -> Iterable[Node]:
        return list(self._nodes.values())

    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    # -- structural queries ------------------------------------------------

    def topological_order(self) -> list[str]:
        """Cached topological order; recomputed lazily after mutation."""
        if self._dirty or self._topo_order is None:
            self._topo_order = self._compute_topological_order()
            self._dirty = False
        return list(self._topo_order)

    def _compute_topological_order(self) -> list[str]:
        in_degree = {nid: 0 for nid in self._nodes}
        for edges in self._adjacency.values():
            for edge in edges:
                in_degree[edge.target] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for edge in self._adjacency.get(nid, []):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(order) != len(self._nodes):
            raise CycleError(f"Graph {self.graph_id!r} contains a cycle")
        return order

    def has_cycle(self) -> bool:
        try:
            self._compute_topological_order()
            return False
        except CycleError:
            return True

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"<Graph {self.graph_id!r} nodes={len(self._nodes)}>"

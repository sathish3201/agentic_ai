import pytest

from graph_engine import CycleError, Edge, ExecutionContext, Graph, SubgraphNode


def test_add_and_get_node(empty_graph, echo_node_cls):
    n = echo_node_cls("a")
    empty_graph.add_node(n)
    assert empty_graph.has_node("a")
    assert empty_graph.get_node("a") is n
    assert len(empty_graph) == 1


def test_add_duplicate_node_raises(empty_graph, echo_node_cls):
    empty_graph.add_node(echo_node_cls("a"))
    with pytest.raises(ValueError):
        empty_graph.add_node(echo_node_cls("a"))


def test_remove_node_also_removes_incoming_edges(empty_graph, echo_node_cls):
    empty_graph.add_node(echo_node_cls("a"))
    empty_graph.add_node(echo_node_cls("b"))
    empty_graph.add_edge(Edge("a", "b"))
    empty_graph.remove_node("b")
    assert empty_graph.neighbors("a") == []


def test_add_edge_unknown_node_raises(empty_graph, echo_node_cls):
    empty_graph.add_node(echo_node_cls("a"))
    with pytest.raises(KeyError):
        empty_graph.add_edge(Edge("a", "missing"))


def test_remove_edge_missing_raises(empty_graph, echo_node_cls):
    empty_graph.add_node(echo_node_cls("a"))
    empty_graph.add_node(echo_node_cls("b"))
    with pytest.raises(KeyError):
        empty_graph.remove_edge("a", "b")


def test_topological_order_linear(empty_graph, echo_node_cls):
    for nid in ("a", "b", "c"):
        empty_graph.add_node(echo_node_cls(nid))
    empty_graph.add_edge(Edge("a", "b"))
    empty_graph.add_edge(Edge("b", "c"))
    assert empty_graph.topological_order() == ["a", "b", "c"]


def test_topological_order_cached_and_invalidated(empty_graph, echo_node_cls):
    empty_graph.add_node(echo_node_cls("a"))
    empty_graph.add_node(echo_node_cls("b"))
    empty_graph.add_edge(Edge("a", "b"))
    order1 = empty_graph.topological_order()
    empty_graph.add_node(echo_node_cls("c"))
    empty_graph.add_edge(Edge("b", "c"))
    order2 = empty_graph.topological_order()
    assert order1 == ["a", "b"]
    assert order2 == ["a", "b", "c"]


def test_cycle_detection(empty_graph, echo_node_cls):
    empty_graph.add_node(echo_node_cls("a"))
    empty_graph.add_node(echo_node_cls("b"))
    empty_graph.add_edge(Edge("a", "b"))
    empty_graph.add_edge(Edge("b", "a"))
    assert empty_graph.has_cycle() is True
    with pytest.raises(CycleError):
        empty_graph.topological_order()


def test_subgraph_node_execution(echo_node_cls, context):
    inner = Graph("inner")
    inner.add_node(echo_node_cls("x"))
    inner.add_node(echo_node_cls("y"))
    inner.add_edge(Edge("x", "y"))

    outer = Graph("outer")
    outer.add_node(SubgraphNode("sub", inner))

    from graph_engine import Executor

    Executor(outer).run(context)
    assert context.get("visited") == ["x", "y"]
    assert context.history == ["x", "y", "sub"]


def test_liskov_substitutable_custom_node(empty_graph, context):
    class DoubleNode(echo_node_cls_factory()):
        pass

    node = DoubleNode("n1")
    empty_graph.add_node(node)
    empty_graph.get_node("n1").execute(context)
    assert context.get("visited") == ["n1"]


def echo_node_cls_factory():
    from graph_engine import ExecutionContext, Node

    class EchoNode(Node):
        def execute(self, ctx: ExecutionContext):
            visited = ctx.get("visited", [])
            visited.append(self.node_id)
            ctx.set("visited", visited)

    return EchoNode

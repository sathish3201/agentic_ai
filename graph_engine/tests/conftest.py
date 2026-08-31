import pytest

from graph_engine import ExecutionContext, Graph, Node


class EchoNode(Node):
    """Minimal concrete Node for tests: appends its id to state['visited']."""

    def execute(self, context: ExecutionContext):
        visited = context.get("visited", [])
        visited.append(self.node_id)
        context.set("visited", visited)
        return self.node_id


@pytest.fixture
def echo_node_cls():
    return EchoNode


@pytest.fixture
def empty_graph():
    return Graph("g1")


@pytest.fixture
def context():
    return ExecutionContext()

# graph_engine

An optimized, metaprogramming-driven graph workflow engine: nodes connected by edges, nodes that can nest whole subgraphs, dynamic workflow CRUD, and role-based auth gating — built as a standalone Python library, SOLID from the ground up.

Status: **Day 1 of a 7-day build.** See [PROGRESS.md](PROGRESS.md) for what's built each day and [docs/DESIGNING_WORKFLOWS.md](docs/DESIGNING_WORKFLOWS.md) for how to build a workflow with what exists today.

## Install (editable, for development)

```bash
cd graph_engine
pip install -e .
pip install -e ".[dev]"        # + pytest
pip install -e ".[langgraph]"  # + langgraph integration deps (Day 6+)
```

## Quick example

```python
from graph_engine import Graph, Edge, Node, ExecutionContext, Executor

class HelloNode(Node):
    def execute(self, context: ExecutionContext):
        context.set("greeting", f"hello, {self.config.get('name', 'world')}")

graph = Graph("demo")
graph.add_node(HelloNode("hello", name="graph_engine"))

context = ExecutionContext()
Executor(graph).run(context)
print(context.state["greeting"])  # "hello, graph_engine"
```

## Run tests

```bash
pytest tests/ -v
```

## Scope note

This library is the **substrate** — the engine and its dynamic node-spec format. A future agentic layer that turns natural-language input into a generated workflow spec is explicitly out of scope for this week and will be built as a separate consumer on top of this library.

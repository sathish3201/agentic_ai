# Designing a Workflow with graph_engine

This is a practical guide to building a workflow with the engine. It grows day by day alongside [PROGRESS.md](../PROGRESS.md) — sections marked **(coming Day N)** aren't implemented yet.

## Mental model

A workflow is a `Graph`: a set of `Node`s connected by `Edge`s.

- **Node** — one unit of work. You subclass `Node` and implement `execute(context)`.
- **Edge** — a directed link from one node to another. Can carry a `condition` function so the edge is only followed when that condition is true (branching).
- **ExecutionContext** — a shared bag of state (`context.state`) that flows through every node in a run, plus a `history` list of node ids that ran.
- **SubgraphNode** — a node whose body is itself a whole `Graph`. Use this to nest one workflow inside another instead of flattening everything into one giant graph.

## Step 1 — Define your nodes

Subclass `Node` and implement `execute`. Read/write shared state through `context.get()` / `context.set()` rather than instance attributes, so nodes stay stateless and reusable across runs.

```python
from graph_engine import Node, ExecutionContext

class FetchDataNode(Node):
    def execute(self, context: ExecutionContext):
        data = fetch_from_somewhere(self.config.get("source"))
        context.set("raw_data", data)

class SummarizeNode(Node):
    def execute(self, context: ExecutionContext):
        raw = context.get("raw_data")
        context.set("summary", summarize(raw))
```

`self.config` holds whatever keyword args you passed to the constructor — use it for per-instance configuration (API endpoints, thresholds, prompt templates, etc.) instead of hardcoding values in the class.

Override `validate()` if a node needs to reject bad configuration early (at `graph.add_node()` time, not at execution time):

```python
class FetchDataNode(Node):
    def validate(self):
        if "source" not in self.config:
            raise ValueError("FetchDataNode requires a 'source' config key")
```

## Step 2 — Wire nodes into a graph

```python
from graph_engine import Graph, Edge

graph = Graph("my_workflow")
graph.add_node(FetchDataNode("fetch", source="https://example.com/data"))
graph.add_node(SummarizeNode("summarize"))
graph.add_edge(Edge("fetch", "summarize"))
```

Rules enforced by `Graph`:
- Node ids must be unique within a graph (`add_node` raises `ValueError` on duplicates).
- An edge's `source`/`target` must already exist as nodes (`add_edge` raises `KeyError` otherwise).
- The graph must stay acyclic — `graph.topological_order()` raises `CycleError` if a cycle exists. Check `graph.has_cycle()` up front if you want to validate without triggering the exception path.

## Step 3 — Branch with conditional edges

An `Edge.condition` is a callable `(context) -> bool`. Use it for if/else-style branching:

```python
def is_long_summary(context):
    return len(context.get("summary", "")) > 500

graph.add_edge(Edge("summarize", "expand_details", condition=is_long_summary))
graph.add_edge(Edge("summarize", "finish", condition=lambda ctx: not is_long_summary(ctx)))
```

*(Full conditional traversal semantics — how the executor evaluates and skips non-traversable edges — land Day 3. The `Edge` shape itself is final as of Day 1.)*

## Step 4 — Nest a workflow as a subgraph

If a chunk of logic is reusable or deserves its own testable unit, build it as its own `Graph` and drop it into a parent graph via `SubgraphNode`:

```python
from graph_engine import SubgraphNode

inner = Graph("data_pipeline")
inner.add_node(FetchDataNode("fetch"))
inner.add_node(SummarizeNode("summarize"))
inner.add_edge(Edge("fetch", "summarize"))

outer = Graph("main")
outer.add_node(SubgraphNode("pipeline", inner))
outer.add_node(ReportNode("report"))
outer.add_edge(Edge("pipeline", "report"))
```

The inner graph shares the same `ExecutionContext` as the outer graph — so `context.state` set inside the subgraph is visible to nodes that run after it in the outer graph.

## Step 5 — Run it

```python
from graph_engine import Executor, ExecutionContext

context = ExecutionContext(initial_state={"user_id": 42})
Executor(graph).run(context)

print(context.state)     # final shared state
print(context.history)   # order nodes actually executed in
```

## Coming soon

- **(coming Day 2)** Define nodes as data instead of Python classes — register a node type once with `@node("fetch_data")`, then instantiate workflows from a plain JSON spec via a dynamic factory. This is the format a future agent layer would generate from natural-language input.
- **(coming Day 3)** Full executor: proper conditional-edge branching, failure propagation/short-circuiting, richer execution reporting.
- **(coming Day 4)** `Workspace` — create/read/update/delete named workflows at runtime, with pluggable persistence.
- **(coming Day 5)** `AuthNode` / `@requires_role(...)` — gate a node (or a whole graph) behind a user's role.
- **(coming Day 6)** Wire a real LangGraph agent in as a node via `integrations.langgraph_adapter`.

## Design guidelines

- Keep nodes small and single-purpose (Single Responsibility) — one node, one job.
- Never hardcode a downstream node's behavior inside an upstream node; connect them only through `Edge`s and shared `context.state` keys.
- Prefer composing a `SubgraphNode` over writing one sprawling flat graph once a workflow crosses roughly 6-8 nodes.
- Treat `context.state` keys as a contract between nodes — name them clearly (`raw_data`, not `x`) since multiple unrelated nodes may read/write them.

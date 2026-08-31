# Design: Inputs & Outputs, Layer by Layer

This is the contract surface of `graph_engine` — what goes in and what comes out at every layer, from a single node up to the whole library. Read this before writing code against a layer; it's the reference we align on before implementing each day.

Status tags: **[Day 1 — shipped]** contracts are implemented and tested today. **[Day N — planned]** contracts are locked in the design but not built yet; shape may still adjust slightly on that day if implementation surfaces an issue, but this doc is the intended target.

---

## 1. Node — the unit of work

**[Day 1 — shipped]**

| | Type | Description |
|---|---|---|
| **Input** | `node_id: str`, `**config: Any` (constructor) | Unique id within its graph, plus arbitrary keyword config (endpoints, thresholds, prompts, etc). |
| **Input** | `context: ExecutionContext` (to `execute()`) | Shared run state: `context.state` dict, `context.user`, `context.history`. |
| **Output** | `Any` (return of `execute()`) | Node-specific return value; not required to be used — most nodes communicate via `context.set(...)` instead. |
| **Output** | mutation of `context.state` | The real output channel between nodes — downstream nodes read keys an upstream node wrote. |
| **Side contract** | `validate() -> None` | Called once at `graph.add_node()` time; raises on bad config so errors surface at wiring time, not run time. |
| **Identity** | `node_type: str` | Class attribute, auto-stamped to the class name by `NodeMeta` unless overridden; used later (Day 2) as the registry key. |

## 2. Edge — a directed, optionally conditional link

**[Day 1 — shipped]**

| | Type | Description |
|---|---|---|
| **Input** | `source: str`, `target: str` | Node ids; both must already exist in the `Graph` when the edge is added. |
| **Input** | `condition: Callable[[ExecutionContext], bool] \| None` | Optional gate; edge is only traversable when this returns `True` (or is `None`). |
| **Input** | `metadata: dict` | Free-form annotation (e.g. a label for UI rendering) — not consumed by the engine itself. |
| **Output** | `is_traversable(context) -> bool` | Evaluates the condition against a given run's context. |

## 3. Graph — structure and lookups

**[Day 1 — shipped]**

| Operation | Input | Output |
|---|---|---|
| `add_node(node)` | `Node` instance | `None`; raises `ValueError` on duplicate id |
| `remove_node(node_id)` | `str` | `None`; also strips any edges targeting it; raises `KeyError` if unknown |
| `add_edge(edge)` | `Edge` instance | `None`; raises `KeyError` if source/target node missing |
| `remove_edge(source, target)` | `str, str` | `None`; raises `KeyError` if no such edge |
| `get_node(node_id)` | `str` | `Node` |
| `neighbors(node_id)` | `str` | `list[Edge]` outgoing from that node |
| `topological_order()` | — | `list[str]` node ids in dependency order; raises `CycleError` if cyclic. Cached, invalidated on any mutation. |
| `has_cycle()` | — | `bool`, non-raising check |

## 4. SubgraphNode — graph-as-node composition

**[Day 1 — shipped]**

| | Type | Description |
|---|---|---|
| **Input** | `node_id: str`, `subgraph: Graph` | Wraps an entire inner `Graph` as a single node in an outer graph. |
| **Input (runtime)** | `context: ExecutionContext` | The **same** context object is threaded into the inner graph's run — state set inside the subgraph is visible to nodes after it in the outer graph. |
| **Output** | `context.state` (via inner `Executor.run`) | Whatever the inner graph wrote to shared state; also appends every inner node id to `context.history` before its own id. |

## 5. Executor — running a graph

**[Day 1 — shipped, minimal] / [Day 3 — planned, full]**

| | Type | Description |
|---|---|---|
| **Input** | `graph: Graph` (constructor), `context: ExecutionContext` (to `run()`) | The graph to execute and the state to execute it against. |
| **Output** | `ExecutionContext` (same object, mutated) | `context.state` holds final shared state; `context.history` holds the ordered list of node ids that ran. |
| **[Day 3 planned]** | conditional traversal | Edges with a `condition` are only followed when true; a node with no traversable outgoing edge simply ends that branch. |
| **[Day 3 planned]** | failure semantics | A node raising an exception short-circuits its branch and surfaces a structured error (exact exception type TBD Day 3) rather than crashing the whole run silently. |

## 6. NodeRegistry + NodeFactory — metaprogramming layer

**[Day 2 — planned]**

| | Type | Description |
|---|---|---|
| **Input** | `@node("type_name")` applied to a `Node` subclass | Registers the class in a global/importable registry keyed by string name, at import time. |
| **Output** | registry lookup | `NodeRegistry.get("type_name") -> type[Node]` |
| **Input** | dynamic spec (dict/JSON) to `NodeFactory.from_spec(spec)` | See §9 below — this is the primary "workflow as data" input surface. |
| **Output** | `Node` instance (or a whole `Graph`, for a full-workflow spec) | Constructed via the registry, fully wired if the spec included edges. |
| **Constraint** | spec must be JSON-round-trippable | No lambdas/class references inside the spec — only strings, numbers, dicts, lists — because a future LLM-driven agent layer (out of scope this week) is expected to generate this spec as structured output. |

## 7. Workspace — dynamic workflow CRUD

**[Day 4 — planned]**

| Operation | Input | Output |
|---|---|---|
| `create_workflow(name, spec)` | `str`, dict spec | new `Graph`, stored under `name` |
| `get_workflow(name)` | `str` | `Graph` |
| `update_workflow(name, spec_patch)` | `str`, partial spec | mutated `Graph` (update-in-place, not full rebuild) |
| `delete_workflow(name)` | `str` | `None` |
| `list_workflows()` | — | `list[str]` names |
| **Persistence** | `WorkspaceStore` interface | in-memory implementation Day 4; swappable for a future DB-backed store without changing `Workspace` itself (Dependency Inversion). |

## 8. Auth — role-gated execution

**[Day 5 — planned]**

| | Type | Description |
|---|---|---|
| **Input** | `User`, `Role`, `Permission` models | Simple in-memory RBAC: role → allowed node-types/actions. |
| **Input** | `context.user` (set by caller before `Executor.run()`) | The acting user for this run; `AuthNode`/`@requires_role` reads this. |
| **Output** | pass-through (no state change) on authorized | Execution continues normally. |
| **Output** | raised exception on unauthorized | Execution of that node (and its branch) is blocked; exact exception type defined Day 5. |

## 9. The dynamic workflow spec — the library's primary external input format

**[Day 2 — planned, format locked now]**

This is the single most important input contract, since it's what a human, a config file, or a future LLM agent would author to *describe* a workflow without writing Python:

```json
{
  "graph_id": "example_workflow",
  "nodes": [
    { "id": "fetch", "type": "fetch_data", "config": { "source": "https://..." } },
    { "id": "summarize", "type": "summarize", "config": {} }
  ],
  "edges": [
    { "source": "fetch", "target": "summarize", "condition": null }
  ]
}
```

- **Input**: the JSON object above.
- **Output**: a fully constructed, wired `Graph`, ready to hand to `Executor`.
- Node `"type"` values are looked up in the `NodeRegistry` (§6) — meaning any node type used in a spec must already be registered via `@node(...)` somewhere imported by the process building the spec.
- `"condition"` is a *name*, not code (e.g. `"is_long_summary"`) — resolved against a small named-condition registry, not `eval`'d, to keep specs safe to accept from an untrusted or LLM-generated source.

## 10. Whole-library boundary — what a consumer imports and calls

**[Cumulative across all days]**

| Consumer need | Entry point |
|---|---|
| Build a workflow in Python | `Graph`, `Node`, `Edge`, `SubgraphNode` (Day 1) |
| Run a workflow | `Executor(graph).run(context)` (Day 1/3) |
| Register a reusable node type | `@node("name")` decorator (Day 2) |
| Build a workflow from data (JSON/dict) | `NodeFactory.from_spec(spec)` (Day 2) |
| Manage multiple named, mutable workflows at runtime | `Workspace` (Day 4) |
| Gate execution by user role | `AuthNode`, `@requires_role(...)` (Day 5) |
| Run a LangGraph-based agent as a node | `integrations.langgraph_adapter` (Day 6) |

Everything in `graph_engine.core`, `.metaprogramming`, `.workspace`, and `.auth` has **zero import-time dependency on LangGraph** — only `graph_engine.integrations` does. A consumer who never touches LangGraph never needs it installed.

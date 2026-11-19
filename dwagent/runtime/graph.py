"""A small stateful agent runtime: a graph of nodes over a JSON state, with conditional routing,
a checkpoint after every node, resume after interruption, and run-level safety limits.

    node(state, env) -> dict        partial update merged into the state
    edge                            fixed next node, or a router: state -> node name
    Interrupt                       pause for a human (approval); resume re-enters the same node
    exception                       the run fails at that node; resume retries it

State is plain JSON so a checkpoint is a row in SQLite and a run survives a process restart.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..store import Database, dumps
from ..tools.registry import ToolContext

END = "__end__"
State = dict[str, Any]
NodeFn = Callable[[State, "Env"], dict[str, Any] | None]


class Interrupt(Exception):
    """Pause the run. ``payload`` says what is being waited for."""
    is_pause = True

    def __init__(self, reason: str, payload: dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason, self.payload = reason, payload or {}


@dataclass
class Env:
    ctx: ToolContext
    services: dict[str, Any] = field(default_factory=dict)   # llm, registry, ...


@dataclass
class Limits:
    max_steps: int = 40            # nodes executed in one run, guards against routing loops
    wall_clock_s: float = 900.0
    token_budget: int = 60000
    max_node_attempts: int = 2     # automatic retries of a node that raises


@dataclass
class RunResult:
    run_id: str
    status: str                    # completed | interrupted | failed | limit
    state: State
    next_node: str
    detail: str = ""


class Checkpointer:
    def __init__(self, db: Database):
        self.db = db

    def save(self, run_id: str, next_node: str, state: State) -> int:
        row = self.db.one("SELECT COALESCE(MAX(seq), 0) AS s FROM checkpoints WHERE run_id=?", (run_id,))
        seq = row["s"] + 1
        self.db.execute("INSERT INTO checkpoints(run_id,seq,node,state,created) VALUES (?,?,?,?,?)",
                        (run_id, seq, next_node, dumps(state), time.time()))
        return seq

    def latest(self, run_id: str) -> tuple[str, State] | None:
        row = self.db.one("SELECT node,state FROM checkpoints WHERE run_id=? ORDER BY seq DESC LIMIT 1", (run_id,))
        return (row["node"], json.loads(row["state"])) if row else None

    def history(self, run_id: str) -> list[dict[str, Any]]:
        return self.db.query("SELECT seq,node,created FROM checkpoints WHERE run_id=? ORDER BY seq", (run_id,))


class Graph:
    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str | Callable[[State], str]] = {}
        self.targets: dict[str, set[str]] = {}
        self.entry = ""

    def node(self, name: str, fn: NodeFn) -> "Graph":
        self.nodes[name] = fn
        if not self.entry:
            self.entry = name
        return self

    def edge(self, src: str, dst: str) -> "Graph":
        self.edges[src], self.targets[src] = dst, {dst}
        return self

    def route(self, src: str, router: Callable[[State], str], targets: set[str]) -> "Graph":
        self.edges[src], self.targets[src] = router, set(targets)
        return self

    def validate(self) -> None:
        for src, dsts in self.targets.items():
            for d in dsts:
                if d != END and d not in self.nodes:
                    raise ValueError(f"{self.name}: edge {src} -> {d} points at an unknown node")
        for n in self.nodes:
            if n not in self.edges:
                raise ValueError(f"{self.name}: node {n} has no outgoing edge")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "entry": self.entry,
                "nodes": list(self.nodes), "edges": {s: sorted(t) for s, t in self.targets.items()}}

    def run(self, env: Env, state: State | None = None, resume: bool = False, limits: Limits | None = None) -> RunResult:
        self.validate()
        limits = limits or Limits()
        cp = Checkpointer(env.ctx.db)
        run_id, tracer = env.ctx.run_id, env.ctx.tracer
        if resume:
            saved = cp.latest(run_id)
            if saved is None:
                raise ValueError(f"no checkpoint for run {run_id}")
            node, state = saved
        else:
            node, state = self.entry, dict(state or {})
            state.setdefault("history", [])
            state.setdefault("usage", {"tokens": 0, "steps": 0})
            cp.save(run_id, node, state)
        assert state is not None
        t0 = time.perf_counter()
        steps = 0
        while node != END:
            if steps >= limits.max_steps:
                return RunResult(run_id, "limit", state, node, "max_steps")
            if time.perf_counter() - t0 > limits.wall_clock_s:
                return RunResult(run_id, "limit", state, node, "wall_clock")
            if state["usage"]["tokens"] >= limits.token_budget:
                return RunResult(run_id, "limit", state, node, "token_budget")
            attempt, update = 0, None
            while True:
                attempt += 1
                t_node = time.perf_counter()
                try:
                    with tracer.span(run_id, "node", f"{self.name}.{node}", input={"attempt": attempt}) as sp:
                        update = self.nodes[node](state, env) or {}
                        sp.output = {"keys": sorted(update)}
                    break
                except Interrupt as wait:
                    state["waiting"] = {"node": node, "reason": wait.reason, **wait.payload}
                    cp.save(run_id, node, state)
                    return RunResult(run_id, "interrupted", state, node, wait.reason)
                except Exception as exc:
                    state["history"].append({"node": node, "status": "error", "attempt": attempt,
                                             "error": f"{type(exc).__name__}: {exc}"[:300]})
                    if attempt >= limits.max_node_attempts:
                        cp.save(run_id, node, state)
                        return RunResult(run_id, "failed", state, node, f"{type(exc).__name__}: {exc}")
            state.pop("waiting", None)
            state.update(update)
            state["history"].append({"node": node, "status": "ok", "attempt": attempt,
                                     "ms": round((time.perf_counter() - t_node) * 1000, 1)})
            steps += 1
            state["usage"]["steps"] = state["usage"].get("steps", 0) + 1
            nxt = self.edges[node]
            node = nxt(state) if callable(nxt) else nxt
            if node != END and node not in self.nodes:
                return RunResult(run_id, "failed", state, node, f"router returned unknown node {node!r}")
            cp.save(run_id, node, state)      # durable before the next node starts
        return RunResult(run_id, "completed", state, END)

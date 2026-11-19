"""Execution traces: one row per run, one span per model call, tool call, graph node or guard.

Each span records latency, token usage, attempts, the error kind and a size-capped copy of its
input and output. ``redacted_events`` is what a company employee sees: timings and error kinds
without any payload, because payloads can contain patient content.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..store import Database, dumps, get_db

MAX_PAYLOAD = 4000


@dataclass
class Span:
    kind: str
    name: str
    input: Any = None
    output: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True
    error_kind: str | None = None
    error: str | None = None
    attempts: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def fail(self, kind: str, message: str) -> None:
        self.ok, self.error_kind, self.error = False, kind, message[:500]


class Tracer:
    def __init__(self, db: Database | None = None, on_event: Callable[[dict[str, Any]], None] | None = None):
        self.db = db or get_db()
        self.on_event = on_event

    def start_run(self, agent: str, task: str, user_id: str = "system", role: str = "system",
                  meta: dict[str, Any] | None = None, run_id: str | None = None) -> str:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        if self.db.one("SELECT run_id FROM runs WHERE run_id=?", (run_id,)):
            self.db.execute("UPDATE runs SET status='running', ended=NULL WHERE run_id=?", (run_id,))
        else:
            self.db.execute("INSERT INTO runs(run_id,agent,task,user_id,role,status,started,meta) VALUES (?,?,?,?,?,?,?,?)",
                            (run_id, agent, task[:2000], user_id, role, "running", time.time(), dumps(meta or {})))
        return run_id

    def end_run(self, run_id: str, status: str, answer: str = "") -> None:
        self.db.execute("UPDATE runs SET status=?, ended=?, answer=? WHERE run_id=?",
                        (status, time.time(), answer[:8000], run_id))

    @contextmanager
    def span(self, run_id: str, kind: str, name: str, input: Any = None) -> Iterator[Span]:
        sp = Span(kind=kind, name=name, input=input)
        t0 = time.perf_counter()
        started = time.time()
        try:
            yield sp
        except Exception as exc:
            if getattr(exc, "is_pause", False):      # waiting for a human is not a failure
                sp.output = {"paused": str(exc)}
            elif sp.ok:
                sp.fail(getattr(exc, "kind", type(exc).__name__), str(exc))
            raise
        finally:
            latency = (time.perf_counter() - t0) * 1000
            sid = self.db.execute(
                "INSERT INTO spans(run_id,kind,name,started,latency_ms,prompt_tokens,completion_tokens,ok,"
                "error_kind,error,attempts,input,output) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, kind, name, started, latency, sp.prompt_tokens, sp.completion_tokens, int(sp.ok),
                 sp.error_kind, sp.error, sp.attempts, dumps(sp.input)[:MAX_PAYLOAD], dumps(sp.output)[:MAX_PAYLOAD]))
            if self.on_event:
                self.on_event({"id": sid, "run_id": run_id, "kind": kind, "name": name, "ok": sp.ok,
                               "latency_ms": round(latency, 1), "error_kind": sp.error_kind})

    def events(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM spans WHERE run_id=? AND id>? ORDER BY id", (run_id, after_id))

    def redacted_events(self, run_id: str) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT id,kind,name,started,latency_ms,prompt_tokens,completion_tokens,ok,error_kind,attempts "
            "FROM spans WHERE run_id=? ORDER BY id", (run_id,))

    def summary(self, run_id: str) -> dict[str, Any]:
        rows = self.events(run_id)
        llm = [r for r in rows if r["kind"] == "llm"]
        tools = [r for r in rows if r["kind"] == "tool"]
        run = self.db.one("SELECT * FROM runs WHERE run_id=?", (run_id,)) or {}
        wall = ((run.get("ended") or time.time()) - run["started"]) * 1000 if run.get("started") else 0.0
        return {
            "run_id": run_id, "status": run.get("status"), "wall_ms": round(wall, 1),
            "llm_calls": len(llm), "tool_calls": len(tools),
            "prompt_tokens": sum(r["prompt_tokens"] for r in llm),
            "completion_tokens": sum(r["completion_tokens"] for r in llm),
            "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in llm),
            "llm_latency_ms": round(sum(r["latency_ms"] for r in llm), 1),
            "tool_latency_ms": round(sum(r["latency_ms"] for r in tools), 1),
            "tool_errors": sum(1 for r in tools if not r["ok"]),
            "tool_retries": sum(max(0, r["attempts"] - 1) for r in tools),
            "errors_by_kind": _count(r["error_kind"] for r in rows if not r["ok"]),
        }


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i or "unknown"] = out.get(i or "unknown", 0) + 1
    return out

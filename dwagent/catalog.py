"""The full registry: atomic tools plus skills."""
from __future__ import annotations

from functools import lru_cache

from .memory.stores import memory_tool
from .runtime.trace import Tracer
from .skills.nodule_assessment import make_assess_nodules
from .store import Database, get_db
from .tools.builtin import build_registry
from .tools.registry import Principal, ToolContext, ToolRegistry


@lru_cache(maxsize=1)
def get_registry() -> ToolRegistry:
    r = build_registry()
    r.register(make_assess_nodules(r))
    r.register(memory_tool())
    return r


def new_context(principal: Principal, run_id: str, db: Database | None = None, tracer: Tracer | None = None) -> ToolContext:
    db = db or get_db()
    return ToolContext(principal=principal, db=db, tracer=tracer or Tracer(db), run_id=run_id)

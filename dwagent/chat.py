"""One conversational turn for a signed-in user: memory in, agent loop, memory out."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .agent.loop import AgentState, run_agent
from .catalog import get_registry, new_context
from .llm.base import LLMClient
from .llm.ollama import get_llm
from .memory.stores import LongTermMemory, SessionMemory
from .profiles import AGENTS
from .runtime.trace import Tracer
from .store import Database, get_db
from .tools.registry import Principal


def chat_turn(principal: Principal, session_id: str, message: str, llm: LLMClient | None = None,
              db: Database | None = None, run_id: str | None = None) -> dict[str, Any]:
    db = db or get_db()
    tracer = Tracer(db)
    cfg = AGENTS[principal.role]
    cfg = replace(cfg, tool_names=[*(cfg.tool_names or []), "remember"])
    run_id = tracer.start_run(cfg.name, message, principal.user_id, principal.role,
                              {"session_id": session_id}, run_id=run_id)
    ctx = new_context(principal, run_id, db, tracer)
    session = SessionMemory(db, f"{principal.user_id}:{session_id}")
    memories = LongTermMemory(db).recall(principal.user_id, message)
    system = cfg.system_prompt + (
        "\nMemory: when the user asks you to remember a preference or lasting fact, call the remember tool. "
        "Never say you have remembered something unless that tool call succeeded.")
    if memories:
        system += "\n\nWhat this user asked you to remember:\n" + "\n".join(f"- {m['text']}" for m in memories)
    messages = [{"role": "system", "content": system}, *session.history(), {"role": "user", "content": message}]
    state = AgentState(run_id=run_id, task=message, messages=messages)
    state = run_agent(llm or get_llm(), get_registry(), ctx, config=cfg, state=state)
    answer = state.answer or "I could not complete that request."
    session.append("user", message)
    session.append("assistant", answer)
    tracer.end_run(run_id, state.status, answer)
    return {"run_id": run_id, "answer": answer, "status": state.status, "stop_reason": state.stop_reason,
            "iterations": state.iterations, "tool_log": state.tool_log, "memories_used": [m["text"] for m in memories],
            "summary": tracer.summary(run_id)}

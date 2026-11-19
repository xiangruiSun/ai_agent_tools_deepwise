"""Short-term and long-term memory.

Short-term (SessionMemory): the turns of one conversation, persisted so a session survives a
restart; only the last few turns are replayed to the model.
Long-term (LongTermMemory): durable facts about one user (preferences, standing context),
written through the ``remember`` tool, retrieved by BM25 and injected into the system prompt.
Both are keyed by the authenticated user id, so one user's memory can never be read by another.
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..rag.bm25 import BM25
from ..store import Database
from ..tools.registry import Tool, ToolContext, ToolError


class SessionMemory:
    def __init__(self, db: Database, session_id: str):
        self.db, self.session_id = db, session_id

    def append(self, role: str, content: str) -> None:
        row = self.db.one("SELECT COALESCE(MAX(seq),0) AS s FROM sessions WHERE session_id=?", (self.session_id,))
        self.db.execute("INSERT INTO sessions(session_id,seq,role,content,created) VALUES (?,?,?,?,?)",
                        (self.session_id, row["s"] + 1, role, content, time.time()))

    def history(self, max_turns: int = 6) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT role,content FROM sessions WHERE session_id=? ORDER BY seq DESC LIMIT ?",
                             (self.session_id, max_turns * 2))
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


class LongTermMemory:
    MAX_PER_USER = 50

    def __init__(self, db: Database):
        self.db = db

    def remember(self, user_id: str, text: str, kind: str = "preference", source: str = "") -> int:
        text = text.strip()
        dup = self.db.one("SELECT id FROM memory WHERE user_id=? AND text=?", (user_id, text))
        if dup:
            return dup["id"]
        n = self.db.one("SELECT count(*) AS n FROM memory WHERE user_id=?", (user_id,))["n"]
        if n >= self.MAX_PER_USER:   # bounded: drop the oldest
            self.db.execute("DELETE FROM memory WHERE id=(SELECT MIN(id) FROM memory WHERE user_id=?)", (user_id,))
        return self.db.execute("INSERT INTO memory(user_id,kind,text,source,created) VALUES (?,?,?,?,?)",
                               (user_id, kind, text, source, time.time()))

    def all(self, user_id: str) -> list[dict[str, Any]]:
        return self.db.query("SELECT id,kind,text,created FROM memory WHERE user_id=? ORDER BY id", (user_id,))

    def recall(self, user_id: str, query: str, k: int = 3) -> list[dict[str, Any]]:
        items = self.all(user_id)
        if not items:
            return []
        hits = BM25([i["text"] for i in items]).search(query, k)
        chosen = [items[i] for i, _ in hits]
        # Preferences always apply, whether or not the current message happens to mention them.
        for i in items:
            if i["kind"] == "preference" and i not in chosen and len(chosen) < k + 2:
                chosen.append(i)
        return chosen

    def forget(self, user_id: str, memory_id: int) -> bool:
        before = self.db.one("SELECT count(*) AS n FROM memory WHERE user_id=? AND id=?", (user_id, memory_id))["n"]
        self.db.execute("DELETE FROM memory WHERE user_id=? AND id=?", (user_id, memory_id))
        return bool(before)


class RememberArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=5, max_length=200, description="One durable fact or preference, as a full sentence.")


_SENSITIVE = ("password", "passcode", "id number", "social security", "credit card", "bank account")


def _remember(a: RememberArgs, ctx: ToolContext) -> dict[str, Any]:
    if any(s in a.text.lower() for s in _SENSITIVE):
        raise ToolError("invalid_arguments", "credentials and identity numbers are never stored in memory")
    mid = LongTermMemory(ctx.db).remember(ctx.principal.user_id, a.text, "preference", ctx.run_id)
    return {"stored": True, "memory_id": mid}


def memory_tool() -> Tool:
    return Tool("remember", "Store one lasting preference or fact the user asked you to remember for future "
                "conversations (for example a preferred language or report style). Not for results or diagnoses.",
                RememberArgs, _remember, side_effect=True, idempotent=False, max_retries=0)

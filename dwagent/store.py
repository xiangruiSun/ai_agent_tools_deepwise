"""SQLite persistence shared by the runtime: traces, checkpoints, approvals, idempotency keys,
memory, reports, users and the audit log. One file, WAL mode, safe for the web server's threads."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, agent TEXT, task TEXT, user_id TEXT, role TEXT, status TEXT,
    started REAL, ended REAL, answer TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, kind TEXT, name TEXT, started REAL,
    latency_ms REAL, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    ok INTEGER DEFAULT 1, error_kind TEXT, error TEXT, attempts INTEGER DEFAULT 1,
    input TEXT, output TEXT);
CREATE INDEX IF NOT EXISTS spans_run ON spans(run_id, id);
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT, seq INTEGER, node TEXT, state TEXT, created REAL, PRIMARY KEY (run_id, seq));
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, call_key TEXT, tool TEXT, arguments TEXT,
    requested_by TEXT, status TEXT, decided_by TEXT, created REAL, decided REAL,
    UNIQUE (run_id, call_key));
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY, tool TEXT, result TEXT, created REAL);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, kind TEXT, text TEXT, source TEXT, created REAL);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT, seq INTEGER, role TEXT, content TEXT, created REAL, PRIMARY KEY (session_id, seq));
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY, patient_id TEXT, study_uid TEXT, status TEXT, body TEXT,
    drafted_by TEXT, signed_by TEXT, created REAL, signed REAL, run_id TEXT);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, user_id TEXT, role TEXT, action TEXT,
    resource TEXT, allowed INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Database:
    def __init__(self, path: Path | str | None = None):
        self.path = str(path or get_settings().db_path)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._con.execute("PRAGMA journal_mode=WAL")
        self._con.executescript(SCHEMA)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._con.execute(sql, tuple(params))
            return cur.lastrowid or 0

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._con.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def audit(self, user_id: str, role: str, action: str, resource: str, allowed: bool, detail: str = "") -> None:
        self.execute("INSERT INTO audit(at,user_id,role,action,resource,allowed,detail) VALUES (?,?,?,?,?,?,?)",
                     (time.time(), user_id, role, action, resource, int(allowed), detail[:500]))

    def close(self) -> None:
        with self._lock:
            self._con.close()


_default: Database | None = None
_default_lock = threading.Lock()


def get_db() -> Database:
    global _default
    with _default_lock:
        if _default is None:
            _default = Database()
        return _default


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)

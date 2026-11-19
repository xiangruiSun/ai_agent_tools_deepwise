"""Shared context for multi-agent work: a blackboard with versioned entries.

Agents do not message each other directly; they read and write named entries here. Every
write carries the version the writer last saw (optimistic concurrency): if another agent
wrote in between, the write is rejected with Conflict and the writer must re-read and merge.
An append-only log records who wrote what, which is the coordination trace.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


class Conflict(Exception):
    pass


@dataclass
class Entry:
    value: Any
    version: int
    author: str
    at: float


class Blackboard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, Entry] = {}
        self.log: list[dict[str, Any]] = []
        self.conflicts = 0

    def read(self, key: str) -> tuple[Any, int]:
        with self._lock:
            e = self._entries.get(key)
            return (e.value, e.version) if e else (None, 0)

    def write(self, key: str, value: Any, author: str, expected_version: int | None = None) -> int:
        with self._lock:
            current = self._entries.get(key)
            version = current.version if current else 0
            if expected_version is not None and expected_version != version:
                self.conflicts += 1
                self.log.append({"at": time.time(), "author": author, "key": key, "event": "conflict",
                                 "expected": expected_version, "actual": version})
                raise Conflict(f"{key}: expected version {expected_version}, found {version}")
            self._entries[key] = Entry(value, version + 1, author, time.time())
            self.log.append({"at": time.time(), "author": author, "key": key, "event": "write", "version": version + 1})
            return version + 1

    def append(self, key: str, item: Any, author: str, retries: int = 5) -> int:
        """Read-modify-write of a list entry, retried on conflict."""
        for _ in range(retries):
            value, version = self.read(key)
            try:
                return self.write(key, [*(value or []), item], author, expected_version=version)
            except Conflict:
                continue
        raise Conflict(f"{key}: could not append after {retries} attempts")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {k: e.value for k, e in self._entries.items()}

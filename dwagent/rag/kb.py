"""Guideline knowledge base: Markdown files in kb/, one chunk per '## ' section, BM25 search.

Every chunk has a stable id (file#section-slug). Answers must cite these ids, and the
validators check that every cited id exists and was actually retrieved in the run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..config import get_settings
from .bm25 import BM25


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    text: str
    source: str
    audience: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@lru_cache(maxsize=1)
def load_chunks() -> tuple[Chunk, ...]:
    chunks = []
    for path in sorted(get_settings().kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        source = (re.search(r"^source:\s*(.+)$", text, re.M) or [None, path.stem])[1]
        audience = (re.search(r"^audience:\s*(\w+)", text, re.M) or [None, "clinician"])[1]
        for block in re.split(r"^## ", text, flags=re.M)[1:]:
            title, _, body = block.partition("\n")
            chunks.append(Chunk(f"{path.stem}#{_slug(title)}", title.strip(), body.strip(), source, audience))
    return tuple(chunks)


@lru_cache(maxsize=1)
def _index() -> BM25:
    return BM25([f"{c.title}. {c.title}. {c.text}" for c in load_chunks()])


def search(query: str, k: int = 3, audience: str | None = None) -> list[dict]:
    chunks = load_chunks()
    hits = []
    for i, score in _index().search(query, k=len(chunks)):
        c = chunks[i]
        if audience and c.audience != audience:
            continue
        hits.append({"chunk_id": c.chunk_id, "title": c.title, "text": c.text, "source": c.source,
                     "score": round(score, 2)})
        if len(hits) == k:
            break
    return hits


def chunk_ids() -> set[str]:
    return {c.chunk_id for c in load_chunks()}


def get_chunk(chunk_id: str) -> Chunk | None:
    return next((c for c in load_chunks() if c.chunk_id == chunk_id), None)

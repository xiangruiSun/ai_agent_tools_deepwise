"""A small Okapi BM25 index. Enough for the guideline knowledge base and long-term memory."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = {"the", "a", "an", "of", "to", "and", "or", "is", "are", "in", "on", "for", "with", "at", "by", "it",
         "that", "this", "be", "as", "than", "what", "which", "do", "does", "i", "my", "me"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.n = len(docs)
        self.avgdl = (sum(len(t) for t in self.tokens) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for t in self.tokens:
            df.update(set(t))
        self.idf = {w: math.log(1 + (self.n - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.tokens[i]) or 1
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

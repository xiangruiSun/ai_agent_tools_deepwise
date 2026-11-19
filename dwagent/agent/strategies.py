"""Reasoning strategies for a single question, without tools, so their cost and accuracy can be
compared on the same items.

    direct       one call, answer only
    cot          one call, reason step by step and then answer (Chain of Thought)
    reflect      answer, critique the answer, revise (Self-Reflection): 3 calls
    tot          several independent reasoning paths, an evaluator scores them, the best is
                 kept (Tree of Thoughts, breadth k, depth 1): k + 1 calls

Every strategy returns the same shape so the evaluation script can score them alike.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from ..llm.base import LLMClient

ANSWER = re.compile(r"ANSWER\s*[:=]\s*\**\s*([A-Za-z0-9]+)", re.I)


@dataclass
class StrategyResult:
    strategy: str
    answer: str | None
    calls: int
    tokens: int
    latency_ms: float
    text: str


def _ask(llm: LLMClient, system: str, user: str, temperature: float = 0.0) -> tuple[str, int]:
    r = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=temperature)
    return r.content, r.usage.total


def _parse(text: str) -> str | None:
    m = ANSWER.findall(text)
    return m[-1].upper() if m else None


def run_strategy(llm: LLMClient, strategy: str, context: str, question: str, k: int = 3) -> StrategyResult:
    base = ("Use only the reference text to answer. " + context +
            "\nFinish with a line of the form 'ANSWER: <value>'.")
    t0 = time.perf_counter()
    calls = tokens = 0
    if strategy == "direct":
        text, n = _ask(llm, base + " Reply with the ANSWER line only, no explanation.", question)
        calls, tokens = 1, n
    elif strategy == "cot":
        text, n = _ask(llm, base + " Think step by step: identify the nodule type, the size that matters, the "
                                   "threshold band it falls in, then give the answer.", question)
        calls, tokens = 1, n
    elif strategy == "reflect":
        first, n1 = _ask(llm, base + " Give a brief justification, then the ANSWER line.", question)
        critique, n2 = _ask(llm, "You are checking another clinician's answer against the reference text. " + context +
                            "\nList any mistake in applying the thresholds. If there is none, say 'No mistake'.",
                            f"Question: {question}\n\nProposed answer:\n{first}")
        text, n3 = _ask(llm, base, f"{question}\n\nYour earlier answer:\n{first}\n\nReviewer comments:\n{critique}\n\n"
                                   "Give the final answer, correcting it if the reviewer found a mistake.")
        calls, tokens = 3, n1 + n2 + n3
    elif strategy == "tot":
        paths = []
        for i in range(k):
            p, n = _ask(llm, base + " Reason step by step in your own way, then give the ANSWER line.", question,
                        temperature=0.7 if i else 0.0)
            paths.append(p)
            tokens += n
        listing = "\n\n".join(f"Path {i + 1}:\n{p}" for i, p in enumerate(paths))
        verdict, n = _ask(llm, "You evaluate reasoning paths against the reference text. " + context +
                          "\nPick the path whose reasoning applies the thresholds correctly. Reply 'BEST: <number>'.",
                          f"Question: {question}\n\n{listing}")
        tokens += n
        calls = k + 1
        m = re.search(r"BEST\s*[:=]\s*(\d+)", verdict)
        best = min(max(int(m.group(1)), 1), k) - 1 if m else 0
        text = paths[best]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return StrategyResult(strategy, _parse(text), calls, tokens, (time.perf_counter() - t0) * 1000, text)


def strategy_names() -> list[str]:
    return ["direct", "cot", "reflect", "tot"]


def describe() -> dict[str, Any]:
    return {"direct": "1 call", "cot": "1 call, longer output", "reflect": "3 calls", "tot": "k + 1 calls"}

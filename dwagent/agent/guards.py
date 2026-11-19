"""Guards against agent failure modes that showed up in the first verification run on real
studies (kept in verification/t01_before_guards.json):

  * hallucinated citation   the answer cited "[chunk_id: 12345]", an id no tool returned
  * hallucinated result     after lung_rads_category failed, the answer still gave a category
  * announced, not done     the run ended on "I will now proceed with the corrected parameters"

``grounding_check`` runs when the model offers a final answer. It is deterministic: it compares
the answer with what the tools actually returned in this run. Problems are sent back to the
model as a numbered list and it gets a bounded number of repair turns (AgentConfig.max_answer_repairs).
"""
from __future__ import annotations

import re
from typing import Any

from ..tools.registry import ToolContext

_CATEGORY = re.compile(r"(?:lung-rads|category)\s*(?:category\s*)?(?:is|of|:|=)?\s*\**\s*(4A|4B|4X|[123])\b", re.I)
_INTENT = re.compile(r"\bI(?:'ll| will)\s+(?:now\s+)?(?:proceed|call|re-?calculate|retry|re-?run|use the|check|look)", re.I)
_CITE = re.compile(r"\[([^\[\]]{3,80})\]")
_RESULT_CATEGORY = re.compile(r'"(?:category|lung_rads|exam_lung_rads)":\s*"(4A|4B|4X|[123])"')


def grounding_check(state: Any, ctx: ToolContext) -> list[str]:
    answer = state.answer_candidate
    problems: list[str] = []
    ok_obs = [m["content"] for m in state.messages if m["role"] == "tool" and m["content"].startswith('{"ok": true')]
    # 1. Categories must have come from a tool in this run.
    returned = {c.upper() for o in ok_obs for c in _RESULT_CATEGORY.findall(o)}
    stated = {c.upper() for c in _CATEGORY.findall(answer)}
    for c in sorted(stated - returned):
        got = f"The tools returned these categories: {', '.join(sorted(returned))}." if returned else \
            "No tool has returned a category yet."
        problems.append(f"You stated Lung-RADS category {c}, but no tool returned that category in this conversation. "
                        f"{got} Use only those (the exam category is the most suspicious of them), or call "
                        "lung_rads_category or assess_nodules.")
    # 2. Citations must be ids that were actually retrieved.
    chunks = set(ctx.extras.get("retrieved_chunks", set()))
    cases = set(ctx.extras.get("retrieved_cases", set()))
    for o in ok_obs:
        chunks.update(re.findall(r'"([a-z0-9_]+#[a-z0-9\-]+)"', o))
        cases.update(re.findall(r'"(case_[0-9a-f]+)"', o))
    for raw in _CITE.findall(answer):
        for cite in [c.strip() for c in re.split(r"[,;]", raw)]:
            cite = re.sub(r"^(chunk_id|case_id)\s*[:=]\s*", "", cite, flags=re.I)
            if "#" in cite and cite not in chunks:
                problems.append(f"Citation [{cite}] is not a guideline chunk id returned by a tool. Cite only ids you retrieved.")
            elif cite.startswith("case_") and cite not in cases:
                problems.append(f"Citation [{cite}] is not a case id returned by a tool.")
            elif re.fullmatch(r"\d{3,}", cite):
                problems.append(f"Citation [{cite}] is not a real id. Cite chunk_id or case_id values exactly as returned.")
    # 3. An announcement is not an answer.
    if _INTENT.search(answer) and len(answer) < 700:
        problems.append("You said what you will do next but did not do it. Call the tool now, or give the final answer.")
    return problems[:5]

"""A small multi-agent team that writes a nodule assessment brief for one study.

Roles
    Planner     turns the request into research questions (structured output)
    Researcher  a tool-calling agent that answers one question and records its evidence
    Writer      composes the brief from the blackboard only (no tools)
    Critic      checks the brief against the evidence; deterministic checks first, then a
                model review; may send it back once

Orchestration patterns
    sequential    planner -> researcher (questions one by one) -> writer -> critic -> revise
    hierarchical  a supervisor agent delegates through tools (research / write / review)
    network       researchers work in parallel on a shared blackboard, then writer and critic
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..agent.loop import AgentConfig, run_agent, tool_messages
from ..ct import lungrads
from ..llm.base import LLMClient
from ..rag import kb
from ..structured.generate import generate_structured
from ..tools.registry import Tool, ToolContext, ToolRegistry
from .blackboard import Blackboard

RESEARCH_TOOLS = ["assess_nodules", "search_guideline", "search_similar_cases", "get_series_metadata", "calculator"]
BRIEF_SPEC = ("The brief must state: every confirmed finding with its side, zone, type and mean diameter; the exam "
              "Lung-RADS category; the management recommendation; at least one guideline chunk id in square brackets; "
              "and one or two similar published cases with their case ids in square brackets. Under 220 words.")


class Questions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[str] = Field(min_length=2, max_length=4)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool
    issues: list[str] = Field(max_length=6)


@dataclass
class TeamResult:
    pattern: str
    brief: str
    status: str
    tokens: int = 0
    llm_calls: int = 0
    wall_s: float = 0.0
    revisions: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)
    board_log: list[dict[str, Any]] = field(default_factory=list)
    conflicts: int = 0


class _Meter:
    """Wraps the model client to count calls and tokens across all agents of a team."""

    def __init__(self, llm: LLMClient):
        self.llm, self.model, self.calls, self.tokens = llm, llm.model, 0, 0

    def chat(self, *a: Any, **k: Any) -> Any:
        r = self.llm.chat(*a, **k)
        self.calls += 1
        self.tokens += r.usage.total
        return r


def collect_evidence(observations: list[dict[str, Any]]) -> dict[str, Any]:
    ev: dict[str, Any] = {"assessment": None, "chunk_ids": set(), "case_ids": set()}
    for o in observations:
        if not o.get("ok"):
            continue
        r = o.get("result") or {}
        if o["name"] == "assess_nodules":
            ev["assessment"] = r
            ev["chunk_ids"].update(r.get("guideline_chunks", []))
        elif o["name"] == "search_guideline":
            ev["chunk_ids"].update(c["chunk_id"] for c in r.get("chunks", []))
        elif o["name"] == "search_similar_cases":
            ev["case_ids"].update(c["case_id"] for c in r.get("cases", []))
    return ev


def score_brief(brief: str, patient_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Deterministic rubric (0-6) against ground truth computed without any model."""
    from ..clinical import review_findings
    from ..structured.rules import expected_category
    truth = review_findings(patient_id)["findings"]
    exam = lungrads.exam_category([expected_category(f)["category"] for f in truth])
    text = brief.lower()
    cited = set(re.findall(r"\[([^\[\]]+)\]", brief))
    cited = {c.strip() for part in cited for c in part.split(",")}
    chunk_cites = {c for c in cited if "#" in c}
    case_cites = {c for c in cited if c.startswith("case_")}
    interval = {"1": "12 months", "2": "12 months", "3": "6 months", "4A": "3 months", "4B": "diagnostic chest ct",
                "4X": "diagnostic chest ct"}[exam]
    sizes = {f["mean_diameter_mm"] for f in truth}
    mentioned = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mm", brief)}
    checks = {
        "exam_category_correct": bool(re.search(rf"lung-rads\s*(category\s*)?{exam.lower()}\b", text)),
        "management_correct": interval in text,
        "all_findings_sized": all(any(abs(s - m) < 0.05 for m in mentioned) for s in sizes),
        "guideline_cited_and_real": bool(chunk_cites) and chunk_cites <= kb.chunk_ids(),
        "cases_cited_and_retrieved": bool(case_cites) and case_cites <= set(evidence.get("case_ids", set())),
        "no_unsupported_numbers": all(any(abs(m - s) < 0.05 for s in sizes) or m in (2.5, 4.0, 6.0, 8.0, 15.0, 30.0, 1.5)
                                      for m in mentioned),
    }
    return {"score": sum(checks.values()), "max": 6, "checks": checks, "truth_exam": exam}


def deterministic_issues(brief: str, patient_id: str, evidence: dict[str, Any]) -> list[str]:
    s = score_brief(brief, patient_id, evidence)
    msg = {"exam_category_correct": f"state the exam category as 'Lung-RADS {s['truth_exam']}'",
           "management_correct": "give the management recommendation exactly as in the assessment",
           "all_findings_sized": "every confirmed finding must appear with its mean diameter in mm",
           "guideline_cited_and_real": "cite at least one guideline chunk id in square brackets, exactly as retrieved",
           "cases_cited_and_retrieved": "cite one or two similar case ids in square brackets, exactly as retrieved",
           "no_unsupported_numbers": "remove any mm value that is not a measured mean diameter"}
    return [msg[k] for k, ok in s["checks"].items() if not ok]


# ----------------------------------------------------------------- roles
def researcher(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, patient_id: str, question: str,
               board: Blackboard, name: str = "researcher") -> None:
    cfg = AgentConfig(name=name, tool_names=RESEARCH_TOOLS, max_iterations=5, token_budget=14000, system_prompt=(
        "You are a research assistant for a radiologist. Answer the question with tools only. Report the facts and "
        "the ids (finding ids, chunk_id, case_id) you found. Be brief. Do not write the final brief."))
    st = run_agent(llm, registry, ctx, f"Patient {patient_id}. {question}", cfg)
    board.append("notes", {"by": name, "question": question, "answer": st.answer[:1500], "status": st.status}, name)
    board.append("observations", tool_messages(st), name)


def writer(llm: LLMClient, board: Blackboard, patient_id: str, feedback: list[str] | None = None) -> str:
    snap = board.snapshot()
    obs = [o for batch in snap.get("observations", []) for o in batch]
    ev = collect_evidence(obs)
    facts = {"assessment": ev["assessment"], "guideline_chunk_ids": sorted(ev["chunk_ids"]),
             "similar_cases": [c for o in obs if o.get("ok") and o["name"] == "search_similar_cases"
                               for c in o["result"]["cases"]][:4],
             "research_notes": [n["answer"] for n in snap.get("notes", [])]}
    prompt = f"Write the nodule assessment brief for patient {patient_id}. {BRIEF_SPEC}\n\nFacts:\n{json.dumps(facts)[:9000]}"
    if feedback:
        prompt += ("\n\nYour previous draft:\n" + (snap.get("brief") or "") +
                   "\n\nFix these problems:\n" + "\n".join(f"- {f}" for f in feedback))
    r = llm.chat([{"role": "system", "content": "You write concise clinical briefs strictly from the facts given. "
                   "Never add a number, category or citation that is not in the facts."},
                  {"role": "user", "content": prompt}])
    _, version = board.read("brief")
    board.write("brief", r.content.strip(), "writer", expected_version=version)
    return r.content.strip()


def critic(llm: LLMClient, board: Blackboard, patient_id: str) -> Review:
    snap = board.snapshot()
    ev = collect_evidence([o for batch in snap.get("observations", []) for o in batch])
    issues = deterministic_issues(snap.get("brief") or "", patient_id, ev)
    if issues:                                   # hard failures: no need to spend a model call
        review = Review(approved=False, issues=issues)
    else:
        res = generate_structured(llm, Review, [
            {"role": "system", "content": "You review a clinical brief against the facts. Approve only if every "
             "statement is supported by the facts. List concrete unsupported statements as issues."},
            {"role": "user", "content": f"Facts:\n{json.dumps(ev['assessment'])[:5000]}\n\nBrief:\n{snap.get('brief')}"}],
            max_attempts=2)
        review = res.value or Review(approved=True, issues=[])
    board.append("reviews", review.model_dump(), "critic")
    return review


def _finish(pattern: str, meter: _Meter, board: Blackboard, t0: float, revisions: int, status: str) -> TeamResult:
    snap = board.snapshot()
    ev = collect_evidence([o for batch in snap.get("observations", []) for o in batch])
    return TeamResult(pattern, snap.get("brief") or "", status, meter.tokens, meter.calls, time.perf_counter() - t0,
                      revisions, {"case_ids": sorted(ev["case_ids"]), "chunk_ids": sorted(ev["chunk_ids"])},
                      board.log, board.conflicts)


def _write_review_loop(meter: _Meter, board: Blackboard, patient_id: str, max_revisions: int = 1) -> tuple[int, str]:
    writer(meter, board, patient_id)
    revisions = 0
    while True:
        review = critic(meter, board, patient_id)
        if review.approved:
            return revisions, "approved"
        if revisions >= max_revisions:
            return revisions, "not_approved"
        revisions += 1
        writer(meter, board, patient_id, review.issues)


def plan_questions(meter: _Meter, patient_id: str) -> list[str]:
    res = generate_structured(meter, Questions, [
        {"role": "system", "content": "You are a planner. Split the request into 2 to 4 independent research "
         "questions that tools can answer: one for the nodule assessment of the study, one for the guideline "
         "management, one for similar published cases."},
        {"role": "user", "content": f"Request: prepare a nodule assessment brief for patient {patient_id}. {BRIEF_SPEC}"}],
        max_attempts=2)
    return res.value.questions if res.value else [
        "Run the nodule assessment for this patient and list the findings and the exam category.",
        "Find the guideline text for the management of this exam category.",
        "Find similar published cases for the dominant finding."]


def run_sequential(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, patient_id: str) -> TeamResult:
    t0, meter, board = time.perf_counter(), _Meter(llm), Blackboard()
    for i, q in enumerate(plan_questions(meter, patient_id), 1):
        researcher(meter, registry, ctx, patient_id, q, board, f"researcher{i}")
    revisions, status = _write_review_loop(meter, board, patient_id)
    return _finish("sequential", meter, board, t0, revisions, status)


def run_network(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, patient_id: str) -> TeamResult:
    """Researchers run concurrently and write to the same blackboard entries."""
    t0, meter, board = time.perf_counter(), _Meter(llm), Blackboard()
    questions = plan_questions(meter, patient_id)
    with cf.ThreadPoolExecutor(max_workers=len(questions)) as pool:
        list(pool.map(lambda iq: researcher(meter, registry, ctx, patient_id, iq[1], board, f"researcher{iq[0]}"),
                      enumerate(questions, 1)))
    revisions, status = _write_review_loop(meter, board, patient_id)
    return _finish("network", meter, board, t0, revisions, status)


class _Q(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=10, max_length=300)


class _None(BaseModel):
    model_config = ConfigDict(extra="forbid")


def run_hierarchical(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, patient_id: str) -> TeamResult:
    """A supervisor agent decides who works next by calling delegation tools."""
    t0, meter, board = time.perf_counter(), _Meter(llm), Blackboard()
    team = ToolRegistry()
    state = {"n": 0, "revisions": 0}

    def delegate_research(a: _Q, c: ToolContext) -> dict[str, Any]:
        state["n"] += 1
        researcher(meter, registry, ctx, patient_id, a.question, board, f"researcher{state['n']}")
        return {"note": board.snapshot()["notes"][-1]["answer"][:600]}

    def delegate_write(a: _None, c: ToolContext) -> dict[str, Any]:
        reviews = board.snapshot().get("reviews") or []
        feedback = reviews[-1]["issues"] if reviews and not reviews[-1]["approved"] else None
        if feedback:
            state["revisions"] += 1
        return {"brief": writer(meter, board, patient_id, feedback)}

    def delegate_review(a: _None, c: ToolContext) -> dict[str, Any]:
        return critic(meter, board, patient_id).model_dump()

    team.register(Tool("delegate_research", "Ask a researcher to answer one question with tools.", _Q,
                       delegate_research, timeout_s=300, max_retries=0, idempotent=False))
    team.register(Tool("delegate_write", "Ask the writer to draft or revise the brief from the research so far.",
                       _None, delegate_write, timeout_s=300, max_retries=0, idempotent=False))
    team.register(Tool("delegate_review", "Ask the critic to review the current brief. Returns approved and issues.",
                       _None, delegate_review, timeout_s=300, max_retries=0, idempotent=False))
    cfg = AgentConfig(name="supervisor", max_iterations=9, token_budget=20000, loop_repeat_limit=4, system_prompt=(
        "You supervise a team that prepares a nodule assessment brief. You do no research or writing yourself. "
        "Delegate: research the nodule assessment, the guideline management and similar published cases (one "
        "delegate_research call each), then delegate_write, then delegate_review. If the review is not approved, "
        "call delegate_write and delegate_review once more. Then reply DONE."))
    st = run_agent(meter, team, ctx, f"Prepare the brief for patient {patient_id}. {BRIEF_SPEC}", cfg)
    reviews = board.snapshot().get("reviews") or []
    status = "approved" if reviews and reviews[-1]["approved"] else ("not_approved" if reviews else f"no_review:{st.stop_reason}")
    return _finish("hierarchical", meter, board, t0, state["revisions"], status)


def run_single(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, patient_id: str) -> TeamResult:
    """The single-agent baseline for the same task."""
    t0, meter = time.perf_counter(), _Meter(llm)
    cfg = AgentConfig(name="single_agent", tool_names=RESEARCH_TOOLS, max_iterations=7, token_budget=20000,
                      system_prompt="You assist a radiologist. Facts, categories and ids come only from tools. "
                                    "Cite chunk_id and case_id values in square brackets exactly as returned.")
    st = run_agent(meter, registry, ctx, f"Prepare a nodule assessment brief for patient {patient_id}. {BRIEF_SPEC}", cfg)
    ev = collect_evidence(tool_messages(st))
    return TeamResult("single", st.answer, st.status, meter.tokens, meter.calls, time.perf_counter() - t0, 0,
                      {"case_ids": sorted(ev["case_ids"]), "chunk_ids": sorted(ev["chunk_ids"])})


PATTERNS = {"single": run_single, "sequential": run_sequential, "hierarchical": run_hierarchical, "network": run_network}

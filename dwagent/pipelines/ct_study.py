"""The CT study pipeline as a runtime graph.

    intake -> quality_gate -+-> reject -> END                       (series unusable)
                            +-> detect -> review -+-> draft_normal ---------+
                                                  +-> categorize -> draft --+-> critical_check
                                                                              -> save_draft -> signoff -> END

Most of it is a fixed workflow. The language model is used at exactly one node (draft), and its
output must pass the schema and the business rules; if it cannot after the bounded retries, a
deterministic template report is used and flagged. signoff pauses for a human approval.
"""
from __future__ import annotations

import json
from typing import Any

from ..ct import lungrads
from ..llm.base import ToolCall
from ..runtime.graph import END, Env, Graph, Interrupt, State
from ..structured.generate import generate_structured
from ..structured.rules import expected_category, validate_report
from ..structured.schemas import CTReport
from ..tools.executor import AwaitingApproval, execute_tool
from ..tools.registry import ToolError

DRAFT_SYSTEM = (
    "You draft chest CT reports for a radiologist to review. You are given the confirmed findings with their "
    "measurements and the Lung-RADS category computed for each one. Write the report as JSON.\n"
    "Rules: report every confirmed finding exactly once and no others; copy laterality, zone, type, slice index "
    "and measurements exactly; use the given categories; state the exam category in the impression as "
    "'Lung-RADS <category>'; the recommendation must be the given management text; cite only the given guideline "
    "chunk ids; mention no measurement that is not in the findings. Use neutral radiology language.")


def _tool(env: Env, name: str, **arguments: Any) -> Any:
    res = execute_tool(env.services["registry"], ToolCall(id=f"node.{name}", name=name, arguments=arguments), env.ctx)
    if not res.ok:
        raise ToolError(res.error_kind or "failed", f"{name}: {res.error}")
    return res.content


def intake(state: State, env: Env) -> dict[str, Any]:
    return {"meta": _tool(env, "get_series_metadata", patient_id=state["patient_id"])}


def quality_gate(state: State, env: Env) -> dict[str, Any]:
    return {"quality": _tool(env, "check_series_quality", patient_id=state["patient_id"])}


def reject(state: State, env: Env) -> dict[str, Any]:
    return {"outcome": "rejected_quality", "report": None}


def detect(state: State, env: Env) -> dict[str, Any]:
    det = _tool(env, "run_nodule_detector", patient_id=state["patient_id"])
    return {"detector": {"version": det["detector"], "n_candidates": det["n_candidates"]}}


def review(state: State, env: Env) -> dict[str, Any]:
    r = _tool(env, "get_confirmed_findings", patient_id=state["patient_id"])
    return {"evidence": {"patient_id": state["patient_id"], "findings": r["findings"]},
            "review": {k: r[k] for k in ("review", "candidates_reviewed", "candidates_rejected", "findings_added_by_reader")}}


def categorize(state: State, env: Env) -> dict[str, Any]:
    cats = {}
    for f in state["evidence"]["findings"]:
        c = _tool(env, "lung_rads_category", nodule_type=f["nodule_type"], mean_diameter_mm=f["mean_diameter_mm"],
                  solid_component_mm=f["solid_component_mm"], is_baseline=True,
                  benign_calcification=f["benign_calcification"], suspicious_features=f["spiculated"])
        cats[f["finding_id"]] = {"category": c["category"], "rule": c["rule"]}
    exam = lungrads.exam_category([c["category"] for c in cats.values()])
    guide = _tool(env, "search_guideline", query=f"Lung-RADS category {exam} management", k=2)
    return {"categories": cats, "exam_category": exam, "management": lungrads.MANAGEMENT[exam],
            "guideline_chunks": [c["chunk_id"] for c in guide["chunks"]]}


def template_report(state: State, note: str) -> dict[str, Any]:
    """Deterministic report built only from the evidence. Used for normal studies and as the
    fallback when the model's draft cannot be validated."""
    m, ev = state["meta"], state["evidence"]
    findings = []
    for f in ev["findings"]:
        cat = expected_category(f)["category"]
        solid = f", solid component {f['solid_component_mm']} mm" if f.get("solid_component_mm") is not None else ""
        findings.append({"finding_id": f["finding_id"], "laterality": f["laterality"], "zone": f["zone"],
                         "nodule_type": f["nodule_type"], "mean_diameter_mm": f["mean_diameter_mm"],
                         "solid_component_mm": f.get("solid_component_mm"), "slice_index": f["slice_index"],
                         "lung_rads": cat,
                         "description": f"{f['nodule_type'].replace('_', '-').capitalize()} nodule in the {f['laterality']} "
                                        f"{f['zone']} zone, mean diameter {f['mean_diameter_mm']} mm{solid} "
                                        f"(image {f['slice_index']})."})
    exam = lungrads.exam_category([f["lung_rads"] for f in findings])
    n = len(findings)
    return CTReport(
        patient_id=ev["patient_id"],
        technique=f"Chest CT, {m.get('SliceThickness')} mm slices, {m.get('ConvolutionKernel')} kernel, "
                  f"{'with' if m.get('ContrastBolusAgent') else 'without'} intravenous contrast.",
        comparison="None available.", findings=findings,
        impression=(f"{n} pulmonary nodule{'s' if n != 1 else ''}. " if n else "No pulmonary nodule identified. ")
                   + f"Lung-RADS {exam}.",
        exam_lung_rads=exam, recommendation=lungrads.MANAGEMENT[exam],
        guideline_citations=state.get("guideline_chunks") or [f"lung_rads_v2022#category-{'1-negative' if exam == '1' else '2-benign-appearance-or-behaviour'}"],
        limitations=note).model_dump()


def draft_normal(state: State, env: Env) -> dict[str, Any]:
    return {"exam_category": "1", "management": lungrads.MANAGEMENT["1"], "categories": {},
            "guideline_chunks": ["lung_rads_v2022#category-1-negative"],
            "report": template_report({**state, "guideline_chunks": ["lung_rads_v2022#category-1-negative"]},
                                      "Template report: no confirmed findings, no language model was used."),
            "draft": {"source": "template", "attempts": 0, "reason": "no findings"}}


def draft(state: State, env: Env) -> dict[str, Any]:
    ev, m = state["evidence"], state["meta"]
    facts = {"patient_id": ev["patient_id"],
             "technique_facts": {"slice_thickness_mm": m.get("SliceThickness"), "kernel": m.get("ConvolutionKernel"),
                                 "contrast": bool(m.get("ContrastBolusAgent")), "scanner": m.get("ManufacturerModelName")},
             "comparison": "None available.",
             "confirmed_findings": [{**{k: f[k] for k in ("finding_id", "laterality", "zone", "nodule_type",
                                                           "mean_diameter_mm", "solid_component_mm", "slice_index",
                                                           "benign_calcification", "spiculated")},
                                     "lung_rads": state["categories"][f["finding_id"]]["category"]} for f in ev["findings"]],
             "exam_lung_rads": state["exam_category"], "management": state["management"],
             "guideline_chunk_ids": state["guideline_chunks"], "quality_warnings": state["quality"]["warnings"]}
    result = generate_structured(
        env.services["llm"], CTReport,
        [{"role": "system", "content": DRAFT_SYSTEM},
         {"role": "user", "content": "Draft the report from these facts:\n" + json.dumps(facts)}],
        rules=lambda r: validate_report(r, ev), max_attempts=env.services.get("draft_attempts", 3),
        ctx=env.ctx, span_name="draft_report")
    usage = dict(state["usage"])
    usage["tokens"] = usage.get("tokens", 0) + result.tokens
    log = [{"attempt": a.number, "stage": a.stage, "errors": a.errors[:4]} for a in result.attempts]
    if result.value is not None:
        return {"report": result.value.model_dump(), "usage": usage,
                "draft": {"source": "model", "attempts": len(result.attempts), "log": log}}
    return {"report": template_report(state, "Template report: the model draft failed validation "
                                             f"{len(result.attempts)} times and was discarded."),
            "usage": usage, "draft": {"source": "template_fallback", "attempts": len(result.attempts), "log": log}}


def critical_check(state: State, env: Env) -> dict[str, Any]:
    critical = state["exam_category"] in ("4B", "4X")
    return {"critical": critical,
            "critical_note": "Category 4B/4X: critical result, must be acknowledged at sign-off." if critical else ""}


def save_draft(state: State, env: Env) -> dict[str, Any]:
    saved = _tool(env, "save_draft_report", patient_id=state["patient_id"], report=state["report"])
    return {"report_id": saved["report_id"], "outcome": "draft_ready"}


def signoff(state: State, env: Env) -> dict[str, Any]:
    """Human approval gate. The sign_report tool requires approval; until a radiologist approves
    the run is interrupted here and can be resumed later, even after a restart."""
    try:
        res = execute_tool(env.services["registry"], ToolCall(id="node.sign_report", name="sign_report",
                                                              arguments={"report_id": state["report_id"]}), env.ctx)
    except AwaitingApproval as wait:
        raise Interrupt("awaiting_signoff", {"approval_id": wait.approval_id, "report_id": state["report_id"],
                                             "critical": state.get("critical", False)}) from None
    if not res.ok:
        return {"outcome": "signoff_rejected" if res.error_kind == "rejected" else "signoff_failed",
                "signoff_error": res.error}
    return {"outcome": "signed", "signed_by": res.content["signed_by"]}


def build() -> Graph:
    g = Graph("ct_study")
    for name, fn in [("intake", intake), ("quality_gate", quality_gate), ("reject", reject), ("detect", detect),
                     ("review", review), ("categorize", categorize), ("draft_normal", draft_normal), ("draft", draft),
                     ("critical_check", critical_check), ("save_draft", save_draft), ("signoff", signoff)]:
        g.node(name, fn)
    g.edge("intake", "quality_gate")
    g.route("quality_gate", lambda s: "detect" if s["quality"]["passed"] else "reject", {"detect", "reject"})
    g.edge("reject", END)
    g.edge("detect", "review")
    g.route("review", lambda s: "categorize" if s["evidence"]["findings"] else "draft_normal", {"categorize", "draft_normal"})
    g.edge("categorize", "draft")
    g.edge("draft_normal", "critical_check")
    g.edge("draft", "critical_check")
    g.edge("critical_check", "save_draft")
    g.edge("save_draft", "signoff")
    g.edge("signoff", END)
    return g

"""Business-rule validation: the second layer, after the schema.

The schema says a report is well formed. These rules say it is *true to the evidence*: every
finding exists, every number matches a measurement, every category is the one the
deterministic Lung-RADS code computes, and every citation points at a real guideline chunk.
Authorization (who may draft, sign or read) is a third, separate layer in authz.py.
"""
from __future__ import annotations

import re
from typing import Any

from ..ct import lungrads
from ..rag import kb
from .schemas import CTReport, PatientExplanation

_MM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm|millimet(?:er|re)s?)\b", re.I)
_INTERVAL = {"1": "12 months", "2": "12 months", "3": "6 months", "4A": "3 months",
             "4B": "diagnostic chest ct", "4X": "diagnostic chest ct"}


def expected_category(f: dict[str, Any]) -> dict[str, Any]:
    return lungrads.categorize(f["nodule_type"], f["mean_diameter_mm"], f.get("solid_component_mm"),
                               is_baseline=True, benign_calcification=bool(f.get("benign_calcification")),
                               suspicious_features=bool(f.get("spiculated")))


def validate_report(report: CTReport, evidence: dict[str, Any]) -> list[str]:
    """``evidence`` = {"patient_id", "findings": [confirmed findings with measurements]}."""
    errors: list[str] = []
    ev = {f["finding_id"]: f for f in evidence["findings"]}
    if report.patient_id != evidence["patient_id"]:
        errors.append(f"patient_id must be {evidence['patient_id']!r}")
    ids = [f.finding_id for f in report.findings]
    for fid in ids:
        if fid not in ev:
            errors.append(f"finding {fid} is not in the confirmed findings; only report {sorted(ev)}")
    if len(set(ids)) != len(ids):
        errors.append("each finding_id may appear only once")
    for fid in ev:
        if fid not in ids:
            errors.append(f"confirmed finding {fid} is missing from the report")
    allowed_mm: set[float] = set()
    cats = []
    for e in ev.values():
        allowed_mm.update(v for v in (e["mean_diameter_mm"], e.get("long_axis_mm"), e.get("short_axis_mm"),
                                      e.get("solid_component_mm")) if v is not None)
    for f in report.findings:
        e = ev.get(f.finding_id)
        if e is None:
            continue
        for name in ("laterality", "zone", "nodule_type", "slice_index"):
            if getattr(f, name) != e[name]:
                errors.append(f"{f.finding_id}.{name} is {getattr(f, name)!r} but the measurement says {e[name]!r}")
        if abs(f.mean_diameter_mm - e["mean_diameter_mm"]) > 0.05:
            errors.append(f"{f.finding_id}.mean_diameter_mm is {f.mean_diameter_mm} but was measured as {e['mean_diameter_mm']}")
        if e.get("solid_component_mm") is not None and (f.solid_component_mm is None
                                                        or abs(f.solid_component_mm - e["solid_component_mm"]) > 0.05):
            errors.append(f"{f.finding_id}.solid_component_mm must be {e['solid_component_mm']}")
        want = expected_category(e)["category"]
        cats.append(want)
        if f.lung_rads != want:
            errors.append(f"{f.finding_id}.lung_rads is {f.lung_rads} but the Lung-RADS rules give {want}")
        errors += _unsupported_numbers(f"{f.finding_id}.description", f.description, allowed_mm)
    exam = lungrads.exam_category(cats)
    if report.exam_lung_rads != exam:
        errors.append(f"exam_lung_rads is {report.exam_lung_rads} but the most suspicious finding gives {exam}")
    if f"lung-rads {exam}".lower() not in report.impression.lower():
        errors.append(f"impression must state 'Lung-RADS {exam}'")
    if _INTERVAL[exam] not in report.recommendation.lower():
        errors.append(f"recommendation must follow Lung-RADS {exam} management ('{lungrads.MANAGEMENT[exam]}')")
    errors += _unsupported_numbers("impression", report.impression, allowed_mm)
    known = kb.chunk_ids()
    for cid in report.guideline_citations:
        if cid not in known:
            errors.append(f"guideline citation {cid!r} does not exist in the knowledge base")
    if not report.guideline_citations:
        errors.append("guideline_citations must contain at least one knowledge-base chunk id")
    return errors


def _unsupported_numbers(where: str, text: str, allowed: set[float]) -> list[str]:
    out = []
    for m in _MM.finditer(text):
        v = float(m.group(1))
        if not any(abs(v - a) <= 0.05 for a in allowed) and v not in (2.5, 6.0, 8.0, 15.0, 30.0, 4.0, 1.5):
            out.append(f"{where} mentions {v} mm, which is not one of the measurements")
    return out


def validate_explanation(exp: PatientExplanation, report: dict[str, Any]) -> list[str]:
    """A patient explanation may only restate the signed report."""
    errors = []
    if exp.next_step.strip() != report["recommendation"].strip():
        errors.append("next_step must be copied word for word from the report recommendation: "
                      + repr(report["recommendation"]))
    allowed = {f["mean_diameter_mm"] for f in report["findings"]}
    allowed |= {f["solid_component_mm"] for f in report["findings"] if f.get("solid_component_mm") is not None}
    for part in [exp.summary, *exp.what_was_found]:
        errors += _unsupported_numbers("explanation", part, allowed)
    lowered = " ".join([exp.summary, *exp.what_was_found]).lower()
    for banned in ("cancer-free", "you have cancer", "definitely", "nothing to worry about", "you should take",
                   "you do not need"):
        if banned in lowered:
            errors.append(f"the explanation must not say {banned!r}: it may not diagnose, reassure absolutely or advise treatment")
    known = kb.chunk_ids()
    for t in exp.terms:
        if t not in known or not t.startswith("patient_glossary#"):
            errors.append(f"term {t!r} is not a patient glossary chunk id")
    return errors

"""Skill: explain a SIGNED report to the patient in plain language.

The explanation is structured (PatientExplanation) and must pass rules that keep it inside the
report: the follow-up sentence is copied word for word, every millimetre value must be one of
the report's measurements, glossary references must be real patient-glossary chunks, and
phrases that diagnose, reassure absolutely or advise treatment are rejected.
"""
from __future__ import annotations

import json
from typing import Any

from ..llm.base import LLMClient
from ..rag import kb
from ..structured.generate import StructuredResult, generate_structured
from ..structured.rules import validate_explanation
from ..structured.schemas import PatientExplanation
from ..tools.registry import ToolContext

SYSTEM = (
    "You explain a signed chest CT report to the patient it belongs to. Use short, calm, plain sentences at a "
    "school reading level. Only restate what the report says. Do not add a diagnosis, a probability, reassurance "
    "that nothing is wrong, or any treatment advice. Copy the follow-up recommendation word for word into "
    "next_step. In terms, list the glossary chunk ids for the medical words you used. Suggest questions the "
    "patient could ask their doctor. Set urgent_escalation to true only if the patient's message mentions "
    "coughing blood, chest pain or trouble breathing.")


def explain_report(llm: LLMClient, report: dict[str, Any], patient_message: str = "",
                   ctx: ToolContext | None = None, max_attempts: int = 3) -> StructuredResult[PatientExplanation]:
    glossary = [{"chunk_id": c.chunk_id, "term": c.title} for c in kb.load_chunks() if c.audience == "patient"]
    facts = {"findings": [{k: f[k] for k in ("laterality", "zone", "nodule_type", "mean_diameter_mm",
                                             "solid_component_mm", "lung_rads")} for f in report["findings"]],
             "impression": report["impression"], "exam_lung_rads": report["exam_lung_rads"],
             "recommendation": report["recommendation"], "glossary": glossary}
    user = "Signed report:\n" + json.dumps(facts)
    if patient_message:
        user += f"\n\nThe patient wrote: {patient_message}"
    return generate_structured(llm, PatientExplanation,
                               [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                               rules=lambda e: validate_explanation(e, report), max_attempts=max_attempts,
                               ctx=ctx, span_name="patient_explanation")

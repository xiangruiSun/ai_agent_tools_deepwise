"""Training-data governance: turning free text into labels that can be trusted.

``extract_case_labels`` asks the model for structured labels from a published case summary
and accepts them only if they are anchored in the text: the supporting quote must be an exact
substring, and the stated size must be the number in that quote (centimetres converted).
Labels that cannot be anchored are sent to human review instead of entering a training set.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from .config import get_settings
from .llm.base import LLMClient
from .structured.generate import StructuredResult, generate_structured
from .structured.schemas import CaseLabels

SYSTEM = (
    "You label published clinical case summaries for a lung-nodule training dataset. Read the text and fill the "
    "fields. largest_lesion_mm is the size of the largest pulmonary nodule or lung mass in millimetres (convert "
    "centimetres); use null if no size is given or there is no pulmonary nodule or mass. evidence_quote must be "
    "copied exactly from the text and must contain that size; use an empty string when largest_lesion_mm is null. "
    "Do not guess.")

_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[x×]\s*\d+(?:\.\d+)?\s*)*(mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?)", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def label_rules(labels: CaseLabels, text: str) -> list[str]:
    errors = []
    if labels.largest_lesion_mm is None:
        return errors
    if not labels.has_pulmonary_nodule_or_mass:
        errors.append("largest_lesion_mm is set but has_pulmonary_nodule_or_mass is false")
    quote = _norm(labels.evidence_quote)
    if not quote or quote not in _norm(text):
        errors.append("evidence_quote must be copied exactly from the text")
        return errors
    sizes = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)", labels.evidence_quote):
        v = float(m.group(1))
        sizes.update({v, v * 10})          # the quote may give centimetres
    if not any(abs(labels.largest_lesion_mm - s) < 0.051 for s in sizes):
        errors.append(f"largest_lesion_mm {labels.largest_lesion_mm} does not match any number in evidence_quote")
    return errors


def extract_case_labels(llm: LLMClient, text: str, max_attempts: int = 3) -> StructuredResult[CaseLabels]:
    return generate_structured(llm, CaseLabels,
                               [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text[:6000]}],
                               rules=lambda lab: label_rules(lab, text), max_attempts=max_attempts)


def fetch_case_texts(query: str, n: int = 5) -> list[dict[str, Any]]:
    """Real published case summaries from the deepwise_rag service (PMC-Patients subset)."""
    s = get_settings()
    h = {"X-API-Key": s.rag_api_key}
    r = httpx.post(f"{s.rag_url}/api/v1/search", headers=h, timeout=30,
                   json={"query_mode": "text", "query_text": query, "top_k": n})
    r.raise_for_status()
    out = []
    for x in r.json()["results"]:
        reps = httpx.get(f"{s.rag_url}/api/v1/cases/{x['case_id']}/reports", headers=h, timeout=30).json()
        if reps:
            out.append({"case_id": x["case_id"], "source_id": reps[0]["report_id"], "text": reps[0]["impression_text"]})
    return out


_TEXTURE_WORDS = {"ground_glass": ("ground-glass", "ground glass", "non-solid", "nonsolid"),
                  "part_solid": ("part-solid", "part solid", "subsolid", "sub-solid"),
                  "solid": ("solid",)}


def cohort_rules(query: Any, request: str) -> list[str]:
    """Business rules for a model-written cohort definition: anything the request names must be
    in the query. Written after the model was seen to leave ``texture`` null in 3 of 5 requests."""
    errors = []
    text = request.lower()
    named = None
    for texture, words in _TEXTURE_WORDS.items():          # most specific first
        if any(w in text for w in words):
            named = texture
            break
    if named and query.texture != named:
        errors.append(f"the request names {named.replace('_', '-')} nodules, so texture must be '{named}'")
    if "non-contrast" in text or "without contrast" in text:
        if query.contrast is not False:
            errors.append("the request asks for non-contrast scans, so contrast must be false")
    elif "contrast" in text and query.contrast is not True:
        errors.append("the request asks for contrast-enhanced scans, so contrast must be true")
    if (query.min_diameter_mm is not None and query.max_diameter_mm is not None
            and query.min_diameter_mm > query.max_diameter_mm):
        errors.append("min_diameter_mm is larger than max_diameter_mm")
    return errors


def regex_sizes_mm(text: str) -> list[float]:
    """Independent, model-free size extraction used to cross-check the model's labels."""
    out = []
    for m in _NUM.finditer(text):
        v = float(m.group(1))
        out.append(round(v * 10 if m.group(2).lower().startswith("c") else v, 1))
    return out

"""Skill: assess_nodules. The nodule assessment workflow for one chest CT study.

    quality gate -> confirmed findings -> Lung-RADS category per finding -> exam category
    -> management -> supporting guideline chunks

Business rules encoded here rather than left to the model:
  * a series that fails the quality gate is never analysed;
  * every confirmed finding is categorised, none may be skipped;
  * spiculation is passed as a suspicious feature;
  * the exam category is the most suspicious finding's category;
  * category 4B and 4X are flagged as critical.
"""
from __future__ import annotations

from typing import Any

from ..ct import lungrads
from ..tools.builtin import PatientArg
from ..tools.registry import Tool, ToolContext, ToolRegistry
from .base import SkillRunner


def make_assess_nodules(registry: ToolRegistry) -> Tool:
    def assess_nodules(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
        run = SkillRunner(registry, ctx, "assess_nodules")
        quality = run.call("check_series_quality", patient_id=a.patient_id)
        if not quality["passed"]:
            return {"patient_id": a.patient_id, "status": "rejected_quality", "issues": quality["issues"],
                    "next_step": "route to rescan or manual read; no automated analysis was run",
                    "steps": run.steps}
        reviewed = run.call("get_confirmed_findings", patient_id=a.patient_id)
        findings = []
        for f in reviewed["findings"]:
            cat = run.call("lung_rads_category", nodule_type=f["nodule_type"], mean_diameter_mm=f["mean_diameter_mm"],
                           solid_component_mm=f["solid_component_mm"], is_baseline=True,
                           benign_calcification=f["benign_calcification"], suspicious_features=f["spiculated"])
            findings.append({**{k: f[k] for k in ("finding_id", "laterality", "zone", "nodule_type", "mean_diameter_mm",
                                                  "solid_component_mm", "slice_index", "source")},
                             "lung_rads": cat["category"], "rule": cat["rule"]})
        exam = lungrads.exam_category([f["lung_rads"] for f in findings])
        guide = run.call("search_guideline", query=f"Lung-RADS category {exam} management", k=2)
        return {"patient_id": a.patient_id, "status": "assessed", "warnings": quality["warnings"],
                "review": reviewed["review"], "findings": findings, "exam_lung_rads": exam,
                "management": lungrads.MANAGEMENT[exam], "critical": exam in ("4B", "4X"),
                "guideline_chunks": [c["chunk_id"] for c in guide["chunks"]], "steps": run.steps}

    return Tool("assess_nodules",
                "Complete nodule assessment for one patient's chest CT: quality gate, confirmed findings, Lung-RADS "
                "category for every finding, exam category, management and guideline references. Prefer this over "
                "calling the individual tools.", PatientArg, assess_nodules,
                scopes=frozenset({"study:analyze", "kb:read"}), timeout_s=180, max_retries=1)

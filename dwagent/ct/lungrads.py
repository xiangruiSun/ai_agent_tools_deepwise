"""Deterministic Lung-RADS v2022 nodule categorisation.

The category is always computed here, in code, never by the language model. The thresholds
follow the ACR Lung-RADS v2022 assessment categories for solid, part-solid and non-solid
(ground-glass) nodules at baseline and at follow-up. Simplifications are listed in
requirements/09_ct_diagnosis_pipeline.md: airway nodules, atypical cysts and the step-down of
stable category 3 and 4A nodules are not modelled.
"""
from __future__ import annotations

from typing import Any

ORDER = ["1", "2", "3", "4A", "4B", "4X"]

MANAGEMENT = {
    "1": "Continue annual screening with low-dose CT in 12 months.",
    "2": "Continue annual screening with low-dose CT in 12 months.",
    "3": "Low-dose CT in 6 months.",
    "4A": "Low-dose CT in 3 months; PET/CT may be considered when a solid nodule or solid component is 8 mm or larger.",
    "4B": "Diagnostic chest CT with or without contrast; PET/CT may be considered when a solid nodule or solid "
          "component is 8 mm or larger; tissue sampling or referral for further clinical evaluation.",
    "4X": "Diagnostic chest CT with or without contrast; PET/CT may be considered when a solid nodule or solid "
          "component is 8 mm or larger; tissue sampling or referral for further clinical evaluation.",
}


class LungRadsInputError(ValueError):
    pass


def categorize(nodule_type: str, mean_diameter_mm: float, solid_component_mm: float | None = None,
               is_baseline: bool = True, is_new: bool = False, is_growing: bool = False,
               benign_calcification: bool = False, suspicious_features: bool = False) -> dict[str, Any]:
    d = round(float(mean_diameter_mm), 1)
    if d <= 0:
        raise LungRadsInputError("mean_diameter_mm must be greater than 0")
    if nodule_type not in ("solid", "part_solid", "ground_glass"):
        raise LungRadsInputError("nodule_type must be solid, part_solid or ground_glass")
    if is_baseline and (is_new or is_growing):
        raise LungRadsInputError("a baseline nodule cannot be new or growing")
    if benign_calcification:
        cat, rule = "1", "nodule with a benign calcification pattern"
    elif nodule_type == "solid":
        if is_new:
            cat, rule = (("2", "new solid nodule < 4 mm") if d < 4 else ("3", "new solid nodule 4 to < 6 mm") if d < 6
                         else ("4A", "new solid nodule 6 to < 8 mm") if d < 8 else ("4B", "new solid nodule >= 8 mm"))
        elif is_growing:
            cat, rule = ("4A", "growing solid nodule < 8 mm") if d < 8 else ("4B", "growing solid nodule >= 8 mm")
        else:
            cat, rule = (("2", "solid nodule < 6 mm") if d < 6 else ("3", "solid nodule 6 to < 8 mm") if d < 8
                         else ("4A", "solid nodule 8 to < 15 mm") if d < 15 else ("4B", "solid nodule >= 15 mm"))
    elif nodule_type == "part_solid":
        if solid_component_mm is None:
            raise LungRadsInputError("solid_component_mm is required for a part-solid nodule")
        s = round(float(solid_component_mm), 1)
        if s > d:
            raise LungRadsInputError("solid_component_mm cannot exceed mean_diameter_mm")
        if is_new or is_growing:
            if is_new and d < 6:
                cat, rule = "3", "new part-solid nodule < 6 mm total"
            elif s < 4:
                cat, rule = "4A", "new or growing solid component < 4 mm"
            else:
                cat, rule = "4B", "new or growing solid component >= 4 mm"
        elif d < 6:
            cat, rule = "2", "part-solid nodule < 6 mm total"
        elif s < 6:
            cat, rule = "3", "part-solid nodule >= 6 mm total with solid component < 6 mm"
        elif s < 8:
            cat, rule = "4A", "part-solid nodule with solid component 6 to < 8 mm"
        else:
            cat, rule = "4B", "part-solid nodule with solid component >= 8 mm"
    else:
        if d < 30:
            cat, rule = "2", "non-solid nodule < 30 mm"
        elif is_baseline or is_new:
            cat, rule = "3", "non-solid nodule >= 30 mm at baseline or new"
        else:
            cat, rule = "2", "non-solid nodule >= 30 mm, stable or slowly growing"
    if suspicious_features and cat in ("3", "4A", "4B"):
        cat, rule = "4X", rule + ", with additional features that increase suspicion"
    return {"category": cat, "rule": rule, "management": MANAGEMENT[cat], "mean_diameter_mm": d,
            "guideline": "ACR Lung-RADS v2022"}


def exam_category(categories: list[str]) -> str:
    """The exam takes the most suspicious nodule's category; no nodules means category 1."""
    return max(categories, key=ORDER.index) if categories else "1"

"""Structured outputs the model must produce. Pydantic gives the JSON Schema and the first
validation layer (types, enums, ranges, required fields)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["1", "2", "3", "4A", "4B", "4X"]
NoduleType = Literal["solid", "part_solid", "ground_glass"]


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str = Field(description="Identifier of the confirmed finding this sentence describes, e.g. F1.")
    laterality: Literal["right", "left"]
    zone: Literal["upper", "mid", "lower"]
    nodule_type: NoduleType
    mean_diameter_mm: float = Field(gt=0, lt=100)
    # Required but nullable: with schema-constrained decoding an optional field is silently
    # dropped by the model, so every field the rules check is required.
    solid_component_mm: float | None = Field(ge=0, lt=100, description="Null unless the nodule is part-solid.")
    slice_index: int = Field(ge=0)
    lung_rads: Category
    description: str = Field(min_length=10, max_length=400, description="One or two report sentences.")


class CTReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    technique: str = Field(min_length=5, max_length=300)
    comparison: str = Field(max_length=200)
    findings: list[Finding] = Field(max_length=12)
    impression: str = Field(min_length=10, max_length=600)
    exam_lung_rads: Category
    recommendation: str = Field(min_length=10, max_length=400)
    guideline_citations: list[str] = Field(description="Knowledge-base chunk ids that support the recommendation.")
    limitations: str = Field(max_length=400)


class PatientExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=20, max_length=700, description="Plain-language summary of the signed report.")
    what_was_found: list[str] = Field(max_length=6)
    terms: list[str] = Field(max_length=6, description="Glossary terms used, each one a glossary chunk id.")
    next_step: str = Field(description="The follow-up recommendation, copied word for word from the report.")
    questions_for_doctor: list[str] = Field(max_length=4)
    urgent_escalation: bool = Field(description="True when the patient mentions symptoms that need urgent care.")


class CaseLabels(BaseModel):
    """Labels extracted from a published case summary, for training-data curation."""
    model_config = ConfigDict(extra="forbid")
    has_pulmonary_nodule_or_mass: bool
    largest_lesion_mm: float | None = Field(gt=0, lt=300)
    laterality: Literal["right", "left", "bilateral", "not_stated"]
    imaging_modalities: list[Literal["CT", "PET", "MRI", "X-ray", "ultrasound"]] = Field(max_length=5)
    malignancy_confirmed: Literal["yes", "no", "not_stated"]
    evidence_quote: str = Field(max_length=300, description="Exact words from the text that support the size.")


class CohortQueryStrict(BaseModel):
    """The same fields as CohortQuery, all required (nullable). Used when the model generates a
    cohort definition: under schema-constrained decoding it omitted every optional field."""
    model_config = ConfigDict(extra="forbid")
    min_diameter_mm: float | None = Field(ge=0, le=100, description="Smallest nodule size in mm, or null.")
    max_diameter_mm: float | None = Field(ge=0, le=100, description="Largest nodule size in mm, or null.")
    max_slice_thickness_mm: float | None = Field(gt=0, le=10, description="Thickest allowed slice in mm, or null.")
    contrast: bool | None = Field(description="true = contrast-enhanced only, false = non-contrast only, null = either.")
    min_readers: int = Field(ge=1, le=4, description="Minimum number of reader annotations on the scan; 1 if not stated.")
    texture: Literal["solid", "part_solid", "ground_glass"] | None = Field(
        description="Set to solid, part_solid or ground_glass whenever the request names a nodule texture "
                    "(ground-glass and non-solid both mean ground_glass). Null only if no texture is named.")
    min_malignancy: int | None = Field(ge=1, le=5, description="Minimum reader malignancy rating 1-5, or null.")


class CohortQuery(BaseModel):
    """A cohort definition over the LIDC-IDRI annotation database."""
    model_config = ConfigDict(extra="forbid")
    min_diameter_mm: float | None = Field(default=None, ge=0, le=100)
    max_diameter_mm: float | None = Field(default=None, ge=0, le=100)
    max_slice_thickness_mm: float | None = Field(default=None, gt=0, le=10)
    contrast: bool | None = None
    min_readers: int = Field(default=1, ge=1, le=4)
    texture: Literal["solid", "part_solid", "ground_glass"] | None = None
    min_malignancy: int | None = Field(default=None, ge=1, le=5)

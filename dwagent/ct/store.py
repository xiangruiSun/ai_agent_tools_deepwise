"""Access to the local CT archive (a stand-in for PACS): study list, DICOM headers, HU volumes,
acquisition quality checks and a de-identification audit of the headers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from ..config import get_settings


class StudyNotFound(KeyError):
    pass


def manifest() -> dict[str, Any]:
    path = get_settings().lidc_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("data/lidc/manifest.json is missing; run scripts/download_lidc.py")
    return json.loads(path.read_text())


def list_studies() -> list[dict[str, Any]]:
    return [{"patient_id": s["patient_id"], "study_uid": s["study_uid"], "series_uid": s["series_uid"],
             "n_slices": s["n_files"], "slice_thickness_mm": s["slice_thickness"]} for s in manifest()["studies"]]


def _extras_path() -> Path:
    return get_settings().runtime_dir / "extra_studies.json"


def register_study(study_key: str, path: Path, study_uid: str, series_uid: str) -> None:
    """Register a series received by the DICOM gateway under a study key."""
    extras = json.loads(_extras_path().read_text()) if _extras_path().exists() else {}
    files = list(Path(path).glob("*.dcm"))
    extras[study_key] = {"patient_id": study_key, "study_uid": study_uid, "series_uid": series_uid,
                         "path": str(path), "n_files": len(files), "slice_thickness": None, "band": "incoming",
                         "bytes": sum(f.stat().st_size for f in files)}
    _extras_path().write_text(json.dumps(extras))


def study_entry(patient_id: str) -> dict[str, Any]:
    for s in manifest()["studies"]:
        if s["patient_id"] == patient_id:
            return s
    if _extras_path().exists():
        extras = json.loads(_extras_path().read_text())
        if patient_id in extras:
            return extras[patient_id]
    raise StudyNotFound(patient_id)


def series_dir(patient_id: str) -> Path:
    path = Path(study_entry(patient_id)["path"])
    return path if path.is_absolute() else get_settings().root / path


@dataclass
class Volume:
    hu: np.ndarray            # (z, y, x) int16 Hounsfield units, z ascending (towards the head)
    z_mm: np.ndarray          # z position of every slice
    spacing: tuple[float, float, float]  # (dz, dy, dx) in mm
    meta: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.hu.shape  # type: ignore[return-value]


HEADER_FIELDS = ["PatientID", "PatientSex", "PatientAge", "StudyDate", "StudyDescription", "SeriesDescription",
                 "Modality", "Manufacturer", "ManufacturerModelName", "BodyPartExamined", "SliceThickness",
                 "KVP", "ConvolutionKernel", "ContrastBolusAgent", "Rows", "Columns", "PatientPosition"]


@lru_cache(maxsize=4)
def load_volume(patient_id: str) -> Volume:
    cache = get_settings().runtime_dir / "volumes"
    cache.mkdir(exist_ok=True)
    npy, js = cache / f"{patient_id}.npy", cache / f"{patient_id}.json"
    if npy.exists() and js.exists():
        info = json.loads(js.read_text())
        return Volume(np.load(npy), np.array(info["z_mm"]), tuple(info["spacing"]), info["meta"])
    files = sorted(series_dir(patient_id).glob("*.dcm"))
    if not files:
        raise StudyNotFound(patient_id)
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        hu = ds.pixel_array.astype(np.int16) * int(float(ds.get("RescaleSlope", 1))) + int(float(ds.get("RescaleIntercept", 0)))
        slices.append((float(ds.ImagePositionPatient[2]), hu, ds))
    slices.sort(key=lambda t: t[0])
    z = np.array([s[0] for s in slices])
    first = slices[0][2]
    dz = float(np.median(np.diff(z))) if len(z) > 1 else float(first.SliceThickness)
    spacing = (dz, float(first.PixelSpacing[0]), float(first.PixelSpacing[1]))
    meta = {k: _plain(first.get(k)) for k in HEADER_FIELDS}
    meta.update(n_slices=len(slices), pixel_spacing_mm=spacing[1], slice_spacing_mm=round(dz, 3),
                z_gaps=int(np.sum(np.abs(np.diff(z) - dz) > 0.25 * abs(dz))) if len(z) > 1 else 0,
                series_uid=str(first.SeriesInstanceUID), study_uid=str(first.StudyInstanceUID))
    vol = np.stack([s[1] for s in slices])
    np.save(npy, vol)
    js.write_text(json.dumps({"z_mm": z.tolist(), "spacing": spacing, "meta": meta}))
    return Volume(vol, z, spacing, meta)


def _plain(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    s = str(v)
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def series_metadata(patient_id: str) -> dict[str, Any]:
    return dict(load_volume(patient_id).meta)


def check_quality(patient_id: str) -> dict[str, Any]:
    """Acquisition checks that decide whether automated nodule analysis may run at all."""
    m = series_metadata(patient_id)
    issues, warnings = [], []
    if m.get("Modality") != "CT":
        issues.append(f"modality is {m.get('Modality')}, expected CT")
    if (m.get("SliceThickness") or 99) > 2.5:
        issues.append(f"slice thickness {m.get('SliceThickness')} mm exceeds the 2.5 mm limit for nodule measurement")
    if m["z_gaps"] > 0:
        issues.append(f"{m['z_gaps']} gaps between slices: the series is incomplete")
    if m["n_slices"] < 60:
        issues.append(f"only {m['n_slices']} slices: the lungs are unlikely to be fully covered")
    if (m.get("pixel_spacing_mm") or 0) > 1.0:
        warnings.append(f"in-plane pixel spacing {m['pixel_spacing_mm']} mm is coarse")
    if m.get("ContrastBolusAgent"):
        warnings.append("contrast-enhanced series: density-based nodule typing is less reliable")
    return {"patient_id": patient_id, "passed": not issues, "issues": issues, "warnings": warnings,
            "slice_thickness_mm": m.get("SliceThickness"), "n_slices": m["n_slices"],
            "kernel": m.get("ConvolutionKernel"), "manufacturer": m.get("Manufacturer")}


# Header fields that would identify a person. Used by the data-governance tools.
PHI_TAGS = ["PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
            "OtherPatientNames", "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
            "InstitutionName", "InstitutionAddress", "AccessionNumber", "StationName", "DeviceSerialNumber"]


def deid_audit(patient_id: str, sample: int = 5) -> dict[str, Any]:
    """Check DICOM headers for identifying fields. Reads header values but returns only field
    names and whether they are populated, never the values themselves."""
    files = sorted(series_dir(patient_id).glob("*.dcm"))
    step = max(1, len(files) // sample)
    populated: dict[str, int] = {}
    burned_in, removed_flag, method = set(), set(), set()
    private_tags = 0
    checked = files[::step][:sample]
    for f in checked:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        for tag in PHI_TAGS:
            if str(ds.get(tag, "") or "").strip():
                populated[tag] = populated.get(tag, 0) + 1
        burned_in.add(str(ds.get("BurnedInAnnotation", "not stated")))
        removed_flag.add(str(ds.get("PatientIdentityRemoved", "not stated")))
        method.add(str(ds.get("DeidentificationMethod", "not stated"))[:80])
        private_tags += sum(1 for e in ds if e.tag.is_private)
    findings = [f"{tag} is populated in {n}/{len(checked)} sampled files" for tag, n in sorted(populated.items())]
    if "YES" in burned_in:
        findings.append("BurnedInAnnotation=YES: pixel data may contain text")
    if removed_flag != {"YES"}:
        findings.append(f"PatientIdentityRemoved is {sorted(removed_flag)}, expected YES")
    return {"patient_id": patient_id, "files_checked": len(checked), "passed": not findings, "findings": findings,
            "identity_removed_flag": sorted(removed_flag), "deidentification_method": sorted(method),
            "burned_in_annotation": sorted(burned_in), "private_tag_count": private_tags}

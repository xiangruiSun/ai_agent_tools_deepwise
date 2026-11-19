"""Clinical records used by the agents: confirmed findings and reports.

``review_findings`` plays the part of the radiologist's review of detector output. In this
research demo the review is SIMULATED from the LIDC reader annotations: a detector candidate
that matches an annotated nodule is confirmed, an unmatched candidate is rejected, and an
annotated nodule the detector missed is added. In a hospital this step is a person.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .ct import annotations, detector
from .store import Database, dumps


def review_findings(patient_id: str) -> dict[str, Any]:
    det = detector.detect(patient_id)
    nodules = annotations.reader_nodules(patient_id)
    m = annotations.match(det["candidates"], nodules)
    by_nodule = {p["nodule_id"]: p["finding_id"] for p in m["pairs"]}
    confirmed = []
    for i, n in enumerate(nodules, start=1):
        confirmed.append({
            "finding_id": f"F{i}",
            "source": "detector_confirmed" if n["nodule_id"] in by_nodule else "reader_added",
            "detector_candidate": by_nodule.get(n["nodule_id"]),
            "slice_index": n["slice_index"], "center_px": n["center_px"],
            "laterality": n["laterality"], "zone": n["zone"], "nodule_type": n["nodule_type"],
            "mean_diameter_mm": n["mean_diameter_mm"], "long_axis_mm": n["long_axis_mm"],
            "short_axis_mm": n["short_axis_mm"], "solid_component_mm": n["solid_component_mm"],
            "benign_calcification": n["benign_calcification"], "spiculated": n["spiculated"],
            "readers_marking": n["n_readers"],
        })
    return {"patient_id": patient_id, "review": "simulated from LIDC reader annotations",
            "detector": det["detector"], "candidates_reviewed": det["n_candidates"],
            "candidates_rejected": len(m["false_positives"]), "findings_added_by_reader": len(m["missed_nodules"]),
            "findings": confirmed}


def save_draft(db: Database, patient_id: str, study_uid: str, body: dict[str, Any], drafted_by: str,
               run_id: str = "") -> str:
    report_id = f"rep_{uuid.uuid4().hex[:10]}"
    db.execute("INSERT INTO reports(report_id,patient_id,study_uid,status,body,drafted_by,created,run_id) "
               "VALUES (?,?,?,?,?,?,?,?)",
               (report_id, patient_id, study_uid, "draft", dumps(body), drafted_by, time.time(), run_id))
    return report_id


def sign_report(db: Database, report_id: str, signer: str) -> dict[str, Any]:
    row = db.one("SELECT * FROM reports WHERE report_id=?", (report_id,))
    if row is None:
        raise KeyError(report_id)
    if row["status"] != "signed":
        # A newer signed report replaces older ones for the same study.
        db.execute("UPDATE reports SET status='superseded' WHERE patient_id=? AND status='signed'", (row["patient_id"],))
        db.execute("UPDATE reports SET status='signed', signed_by=?, signed=? WHERE report_id=?",
                   (signer, time.time(), report_id))
    return {"report_id": report_id, "status": "signed", "signed_by": signer}


def get_report(db: Database, patient_id: str, signed_only: bool = True) -> dict[str, Any] | None:
    cond = "status='signed'" if signed_only else "status IN ('signed','draft')"
    row = db.one(f"SELECT * FROM reports WHERE patient_id=? AND {cond} ORDER BY "
                 "CASE status WHEN 'signed' THEN 0 ELSE 1 END, created DESC LIMIT 1", (patient_id,))
    return _row(row) if row else None


def get_report_by_id(db: Database, report_id: str) -> dict[str, Any] | None:
    row = db.one("SELECT * FROM reports WHERE report_id=?", (report_id,))
    return _row(row) if row else None


def list_reports(db: Database) -> list[dict[str, Any]]:
    return [_row(r, with_body=False) for r in db.query(
        "SELECT * FROM reports WHERE status IN ('draft','signed') ORDER BY created DESC")]


def _row(row: dict[str, Any], with_body: bool = True) -> dict[str, Any]:
    out = {k: row[k] for k in ("report_id", "patient_id", "study_uid", "status", "drafted_by", "signed_by",
                               "created", "signed", "run_id")}
    if with_body:
        out["body"] = json.loads(row["body"])
    return out

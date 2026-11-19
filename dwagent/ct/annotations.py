"""LIDC-IDRI reader annotations (up to four thoracic radiologists per scan), read from the
SQLite database shipped with pylidc. Annotations of the same physical nodule by different
readers are clustered, so the result is one record per nodule with reader agreement."""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config import get_settings
from .geometry import axes_mm, laterality, mean_diameter, zone
from .store import load_volume

BENIGN_CALCIFICATION = {1: "popcorn", 2: "laminated", 3: "solid", 5: "central"}  # LIDC codes; 6 = absent


def _con() -> sqlite3.Connection:
    path = get_settings().lidc_dir / "_annotations" / "pylidc.sqlite"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _polygon_mask(points_rc: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    img = Image.new("1", (shape[1], shape[0]), 0)
    xy = [(c, r) for r, c in points_rc]
    if len(xy) >= 3:
        ImageDraw.Draw(img).polygon(xy, outline=1, fill=1)
    else:
        for x, y in xy:
            img.putpixel((int(x), int(y)), 1)
    return np.array(img, dtype=bool)


@lru_cache(maxsize=16)
def reader_nodules(patient_id: str) -> list[dict[str, Any]]:
    con = _con()
    scan = con.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY id LIMIT 1", (patient_id,)).fetchone()
    if scan is None:
        return []
    vol = load_volume(patient_id)
    dz, dy, dx = vol.spacing
    from .detector import lung_mask
    zs = np.nonzero(lung_mask(vol).any(axis=(1, 2)))[0]
    lung_z = (float(vol.z_mm[zs[0]]), float(vol.z_mm[zs[-1]])) if len(zs) else (0.0, 0.0)
    anns = []
    for a in con.execute("SELECT * FROM annotations WHERE scan_id=?", (scan["id"],)):
        best = None
        pts_all = []
        for c in con.execute("SELECT * FROM contours WHERE annotation_id=? AND inclusion=1", (a["id"],)):
            # pylidc stores each contour point as "x,y" (column, row). Checked on the 10 local
            # studies: mean HU at contour centres is -168 read this way and -406 the other way.
            xy = [tuple(map(float, ln.split(","))) for ln in c["coords"].strip().splitlines() if ln.strip()]
            pts = [(y, x) for x, y in xy]
            k = int(np.argmin(np.abs(vol.z_mm - c["image_z_position"])))
            mask = _polygon_mask(pts, vol.shape[1:])
            area = int(mask.sum())
            pts_all += [(c["image_z_position"], r * dy, col * dx) for r, col in pts]
            if best is None or area > best[0]:
                best = (area, k, mask, pts)
        if best is None:
            continue
        _, k, mask, pts = best
        long_mm, short_mm = axes_mm(mask, dy, dx)
        solid = mask & (vol.hu[k] > -300)
        s_long, s_short = axes_mm(solid, dy, dx)
        centroid = np.mean(np.array(pts_all), axis=0)
        anns.append({"annotation_id": a["id"], "centroid_mm": centroid, "slice_index": k,
                     "long_mm": long_mm, "short_mm": short_mm, "solid_mean_mm": mean_diameter(s_long, s_short),
                     "center_px": (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))),
                     **{f: a[f] for f in ("subtlety", "calcification", "margin", "lobulation", "spiculation",
                                          "texture", "malignancy")}})
    # Cluster annotations whose centroids are closer than the larger lesion's radius (min 5 mm).
    parent = list(range(len(anns)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            d = float(np.linalg.norm(anns[i]["centroid_mm"] - anns[j]["centroid_mm"]))
            if d <= max(5.0, 0.5 * max(anns[i]["long_mm"], anns[j]["long_mm"])):
                parent[find(i)] = find(j)
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, a in enumerate(anns):
        groups.setdefault(find(i), []).append(a)
    nodules = []
    for members in groups.values():
        long_mm = float(np.mean([m["long_mm"] for m in members]))
        short_mm = float(np.mean([m["short_mm"] for m in members]))
        texture = median(m["texture"] for m in members)
        calc = int(median(m["calcification"] for m in members))
        centroid = np.mean([m["centroid_mm"] for m in members], axis=0)
        ref = max(members, key=lambda m: m["long_mm"])
        ntype = "ground_glass" if texture <= 2 else "part_solid" if texture < 4 else "solid"
        nodules.append({
            "n_readers": len(members), "slice_index": ref["slice_index"],
            "center_px": [round(ref["center_px"][0], 1), round(ref["center_px"][1], 1)],
            "center_mm": [round(float(v), 1) for v in centroid],
            "long_axis_mm": round(long_mm, 1), "short_axis_mm": round(short_mm, 1),
            "mean_diameter_mm": mean_diameter(long_mm, short_mm),
            "nodule_type": ntype,
            "solid_component_mm": round(float(np.mean([m["solid_mean_mm"] for m in members])), 1)
            if ntype == "part_solid" else None,
            "benign_calcification": calc in BENIGN_CALCIFICATION,
            "calcification_pattern": BENIGN_CALCIFICATION.get(calc, "absent" if calc == 6 else "non-central"),
            "spiculated": median(m["spiculation"] for m in members) >= 4,
            "malignancy_rating_median": float(median(m["malignancy"] for m in members)),
            "laterality": laterality(ref["center_px"][1], vol.shape[2]),
            "zone": zone(float(centroid[0]), *lung_z),
        })
    nodules.sort(key=lambda n: -n["mean_diameter_mm"])
    for i, n in enumerate(nodules, start=1):
        n["nodule_id"] = f"R{i}"
    return nodules


def match(candidates: list[dict[str, Any]], nodules: list[dict[str, Any]]) -> dict[str, Any]:
    """Match detector candidates to reader nodules by centre distance (within the nodule radius,
    minimum 4 mm). Returns pairs plus unmatched items on both sides."""
    pairs, used = [], set()
    for n in nodules:
        best, best_d = None, 1e9
        for c in candidates:
            if c["finding_id"] in used:
                continue
            d = float(np.linalg.norm(np.array(c["center_mm"]) - np.array(n["center_mm"])))
            if d < best_d:
                best, best_d = c, d
        if best is not None and best_d <= max(4.0, 0.5 * n["long_axis_mm"]):
            used.add(best["finding_id"])
            pairs.append({"nodule_id": n["nodule_id"], "finding_id": best["finding_id"], "distance_mm": round(best_d, 1)})
    matched_nodules = {p["nodule_id"] for p in pairs}
    return {"pairs": pairs,
            "missed_nodules": [n["nodule_id"] for n in nodules if n["nodule_id"] not in matched_nodules],
            "false_positives": [c["finding_id"] for c in candidates if c["finding_id"] not in used]}


@lru_cache(maxsize=1)
def _annotation_extent_px() -> dict[int, float]:
    """Largest in-plane contour extent of every annotation, in pixels. One pass over the
    contour table (41,406 rows); the table has no index on annotation_id, so per-annotation
    queries took minutes."""
    out: dict[int, float] = {}
    for aid, coords in _con().execute("SELECT annotation_id, coords FROM contours WHERE inclusion=1"):
        pts = np.array([list(map(float, ln.split(","))) for ln in coords.strip().splitlines() if ln.strip()])
        if len(pts) > 1:
            out[aid] = max(out.get(aid, 0.0), float(np.max(np.ptp(pts, axis=0))))
    return out


def cohort_query(min_diameter_mm: float | None = None, max_diameter_mm: float | None = None,
                 max_slice_thickness_mm: float | None = None, contrast: bool | None = None,
                 min_readers: int = 1, texture: str | None = None, min_malignancy: int | None = None) -> dict[str, Any]:
    """Count scans in the full LIDC-IDRI annotation database (1,018 scans) that match a
    structured cohort definition. Uses annotation-level approximations: an annotation's size is
    the largest in-plane contour extent, and reader count is annotations per scan."""
    con = _con()
    where, params = ["1=1"], []
    if max_slice_thickness_mm is not None:
        where.append("s.slice_thickness <= ?"); params.append(max_slice_thickness_mm)
    if contrast is not None:
        where.append("s.contrast_used = ?"); params.append(int(contrast))
    a_where = ["1=1"]
    if texture == "solid":
        a_where.append("a.texture >= 4")
    elif texture == "part_solid":
        a_where.append("a.texture = 3")
    elif texture == "ground_glass":
        a_where.append("a.texture <= 2")
    if min_malignancy is not None:
        a_where.append("a.malignancy >= ?"); params.append(min_malignancy)
    sql = (f"SELECT s.id, s.patient_id, s.pixel_spacing, a.id AS aid FROM scans s "
           f"JOIN annotations a ON a.scan_id = s.id WHERE {' AND '.join(where)} AND {' AND '.join(a_where)}")
    rows = con.execute(sql, params).fetchall()
    scans: dict[int, dict[str, Any]] = {}
    need_size = min_diameter_mm is not None or max_diameter_mm is not None
    extents = _annotation_extent_px() if need_size else {}
    for r in rows:
        if need_size:
            size = extents.get(r["aid"], 0.0) * r["pixel_spacing"]
            if min_diameter_mm is not None and size < min_diameter_mm:
                continue
            if max_diameter_mm is not None and size > max_diameter_mm:
                continue
        s = scans.setdefault(r["id"], {"patient_id": r["patient_id"], "annotations": 0})
        s["annotations"] += 1
    hits = {k: v for k, v in scans.items() if v["annotations"] >= min_readers}
    total = con.execute("SELECT count(*) FROM scans").fetchone()[0]
    return {"total_scans_in_database": total, "matching_scans": len(hits),
            "matching_patients": len({v["patient_id"] for v in hits.values()}),
            "matching_annotations": sum(v["annotations"] for v in hits.values()),
            "example_patient_ids": sorted({v["patient_id"] for v in hits.values()})[:8],
            "definition": {"min_diameter_mm": min_diameter_mm, "max_diameter_mm": max_diameter_mm,
                           "max_slice_thickness_mm": max_slice_thickness_mm, "contrast": contrast,
                           "min_annotations_per_scan": min_readers, "texture": texture, "min_malignancy": min_malignancy}}

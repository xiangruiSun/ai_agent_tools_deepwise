"""classical_v1: a transparent, rule-based lung nodule candidate detector.

This is NOT a Deepwise model and is not a medical device. It stands where a registered
detection model would sit so the agents can be exercised against a real algorithm with real
latency and real false positives. It segments the lungs by thresholding, then keeps compact
dense objects inside them. It cannot see pure ground-glass nodules or nodules fused to vessels.
Its measured sensitivity against the LIDC readers is reported in verification/ct_detector.md.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from scipy import ndimage

from ..config import get_settings
from .geometry import axes_mm, laterality, mean_diameter, zone
from .store import Volume, load_volume

DETECTOR_VERSION = "classical_v1.1"


def lung_mask(vol: Volume) -> np.ndarray:
    air = vol.hu < -400
    labels, n = ndimage.label(air)
    if n == 0:
        return np.zeros_like(air)
    # Air touching the in-plane border of any slice is outside the body.
    border = np.unique(np.concatenate([labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
                                       labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    sizes = ndimage.sum(air, labels, index=np.arange(1, n + 1))
    voxel_ml = np.prod(vol.spacing) / 1000.0
    keep = [i + 1 for i in np.argsort(sizes)[::-1][:6]
            if (i + 1) not in border and sizes[i] * voxel_ml > 200]  # lungs are well over 200 ml
    mask = np.isin(labels, keep[:2])
    # Close each slice so juxta-pleural and vessel-adjacent tissue is inside the lung region.
    r = max(2, int(round(6.0 / vol.spacing[1])))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (yy * yy + xx * xx) <= r * r
    out = np.zeros_like(mask)
    for k in range(mask.shape[0]):
        if mask[k].any():
            out[k] = ndimage.binary_fill_holes(ndimage.binary_closing(mask[k], structure=disk, border_value=0))
    return out


def detect(patient_id: str, min_mm: float = 3.0, max_mm: float = 35.0, max_candidates: int = 12,
           use_cache: bool = True) -> dict[str, Any]:
    cache = get_settings().runtime_dir / "detector"
    cache.mkdir(exist_ok=True)
    path = cache / f"{patient_id}.{DETECTOR_VERSION}.json"
    if use_cache and path.exists():
        out = json.loads(path.read_text())
        out["from_cache"] = True
        return out
    t0 = time.perf_counter()
    vol = load_volume(patient_id)
    dz, dy, dx = vol.spacing
    lungs = lung_mask(vol)
    zs = np.nonzero(lungs.any(axis=(1, 2)))[0]
    lung_z = (float(vol.z_mm[zs[0]]), float(vol.z_mm[zs[-1]])) if len(zs) else (0.0, 0.0)
    dense = lungs & (vol.hu > -450)
    voxel = dz * dy * dx
    cands: list[dict[str, Any]] = []

    def collect(mask: np.ndarray, d_min: float, d_max: float, max_elong: float, min_fill: float, source: str) -> None:
        labels, _ = ndimage.label(mask)
        for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            comp = labels[sl] == idx
            nvox = int(comp.sum())
            d_eq = (6 * nvox * voxel / np.pi) ** (1 / 3)
            if not (d_min <= d_eq <= d_max):
                continue
            ext = np.array([(sl[0].stop - sl[0].start) * dz, (sl[1].stop - sl[1].start) * dy,
                            (sl[2].stop - sl[2].start) * dx])
            elong = float(ext.max() / d_eq)
            fill = float(nvox * voxel / np.prod(ext))
            if elong > max_elong or fill < min_fill:   # vessels are long and fill little of their box
                continue
            k_rel = int(np.argmax(comp.sum(axis=(1, 2))))
            long_mm, short_mm = axes_mm(comp[k_rel], dy, dx)
            cz, cy, cx = ndimage.center_of_mass(comp)
            k = sl[0].start + k_rel
            center = [float(vol.z_mm[min(sl[0].start + int(round(cz)), len(vol.z_mm) - 1)]),
                      round(float((sl[1].start + cy) * dy), 1), round(float((sl[2].start + cx) * dx), 1)]
            if any(np.linalg.norm(np.array(center) - np.array(c["center_mm"])) < 5.0 for c in cands):
                continue                                # already found by the first pass
            hu_vals = vol.hu[sl][comp]
            solid_frac = float((hu_vals > -300).mean())
            n_slices = sl[0].stop - sl[0].start
            cands.append({
                "slice_index": int(k), "z_mm": float(vol.z_mm[k]),
                "center_px": [round(float(sl[1].start + cy), 1), round(float(sl[2].start + cx), 1)],
                "center_mm": center,
                "long_axis_mm": round(long_mm, 1), "short_axis_mm": round(short_mm, 1),
                "mean_diameter_mm": mean_diameter(long_mm, short_mm),
                "volume_mm3": round(nvox * voxel, 1), "mean_hu": round(float(hu_vals.mean()), 0),
                "nodule_type": "solid" if solid_frac >= 0.6 else "part_solid",
                "solid_component_mm": None if solid_frac >= 0.6 else round(mean_diameter(long_mm, short_mm) * solid_frac ** 0.5, 1),
                "solid_fraction": round(solid_frac, 2),
                "laterality": laterality(sl[2].start + cx, vol.shape[2]),
                "zone": zone(float(vol.z_mm[k]), *lung_z),
                # Compact, larger objects seen on at least two slices rank first.
                "score": round(fill / elong * d_eq * (1.0 if n_slices >= 2 else 0.3), 3),
                "pass": source,
            })

    # Pass A: compact dense objects as they are (small and medium nodules).
    collect(dense, min_mm, max_mm, 1.9, 0.28, "A")
    # Pass B: open each slice first so large masses detach from the vessels they touch.
    r = 3
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (yy * yy + xx * xx) <= r * r
    opened = np.zeros_like(dense)
    for k in range(dense.shape[0]):
        if dense[k].any():
            opened[k] = ndimage.binary_opening(dense[k], structure=disk)
    collect(opened, 9.0, 45.0, 2.2, 0.2, "B")
    cands.sort(key=lambda c: (-c["score"], -c["mean_diameter_mm"]))
    total = len(cands)
    cands = cands[:max_candidates]
    for i, c in enumerate(cands, start=1):
        c["finding_id"] = f"D{i}"
    out = {"patient_id": patient_id, "detector": DETECTOR_VERSION, "n_candidates": len(cands),
           "n_before_cap": total, "lung_z_mm": lung_z, "candidates": cands,
           "runtime_s": round(time.perf_counter() - t0, 2), "from_cache": False}
    path.write_text(json.dumps(out))
    return out

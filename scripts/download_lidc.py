"""Download a small, deterministic subset of LIDC-IDRI chest CT studies.

Source: The Cancer Imaging Archive (TCIA), collection LIDC-IDRI, CC BY 3.0.
Annotations: the SQLite database shipped inside the MIT-licensed ``pylidc`` package
(the four-reader XML annotations of LIDC-IDRI, already parsed).

The subset is chosen from the annotation database so that the agents meet every
branch of the nodule workflow: scans without any annotated nodule >= 3 mm, and scans
whose largest nodule falls in each size band used by Lung-RADS.

Usage:
    python scripts/download_lidc.py              # 10 studies (about 1.5 GB)
    python scripts/download_lidc.py --per-band 1 # 5 studies
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIDC_DIR = ROOT / "data" / "lidc"
ANN_DB = LIDC_DIR / "_annotations" / "pylidc.sqlite"
NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
PYLIDC_JSON = "https://pypi.org/pypi/pylidc/json"

# Size bands (mm, largest annotated nodule on the scan). "none" = no nodule >= 3 mm annotated.
BANDS = [("none", None, None), ("lt6", 0.0, 6.0), ("6to8", 6.0, 8.0), ("8to15", 8.0, 15.0), ("ge15", 15.0, 1e9)]
MAX_SLICES = 160  # keeps each study small enough to download and to process on a laptop


def ensure_annotation_db() -> None:
    if ANN_DB.exists():
        return
    ANN_DB.parent.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(PYLIDC_JSON, timeout=60))
    url = next(u["url"] for u in meta["urls"] if u["filename"].endswith(".whl"))
    blob = urllib.request.urlopen(url, timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".sqlite"))
        ANN_DB.write_bytes(zf.read(name))
    print(f"annotation database -> {ANN_DB}")


def contour_diameter_mm(coords: str, spacing: float) -> float:
    pts = [tuple(map(float, line.split(","))) for line in coords.strip().splitlines() if line.strip()]
    best = 0.0
    for i, (x1, y1) in enumerate(pts):
        for x2, y2 in pts[i + 1:]:
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if d > best:
                best = d
    return best * spacing


def scan_summaries(con: sqlite3.Connection) -> list[dict]:
    out = []
    for sid, study, series, pid, thick, spacing, contrast, names in con.execute(
        "select id, study_instance_uid, series_instance_uid, patient_id, slice_thickness, pixel_spacing,"
        " contrast_used, sorted_dicom_file_names from scans order by patient_id, id"
    ):
        n_slices = len(names.split(","))
        largest = 0.0
        n_ann = 0
        for (aid,) in con.execute("select id from annotations where scan_id=?", (sid,)):
            n_ann += 1
            for (coords,) in con.execute("select coords from contours where annotation_id=? and inclusion=1", (aid,)):
                largest = max(largest, contour_diameter_mm(coords, spacing))
        out.append(dict(scan_id=sid, study_uid=study, series_uid=series, patient_id=pid, slice_thickness=thick,
                        pixel_spacing=spacing, contrast_used=bool(contrast), n_slices=n_slices,
                        n_annotations=n_ann, largest_mm=round(largest, 1)))
    return out


def choose(summaries: list[dict], per_band: int) -> list[dict]:
    seen_patients: set[str] = set()
    picked = []
    eligible = [s for s in summaries if s["n_slices"] <= MAX_SLICES and s["slice_thickness"] <= 2.5]
    for band, lo, hi in BANDS:
        n = 0
        for s in eligible:
            if s["patient_id"] in seen_patients:
                continue
            if band == "none":
                ok = s["n_annotations"] == 0
            else:
                ok = s["n_annotations"] >= 3 and lo <= s["largest_mm"] < hi
            if ok:
                picked.append({**s, "band": band})
                seen_patients.add(s["patient_id"])
                n += 1
                if n == per_band:
                    break
    return picked


def download_series(series_uid: str, dest: Path, retries: int = 3) -> int:
    if dest.exists() and any(dest.glob("*.dcm")):
        return len(list(dest.glob("*.dcm")))
    url = f"{NBIA}/getImage?SeriesInstanceUID={series_uid}"
    last = None
    for attempt in range(1, retries + 1):
        try:
            blob = urllib.request.urlopen(url, timeout=900).read()
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                dest.mkdir(parents=True, exist_ok=True)
                n = 0
                for name in zf.namelist():
                    if name.lower().endswith(".dcm"):
                        (dest / Path(name).name).write_bytes(zf.read(name))
                        n += 1
            return n
        except Exception as exc:  # network errors are retried with backoff
            last = exc
            time.sleep(5 * attempt)
    raise RuntimeError(f"download failed for {series_uid}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=2)
    args = ap.parse_args()
    ensure_annotation_db()
    con = sqlite3.connect(ANN_DB)
    picked = choose(scan_summaries(con), args.per_band)
    manifest = []
    for s in picked:
        dest = LIDC_DIR / s["patient_id"] / s["series_uid"]
        t0 = time.time()
        n = download_series(s["series_uid"], dest)
        size = sum(f.stat().st_size for f in dest.glob("*.dcm"))
        print(f"{s['patient_id']} band={s['band']:5s} largest={s['largest_mm']:5.1f} mm files={n} "
              f"{size / 1e6:6.1f} MB {time.time() - t0:5.1f}s", flush=True)
        manifest.append({**s, "n_files": n, "bytes": size, "path": str(dest.relative_to(ROOT)).replace("\\", "/")})
    (LIDC_DIR / "manifest.json").write_text(json.dumps({
        "source": "TCIA LIDC-IDRI", "license": "CC BY 3.0",
        "citation": "Armato SG III et al. Data From LIDC-IDRI. The Cancer Imaging Archive. doi:10.7937/K9/TCIA.2015.LO9QL9SX",
        "annotation_source": "pylidc 0.2.3 (MIT) pylidc.sqlite", "studies": manifest}, indent=2))
    print(f"{len(manifest)} studies, {sum(m['bytes'] for m in manifest) / 1e9:.2f} GB -> {LIDC_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

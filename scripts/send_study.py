"""Act as the CT scanner: send a study's DICOM images to the gateway by C-STORE.

    python scripts/send_study.py LIDC-IDRI-0005 [--limit 40] [--port 11112] [--ae CT_SCANNER_1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dwagent.ct import store  # noqa: E402
from dwagent.gateway.scp import send_series  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("patient_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=11112)
    ap.add_argument("--ae", default="CT_SCANNER_1")
    a = ap.parse_args()
    files = sorted(store.series_dir(a.patient_id).glob("*.dcm"))
    print(send_series(files[:a.limit] if a.limit else files, a.port, a.ae))

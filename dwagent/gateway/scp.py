"""DICOM gateway: the door a hospital CT scanner or PACS uses to hand a study to the AI server.

A DICOM Storage SCP (C-STORE) built on pynetdicom. Received CT images are written under
data/runtime/incoming/<PatientID>/<SeriesInstanceUID>/ and, when the scanner closes the
association, one "study arrived" record is added to the worklist. Nothing is sent anywhere:
in the on-prem profile this process only listens on the hospital LAN.

Run:  python -m dwagent.gateway.scp --port 11112
Send: python scripts/send_study.py LIDC-IDRI-0005 --limit 20
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from pynetdicom import AE, evt
from pynetdicom.sop_class import CTImageStorage, Verification

from ..config import get_settings
from ..store import Database, get_db

AE_TITLE = "DEEPWISE_AI"
ALLOWED_CALLING_AE = {"CT_SCANNER_1", "HOSPITAL_PACS"}   # association allowlist


class Gateway:
    def __init__(self, port: int = 11112, db: Database | None = None, incoming: Path | None = None,
                 on_study: Callable[[dict[str, Any]], None] | None = None):
        self.port, self.db = port, db or get_db()
        self.incoming = incoming or (get_settings().runtime_dir / "incoming")
        self.on_study = on_study
        self._assoc: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._server = None
        self.rejected = 0

    def _rejected(self, event: Any) -> None:
        self.rejected += 1             # unknown sender: refused before any data is accepted
        self.db.audit("unknown-device", "device", "dicom:association", "-", False, "calling AE not in allowlist")

    def _store(self, event: Any) -> int:
        ds = event.dataset
        ds.file_meta = event.file_meta
        if getattr(ds, "Modality", "") != "CT":
            return 0xA900              # data set does not match SOP class
        dest = self.incoming / str(ds.PatientID) / str(ds.SeriesInstanceUID)
        dest.mkdir(parents=True, exist_ok=True)
        ds.save_as(dest / f"{ds.SOPInstanceUID}.dcm", enforce_file_format=True)
        with self._lock:
            a = self._assoc.setdefault(id(event.assoc), {"patient_id": str(ds.PatientID),
                                                         "series_uid": str(ds.SeriesInstanceUID),
                                                         "study_uid": str(ds.StudyInstanceUID), "files": 0,
                                                         "calling_ae": event.assoc.requestor.ae_title.strip(),
                                                         "started": time.time()})
            a["files"] += 1
        return 0x0000

    def _released(self, event: Any) -> None:
        with self._lock:
            a = self._assoc.pop(id(event.assoc), None)
        if not a:
            return
        a["finished"] = time.time()
        self.db.execute("INSERT OR REPLACE INTO kv(k,v) VALUES (?,?)",
                        (f"arrival:{a['series_uid']}", json.dumps(a)))
        self.db.audit(a["calling_ae"], "device", "dicom:c-store", a["series_uid"], True, f"{a['files']} files")
        if self.on_study:
            self.on_study(a)

    def start(self) -> "Gateway":
        ae = AE(ae_title=AE_TITLE)
        ae.require_calling_aet = sorted(ALLOWED_CALLING_AE)   # the library rejects every other sender
        ae.add_supported_context(CTImageStorage)
        ae.add_supported_context(Verification)
        handlers = [(evt.EVT_REJECTED, self._rejected), (evt.EVT_C_STORE, self._store),
                    (evt.EVT_RELEASED, self._released)]
        self._server = ae.start_server(("127.0.0.1", self.port), block=False, evt_handlers=handlers)
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def arrivals(self) -> list[dict[str, Any]]:
        return [json.loads(r["v"]) for r in self.db.query("SELECT v FROM kv WHERE k LIKE 'arrival:%'")]


def send_series(files: list[Path], port: int = 11112, calling_ae: str = "CT_SCANNER_1") -> dict[str, Any]:
    """Storage SCU: what the scanner does. Returns how many images the gateway accepted."""
    import pydicom
    ae = AE(ae_title=calling_ae)
    ae.add_requested_context(CTImageStorage)
    assoc = ae.associate("127.0.0.1", port, ae_title=AE_TITLE)
    if not assoc.is_established:
        return {"established": False, "sent": 0, "accepted": 0}
    accepted = 0
    for f in files:
        status = assoc.send_c_store(pydicom.dcmread(f))
        if status and status.Status == 0x0000:
            accepted += 1
    assoc.release()
    return {"established": True, "sent": len(files), "accepted": accepted}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11112)
    args = ap.parse_args()
    gw = Gateway(args.port, on_study=lambda a: print("study arrived:", a, flush=True)).start()
    print(f"DICOM gateway {AE_TITLE} listening on 127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        gw.stop()

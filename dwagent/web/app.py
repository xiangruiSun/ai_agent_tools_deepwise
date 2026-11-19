"""Web portal API. One FastAPI process serves the single-page app and the role-scoped API.

Access model
    login          demo accounts, PBKDF2 password hashes, signed session cookie (httpOnly, SameSite)
    role           patient | doctor | employee, fixed at login, never taken from the request body
    patient bind   a patient session carries its own patient id; every patient route uses that
                   id and nothing the browser sends
    tools          the agent for a role only sees that role's tools, and the tool executor checks
                   scopes and record-level access again on every call
    audit          logins, denied requests and every tool call are written to the audit table
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jwt
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .. import clinical
from ..catalog import get_registry, new_context
from ..chat import chat_turn
from ..config import ROOT, get_settings
from ..ct import store as ct_store
from ..llm.ollama import get_llm
from ..memory.stores import LongTermMemory
from ..pipelines import ct_study
from ..profiles import AGENTS, DEMO_USERS, principal
from ..runtime.graph import Checkpointer, Env
from ..runtime.trace import Tracer
from ..skills.patient_explain import explain_report
from ..store import get_db
from ..tools.registry import Principal

STATIC = Path(__file__).parent / "static"
COOKIE = "dw_session"
SESSION_HOURS = 8
app = FastAPI(title="Deepwise agent tools demo", docs_url="/api/docs", openapi_url="/api/openapi.json")
pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="run")
JOBS: dict[str, dict[str, Any]] = {}
_login_fail: dict[str, list[float]] = {}
_lock = threading.Lock()


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)


_SALT = secrets.token_bytes(16)
USERS = {u: {"hash": _hash(pw, _SALT), "role": role, "patient_id": pid, "name": name}
         for u, pw, role, pid, name in DEMO_USERS}


def current(request: Request) -> Principal:
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "not signed in")
    try:
        claims = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "session expired") from None
    u = USERS.get(claims["sub"])
    if not u:
        raise HTTPException(401, "unknown user")
    return principal(claims["sub"], u["role"], u["patient_id"], u["name"])   # role comes from the server-side table


def require(*roles: str):
    def dep(request: Request, who: Principal = Depends(current)) -> Principal:
        if who.role not in roles:
            get_db().audit(who.user_id, who.role, f"http:{request.method} {request.url.path}", "-", False, "role not allowed")
            raise HTTPException(403, "your role cannot use this")
        return who
    return dep


class Login(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


@app.post("/api/login")
def login(body: Login, response: Response, request: Request) -> dict[str, Any]:
    ip = request.client.host if request.client else "?"
    now = time.time()
    with _lock:
        recent = [t for t in _login_fail.get(ip, []) if now - t < 300]
        _login_fail[ip] = recent
        if len(recent) >= 8:
            raise HTTPException(429, "too many attempts, wait a few minutes")
    u = USERS.get(body.username)
    ok = bool(u) and hmac.compare_digest(u["hash"], _hash(body.password, _SALT))
    get_db().audit(body.username[:40], u["role"] if u else "-", "login", "-", ok)
    if not ok:
        with _lock:
            _login_fail.setdefault(ip, []).append(now)
        raise HTTPException(401, "wrong username or password")
    token = jwt.encode({"sub": body.username, "exp": int(now + SESSION_HOURS * 3600)}, get_settings().secret_key, "HS256")
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=SESSION_HOURS * 3600,
                        secure=request.url.scheme == "https")
    return me(principal(body.username, u["role"], u["patient_id"], u["name"]))


@app.post("/api/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(who: Principal = Depends(current)) -> dict[str, Any]:
    cfg = AGENTS[who.role]
    tools = get_registry().allowed_for(who, [*(cfg.tool_names or []), "remember"])
    return {"user_id": who.user_id, "name": who.display_name, "role": who.role, "patient_id": who.patient_id,
            "agent": cfg.name, "scopes": sorted(who.scopes),
            "limits": {"max_iterations": cfg.max_iterations, "token_budget": cfg.token_budget},
            "tools": [{"name": t.name, "description": t.description, "kind": "skill" if t.name == "assess_nodules" else "tool",
                       "writes": t.side_effect, "needs_approval": t.requires_approval} for t in tools]}


@app.get("/api/demo-accounts")
def demo_accounts() -> list[dict[str, str]]:
    """The demo credentials are public on purpose: this site holds only public research data."""
    return [{"username": u, "password": pw, "role": role, "name": name} for u, pw, role, _, name in DEMO_USERS]


# ----------------------------------------------------------------- agent chat (all roles)
class ChatIn(BaseModel):
    session_id: str = Field(max_length=64)
    message: str = Field(min_length=1, max_length=2000)


def _job(run_id: str, owner: str, fn: Any) -> None:
    JOBS[run_id] = {"owner": owner, "status": "running", "result": None}

    def work() -> None:
        try:
            JOBS[run_id].update(status="done", result=fn())
        except Exception as exc:  # surfaced to the UI instead of a hung spinner
            Tracer(get_db()).end_run(run_id, "failed", str(exc))
            JOBS[run_id].update(status="error", result={"error": f"{type(exc).__name__}: {exc}"[:300]})
    pool.submit(work)


@app.post("/api/chat")
def chat(body: ChatIn, who: Principal = Depends(current)) -> dict[str, str]:
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    _job(run_id, who.user_id, lambda: chat_turn(who, body.session_id, body.message, run_id=run_id))
    return {"run_id": run_id}


def _events(run_id: str, who: Principal, own: bool) -> list[dict[str, Any]]:
    tr = Tracer(get_db())
    if not own:                       # staff looking at someone else's run: timings only, no payloads
        return tr.redacted_events(run_id)
    out = []
    for e in tr.events(run_id):
        out.append({k: e[k] for k in ("id", "kind", "name", "latency_ms", "prompt_tokens", "completion_tokens", "ok",
                                      "error_kind", "error", "attempts")} | {"input": e["input"][:600], "output": e["output"][:900]})
    return out


@app.get("/api/runs/{run_id}")
def run_status(run_id: str, who: Principal = Depends(current)) -> dict[str, Any]:
    db = get_db()
    run = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    job = JOBS.get(run_id)
    if run is None and job is None:
        raise HTTPException(404, "no such run")
    own = bool(run and run["user_id"] == who.user_id) or bool(job and job["owner"] == who.user_id)
    if not own and who.role != "employee":
        db.audit(who.user_id, who.role, "http:read run", run_id, False, "not the owner")
        raise HTTPException(403, "this run belongs to someone else")
    tr = Tracer(db)
    return {"run_id": run_id, "job": job["status"] if job else "done", "result": (job or {}).get("result") if own else None,
            "summary": tr.summary(run_id) if run else None, "events": _events(run_id, who, own) if run else []}


@app.get("/api/memory")
def memory(who: Principal = Depends(current)) -> list[dict[str, Any]]:
    return LongTermMemory(get_db()).all(who.user_id)


@app.delete("/api/memory/{memory_id}")
def forget(memory_id: int, who: Principal = Depends(current)) -> dict[str, bool]:
    return {"deleted": LongTermMemory(get_db()).forget(who.user_id, memory_id)}


# ----------------------------------------------------------------- patient
@app.get("/api/my/report")
def my_report(who: Principal = Depends(require("patient"))) -> dict[str, Any]:
    rep = clinical.get_report(get_db(), who.patient_id or "", signed_only=True)
    if rep is None:
        return {"available": False}
    key = [{"finding_id": f["finding_id"], "slice_index": f["slice_index"]} for f in rep["body"]["findings"]]
    return {"available": True, "report": rep["body"], "signed_by": rep["signed_by"], "signed": rep["signed"],
            "patient_id": who.patient_id, "key_slices": key}


@app.post("/api/my/explanation")
def my_explanation(who: Principal = Depends(require("patient"))) -> dict[str, str]:
    rep = clinical.get_report(get_db(), who.patient_id or "", signed_only=True)
    if rep is None:
        raise HTTPException(404, "no signed report yet")
    tr = Tracer(get_db())
    run_id = tr.start_run("patient_explanation", "explain my report", who.user_id, who.role)

    def work() -> dict[str, Any]:
        res = explain_report(get_llm(), rep["body"], ctx=new_context(who, run_id))
        tr.end_run(run_id, "completed" if res.ok else "failed")
        return {"valid": res.ok, "attempts": [{"n": a.number, "stage": a.stage, "errors": a.errors[:3]} for a in res.attempts],
                "explanation": res.value.model_dump() if res.value else None}
    _job(run_id, who.user_id, work)
    return {"run_id": run_id}


# ----------------------------------------------------------------- images (doctor: any study; patient: own)
def _may_view(who: Principal, patient_id: str) -> None:
    if who.role == "doctor" or (who.role == "patient" and who.patient_id == patient_id):
        return
    get_db().audit(who.user_id, who.role, "http:image", patient_id, False, "not allowed")
    raise HTTPException(403, "you cannot view this study")


@app.get("/api/studies/{patient_id}/slice/{index}.png")
def slice_png(patient_id: str, index: int, marks: int = 1, who: Principal = Depends(current)) -> Response:
    _may_view(who, patient_id)
    try:
        vol = ct_store.load_volume(patient_id)
    except (ct_store.StudyNotFound, FileNotFoundError):
        raise HTTPException(404, "no such study") from None
    k = max(0, min(index, vol.shape[0] - 1))
    lo, hi = -1350.0, 150.0                         # lung window
    img = np.clip((vol.hu[k].astype(np.float32) - lo) / (hi - lo), 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    if marks:
        d = ImageDraw.Draw(im)
        for f in clinical.review_findings(patient_id)["findings"]:
            if abs(f["slice_index"] - k) <= 1:
                r, c = f["center_px"]
                rad = max(9, f["mean_diameter_mm"] / vol.spacing[1] * 0.9)
                d.ellipse([c - rad, r - rad, c + rad, r + rad], outline=(0, 200, 170), width=2)
                d.text((c + rad + 3, r - 6), f["finding_id"], fill=(0, 200, 170))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------- doctor
@app.get("/api/worklist")
def worklist(who: Principal = Depends(require("doctor"))) -> list[dict[str, Any]]:
    db = get_db()
    out = []
    for s in ct_store.list_studies():
        rep = clinical.get_report(db, s["patient_id"], signed_only=False)
        out.append({**s, "report_status": rep["status"] if rep else "none", "report_id": rep["report_id"] if rep else None,
                    "exam_lung_rads": rep["body"]["exam_lung_rads"] if rep else None})
    return out


@app.get("/api/studies/{patient_id}")
def study(patient_id: str, who: Principal = Depends(require("doctor"))) -> dict[str, Any]:
    try:
        meta = ct_store.series_metadata(patient_id)
    except (ct_store.StudyNotFound, FileNotFoundError):
        raise HTTPException(404, "no such study") from None
    rep = clinical.get_report(get_db(), patient_id, signed_only=False)
    return {"patient_id": patient_id, "meta": meta, "quality": ct_store.check_quality(patient_id),
            "review": clinical.review_findings(patient_id), "report": rep}


@app.post("/api/studies/{patient_id}/pipeline")
def start_pipeline(patient_id: str, who: Principal = Depends(require("doctor"))) -> dict[str, str]:
    try:
        ct_store.study_entry(patient_id)
    except ct_store.StudyNotFound:
        raise HTTPException(404, "no such study") from None
    tr = Tracer(get_db())
    run_id = tr.start_run("ct_study", patient_id, who.user_id, who.role)

    def work() -> dict[str, Any]:
        env = Env(new_context(who, run_id), {"llm": get_llm(), "registry": get_registry()})
        res = ct_study.build().run(env, {"patient_id": patient_id})
        tr.end_run(run_id, res.status)
        return _pipeline_view(res.status, res.state, res.next_node)
    _job(run_id, who.user_id, work)
    return {"run_id": run_id}


def _pipeline_view(status: str, state: dict[str, Any], next_node: str) -> dict[str, Any]:
    return {"status": status, "next_node": next_node, "outcome": state.get("outcome"), "waiting": state.get("waiting"),
            "history": state.get("history"), "quality": state.get("quality"), "review": state.get("review"),
            "exam_category": state.get("exam_category"), "critical": state.get("critical"),
            "draft": state.get("draft"), "report": state.get("report"), "report_id": state.get("report_id")}


@app.get("/api/approvals")
def approvals(who: Principal = Depends(require("doctor"))) -> list[dict[str, Any]]:
    db = get_db()
    rows = db.query("SELECT * FROM approvals WHERE status='pending' ORDER BY created DESC")
    for r in rows:
        r["arguments"] = json.loads(r["arguments"])
        rep = clinical.get_report_by_id(db, r["arguments"].get("report_id", ""))
        r["report"] = rep
    return rows


class Decision(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")


@app.post("/api/approvals/{approval_id}")
def decide(approval_id: int, body: Decision, who: Principal = Depends(require("doctor"))) -> dict[str, Any]:
    db = get_db()
    row = db.one("SELECT * FROM approvals WHERE id=?", (approval_id,))
    if row is None or row["status"] != "pending":
        raise HTTPException(404, "no pending approval with this id")
    db.execute("UPDATE approvals SET status=?, decided_by=?, decided=? WHERE id=?",
               (body.decision, who.user_id, time.time(), approval_id))
    db.audit(who.user_id, who.role, f"approval:{body.decision}", row["tool"], True, row["arguments"][:200])
    # Resume the paused run from its checkpoint. Nothing from the earlier process is needed.
    env = Env(new_context(who, row["run_id"]), {"llm": get_llm(), "registry": get_registry()})
    if Checkpointer(db).latest(row["run_id"]) is None:
        return {"decision": body.decision, "resumed": False}
    res = ct_study.build().run(env, resume=True)
    Tracer(db).end_run(row["run_id"], res.status)
    return {"decision": body.decision, "resumed": True, **_pipeline_view(res.status, res.state, res.next_node)}


# ----------------------------------------------------------------- company employee
@app.get("/api/ops/overview")
def ops_overview(hours: float = 168, who: Principal = Depends(require("employee"))) -> dict[str, Any]:
    reg = get_registry()
    ctx = new_context(who, "adhoc")
    from ..tools.builtin import HoursArg, NoArgs
    return {"stats": reg.get("ops_run_stats").fn(HoursArg(hours=hours), ctx),      # type: ignore[union-attr]
            "health": reg.get("node_health").fn(NoArgs(), ctx),                      # type: ignore[union-attr]
            "dataset": reg.get("dataset_overview").fn(NoArgs(), ctx)}                # type: ignore[union-attr]


@app.get("/api/ops/runs")
def ops_runs(limit: int = 40, who: Principal = Depends(require("employee"))) -> list[dict[str, Any]]:
    db = get_db()
    rows = db.query("SELECT run_id,agent,role,status,started,ended FROM runs ORDER BY started DESC LIMIT ?", (min(limit, 200),))
    tr = Tracer(db)
    for r in rows:        # no task text, no answer: those can contain patient content
        s = tr.summary(r["run_id"])
        r.update({k: s[k] for k in ("llm_calls", "tool_calls", "total_tokens", "tool_errors", "wall_ms")})
    return rows


@app.get("/api/ops/audit")
def ops_audit(limit: int = 60, who: Principal = Depends(require("employee"))) -> list[dict[str, Any]]:
    return get_db().query("SELECT at,user_id,role,action,allowed,detail FROM audit ORDER BY id DESC LIMIT ?", (min(limit, 300),))


@app.get("/api/ops/verification")
def ops_verification(who: Principal = Depends(require("employee"))) -> list[dict[str, Any]]:
    out = []
    for f in sorted((ROOT / "verification").glob("t*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append({"task": d["task"], "ran_at": d["ran_at"], "model": d["model"],
                    "summary": d.get("summary") or {k: v.get("summary") for k, v in d.items() if isinstance(v, dict) and "summary" in v}
                    or {k: v for k, v in d.items() if k.endswith("summary")}})
    return out


@app.get("/api/ops/architecture")
def ops_architecture(who: Principal = Depends(require("employee"))) -> dict[str, Any]:
    reg = get_registry()
    return {"pipeline": ct_study.build().describe(),
            "tools": [{"name": n, "scopes": sorted(reg.get(n).scopes), "timeout_s": reg.get(n).timeout_s,       # type: ignore[union-attr]
                       "max_retries": reg.get(n).max_retries, "writes": reg.get(n).side_effect,                 # type: ignore[union-attr]
                       "needs_approval": reg.get(n).requires_approval} for n in reg.names()],                   # type: ignore[union-attr]
            "roles": {r: sorted(c.tool_names or []) for r, c in AGENTS.items()},
            "profile": get_settings().profile, "model": get_settings().model}


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

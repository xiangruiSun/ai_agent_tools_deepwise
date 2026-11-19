"""The tool catalogue. Each tool is one atomic operation with a typed argument model, the
scopes it needs and its execution policy. Role toolsets are built from these in profiles.py."""
from __future__ import annotations

import ast
import operator
import shutil
import subprocess
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .. import clinical
from ..config import EgressBlocked, assert_egress_allowed, get_settings
from ..ct import annotations, detector, lungrads, store as ct_store
from ..rag import kb
from ..structured.schemas import CohortQuery
from .registry import Tool, ToolContext, ToolError, ToolRegistry, TransientToolError


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Args):
    pass


class PatientArg(_Args):
    patient_id: str = Field(description="Patient identifier, e.g. LIDC-IDRI-0005.")


def _study(patient_id: str) -> None:
    try:
        ct_store.study_entry(patient_id)
    except ct_store.StudyNotFound:
        known = ", ".join(s["patient_id"] for s in ct_store.list_studies())
        raise ToolError("invalid_arguments", f"no study for patient {patient_id!r}",
                        hint=f"known patients: {known}") from None


# ---------------------------------------------------------------- calculator
class CalcArgs(_Args):
    expression: str = Field(description="Arithmetic only: numbers, + - * / ** and parentheses.", max_length=200)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ToolError("invalid_arguments", "exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("invalid_arguments", "only arithmetic is allowed", hint="example: (7.2 + 5.1) / 2")


def calculator(a: CalcArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        value = _eval(ast.parse(a.expression, mode="eval").body)
    except SyntaxError:
        raise ToolError("invalid_arguments", "the expression could not be parsed", hint="example: (7.2 + 5.1) / 2")
    except ZeroDivisionError:
        raise ToolError("invalid_arguments", "division by zero")
    return {"expression": a.expression, "value": round(float(value), 4)}


# ---------------------------------------------------------------- CT archive and analysis
def list_ct_studies(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    studies = ct_store.list_studies()
    return {"count": len(studies), "studies": studies}


def get_series_metadata(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
    _study(a.patient_id)
    return ct_store.series_metadata(a.patient_id)


def check_series_quality(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
    _study(a.patient_id)
    return ct_store.check_quality(a.patient_id)


def run_nodule_detector(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
    _study(a.patient_id)
    out = detector.detect(a.patient_id)
    keep = ("finding_id", "slice_index", "laterality", "zone", "nodule_type", "mean_diameter_mm",
            "long_axis_mm", "short_axis_mm", "solid_component_mm", "mean_hu", "score")
    return {"patient_id": a.patient_id, "detector": out["detector"], "n_candidates": out["n_candidates"],
            "note": "candidates are unreviewed and most are normal vessels; a radiologist must confirm them",
            "candidates": [{k: c[k] for k in keep} for c in out["candidates"]]}


def get_confirmed_findings(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
    _study(a.patient_id)
    return clinical.review_findings(a.patient_id)


class LungRadsArgs(_Args):
    nodule_type: Literal["solid", "part_solid", "ground_glass"]
    mean_diameter_mm: float = Field(gt=0, lt=100, description="Average of long and short axis, in mm.")
    solid_component_mm: float | None = Field(default=None, ge=0, lt=100, description="Required for part_solid.")
    is_baseline: bool = Field(default=True, description="True when there is no prior scan to compare with.")
    is_new: bool = Field(default=False, description="Follow-up scans only: the nodule was absent on the prior scan. "
                                                    "Must be false when is_baseline is true.")
    is_growing: bool = Field(default=False, description="Follow-up scans only: mean diameter grew by more than 1.5 mm. "
                                                        "Must be false when is_baseline is true.")
    benign_calcification: bool = False
    suspicious_features: bool = Field(default=False, description="For example spiculation or lymphadenopathy.")


def lung_rads_category(a: LungRadsArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        return lungrads.categorize(**a.model_dump())
    except lungrads.LungRadsInputError as exc:
        raise ToolError("invalid_arguments", str(exc), hint="correct the inputs and call again") from None


# ---------------------------------------------------------------- knowledge
class GuidelineArgs(_Args):
    query: str = Field(min_length=3, max_length=300)
    k: int = Field(default=3, ge=1, le=5)


def search_guideline(a: GuidelineArgs, ctx: ToolContext) -> dict[str, Any]:
    # Patients search the plain-language glossary; everyone else searches the clinical guidelines.
    audience = "patient" if ctx.principal.role == "patient" else "clinician"
    hits = kb.search(a.query, a.k, audience=audience)
    ctx.extras.setdefault("retrieved_chunks", set()).update(h["chunk_id"] for h in hits)
    return {"query": a.query, "chunks": hits, "note": "cite chunk_id values for anything you state from these"}


class CaseSearchArgs(_Args):
    query: str = Field(min_length=5, max_length=500, description="Clinical description to find similar published cases.")
    top_k: int = Field(default=3, ge=1, le=5)


def search_similar_cases(a: CaseSearchArgs, ctx: ToolContext) -> dict[str, Any]:
    s = get_settings()
    try:
        assert_egress_allowed(s.rag_url)
    except EgressBlocked as exc:
        raise ToolError("failed", str(exc), hint="do not retry; the deployment forbids this destination") from None
    try:
        r = httpx.post(f"{s.rag_url}/api/v1/search", headers={"X-API-Key": s.rag_api_key}, timeout=20,
                       json={"query_mode": "text", "query_text": a.query, "top_k": a.top_k})
    except httpx.HTTPError as exc:
        raise TransientToolError(f"case retrieval service unreachable: {type(exc).__name__}") from None
    if r.status_code >= 500:
        raise TransientToolError(f"case retrieval service returned {r.status_code}")
    if r.status_code != 200:
        raise ToolError("failed", f"case retrieval service rejected the request ({r.status_code})")
    data = r.json()
    cases = [{"case_id": x["case_id"], "source_id": (x["evidence"][0]["source_id"] if x["evidence"] else None),
              "age": x.get("age"), "sex": x.get("gender"), "score": round(x["final_score"], 2),
              "excerpt": (x["evidence"][0]["excerpt"][:420] if x["evidence"] else "")} for x in data["results"]]
    ctx.extras.setdefault("retrieved_cases", set()).update(c["case_id"] for c in cases)
    return {"corpus": "PMC-Patients subset (published case reports, CC BY-NC-SA)", "eligible_cases": data["total_eligible_cases"],
            "cases": cases, "note": "published case reports, not this hospital's patients; cite case_id"}


# ---------------------------------------------------------------- reports
class DraftArgs(_Args):
    patient_id: str
    report: dict[str, Any] = Field(description="A CTReport object.")


def save_draft_report(a: DraftArgs, ctx: ToolContext) -> dict[str, Any]:
    from ..structured.rules import validate_report
    from ..structured.schemas import CTReport
    _study(a.patient_id)
    try:
        rep = CTReport.model_validate(a.report)
    except Exception as exc:
        raise ToolError("invalid_arguments", f"report does not match the CTReport schema: {exc}"[:600]) from None
    problems = validate_report(rep, clinical.review_findings(a.patient_id))
    if problems:
        raise ToolError("invalid_arguments", "report failed business rules: " + "; ".join(problems[:6]),
                        hint="fix the report and save again")
    rid = clinical.save_draft(ctx.db, a.patient_id, ct_store.study_entry(a.patient_id)["study_uid"],
                              rep.model_dump(), ctx.principal.user_id, ctx.run_id)
    return {"report_id": rid, "status": "draft"}


class ReportIdArg(_Args):
    report_id: str


def sign_report(a: ReportIdArg, ctx: ToolContext) -> dict[str, Any]:
    try:
        return clinical.sign_report(ctx.db, a.report_id, ctx.principal.user_id)
    except KeyError:
        raise ToolError("invalid_arguments", f"no report {a.report_id!r}") from None


def _may_read_report(a: PatientArg, ctx: ToolContext) -> None:
    p = ctx.principal
    if p.role == "patient" and a.patient_id != p.patient_id:
        raise ToolError("unauthorized", "a patient may only read their own report")
    if p.role == "employee":
        raise ToolError("unauthorized", "company staff may not read patient reports")


def get_report(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
    rep = clinical.get_report(ctx.db, a.patient_id, signed_only=ctx.principal.role == "patient")
    if rep is None:
        return {"patient_id": a.patient_id, "available": False,
                "note": "no signed report is available yet" if ctx.principal.role == "patient" else "no report yet"}
    return {"patient_id": a.patient_id, "available": True, "status": rep["status"], "report_id": rep["report_id"],
            "report": rep["body"]}


def get_my_report(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    if not ctx.principal.patient_id:
        raise ToolError("unauthorized", "this account is not linked to a patient record")
    return get_report(PatientArg(patient_id=ctx.principal.patient_id), ctx)


class TermArg(_Args):
    term: str = Field(min_length=2, max_length=80)


def explain_term(a: TermArg, ctx: ToolContext) -> dict[str, Any]:
    hits = kb.search(a.term, 2, audience="patient")
    ctx.extras.setdefault("retrieved_chunks", set()).update(h["chunk_id"] for h in hits)
    return {"term": a.term, "glossary": hits} if hits else {"term": a.term, "glossary": [],
                                                           "note": "not in the glossary; say you cannot explain it"}


class CallbackArgs(_Args):
    reason: str = Field(min_length=5, max_length=300)


def request_callback(a: CallbackArgs, ctx: ToolContext) -> dict[str, Any]:
    ctx.db.execute("INSERT INTO memory(user_id,kind,text,source,created) VALUES (?,?,?,?,?)",
                   ("care_team", "callback_request", f"{ctx.principal.patient_id}: {a.reason}", ctx.run_id, time.time()))
    return {"requested": True, "note": "the care team has been asked to contact the patient"}


# ---------------------------------------------------------------- data governance and operations
def dataset_overview(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    m = ct_store.manifest()
    full = annotations.cohort_query()
    return {"local_ct_studies": len(m["studies"]), "local_bytes": sum(s["bytes"] for s in m["studies"]),
            "source": m["source"], "license": m["license"],
            "size_bands": {b: sum(1 for s in m["studies"] if s["band"] == b) for b in ("none", "lt6", "6to8", "8to15", "ge15")},
            "annotation_database": {"scans": full["total_scans_in_database"], "scans_with_annotations": full["matching_scans"],
                                    "annotations": full["matching_annotations"]}}


def deid_audit(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
    _study(a.patient_id)
    return ct_store.deid_audit(a.patient_id)


def build_cohort(a: CohortQuery, ctx: ToolContext) -> dict[str, Any]:
    if a.min_diameter_mm is not None and a.max_diameter_mm is not None and a.min_diameter_mm > a.max_diameter_mm:
        raise ToolError("invalid_arguments", "min_diameter_mm is larger than max_diameter_mm")
    return annotations.cohort_query(**a.model_dump())


def detector_vs_readers(a: PatientArg, ctx: ToolContext) -> dict[str, Any]:
    _study(a.patient_id)
    det = detector.detect(a.patient_id)
    nodules = annotations.reader_nodules(a.patient_id)
    m = annotations.match(det["candidates"], nodules)
    return {"patient_id": a.patient_id, "detector": det["detector"], "reader_nodules": len(nodules),
            "detected": len(m["pairs"]), "missed": len(m["missed_nodules"]), "false_positives": len(m["false_positives"]),
            "missed_detail": [{"nodule_id": n["nodule_id"], "mean_diameter_mm": n["mean_diameter_mm"],
                               "nodule_type": n["nodule_type"], "readers": n["n_readers"]}
                              for n in nodules if n["nodule_id"] in m["missed_nodules"]],
            "use": "missed nodules are hard examples for the next training round"}


class HoursArg(_Args):
    hours: float = Field(default=24, gt=0, le=24 * 30)


def ops_run_stats(a: HoursArg, ctx: ToolContext) -> dict[str, Any]:
    since = time.time() - a.hours * 3600
    runs = ctx.db.query("SELECT agent,status,started,ended FROM runs WHERE started>=?", (since,))
    spans = ctx.db.query("SELECT s.kind,s.name,s.latency_ms,s.prompt_tokens,s.completion_tokens,s.ok,s.error_kind,s.attempts "
                         "FROM spans s JOIN runs r ON r.run_id=s.run_id WHERE r.started>=?", (since,))

    def pct(vals: list[float], q: float) -> float:
        vals = sorted(vals)
        return round(vals[min(len(vals) - 1, int(q * len(vals)))], 1) if vals else 0.0

    by_status: dict[str, int] = {}
    by_agent: dict[str, int] = {}
    for r in runs:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_agent[r["agent"]] = by_agent.get(r["agent"], 0) + 1
    llm = [s for s in spans if s["kind"] == "llm"]
    tools = [s for s in spans if s["kind"] == "tool"]
    errors: dict[str, int] = {}
    for s in spans:
        if not s["ok"]:
            errors[s["error_kind"] or "unknown"] = errors.get(s["error_kind"] or "unknown", 0) + 1
    slow: dict[str, list[float]] = {}
    for s in tools:
        slow.setdefault(s["name"], []).append(s["latency_ms"])
    return {"window_hours": a.hours, "runs": len(runs), "runs_by_status": by_status, "runs_by_agent": by_agent,
            "llm_calls": len(llm), "llm_latency_ms_p50": pct([s["latency_ms"] for s in llm], 0.5),
            "llm_latency_ms_p95": pct([s["latency_ms"] for s in llm], 0.95),
            "tokens": sum(s["prompt_tokens"] + s["completion_tokens"] for s in llm),
            "tool_calls": len(tools), "tool_error_rate": round(sum(1 for s in tools if not s["ok"]) / len(tools), 3) if tools else 0.0,
            "tool_retries": sum(max(0, s["attempts"] - 1) for s in tools), "errors_by_kind": errors,
            "tool_latency_ms_p95": {k: pct(v, 0.95) for k, v in sorted(slow.items())}}


class LimitArg(_Args):
    limit: int = Field(default=10, ge=1, le=50)


def ops_recent_failures(a: LimitArg, ctx: ToolContext) -> dict[str, Any]:
    rows = ctx.db.query("SELECT run_id,agent,role,status,started,ended FROM runs WHERE status NOT IN ('completed','running') "
                        "ORDER BY started DESC LIMIT ?", (a.limit,))
    for r in rows:
        errs = ctx.db.query("SELECT kind,name,error_kind FROM spans WHERE run_id=? AND ok=0 ORDER BY id LIMIT 5", (r["run_id"],))
        r["errors"] = errs   # names and kinds only: payloads may contain patient content
    return {"count": len(rows), "runs": rows}


def node_health(a: NoArgs, ctx: ToolContext) -> dict[str, Any]:
    """Health of this deployment node: model server, retrieval sidecar, GPU, disk, database."""
    s = get_settings()
    out: dict[str, Any] = {"profile": s.profile}
    try:
        tags = httpx.get(f"{s.ollama_url}/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        out["model_server"] = {"ok": s.model in names, "configured_model": s.model, "installed_models": names}
    except httpx.HTTPError as exc:
        out["model_server"] = {"ok": False, "error": type(exc).__name__}
    try:
        out["retrieval_service"] = {"ok": httpx.get(f"{s.rag_url}/api/v1/health/ready", timeout=5).status_code == 200}
    except httpx.HTTPError as exc:
        out["retrieval_service"] = {"ok": False, "error": type(exc).__name__}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        name, used, total, util = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        out["gpu"] = {"ok": True, "name": name, "memory_used_mb": int(used), "memory_total_mb": int(total), "utilization_pct": int(util)}
    except Exception:
        out["gpu"] = {"ok": False, "error": "nvidia-smi not available"}
    du = shutil.disk_usage(s.data_dir)
    out["disk"] = {"ok": du.free / du.total > 0.05, "free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1)}
    out["ct_archive"] = {"ok": True, "studies": len(ct_store.list_studies())}
    out["database"] = {"ok": True, "runs": ctx.db.one("SELECT count(*) AS n FROM runs")["n"],
                       "pending_approvals": ctx.db.one("SELECT count(*) AS n FROM approvals WHERE status='pending'")["n"]}
    out["healthy"] = all(v.get("ok", True) for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------- registration
def build_registry() -> ToolRegistry:
    r = ToolRegistry()
    S = frozenset
    r.register(Tool("calculator", "Evaluate an arithmetic expression exactly. Use it for every calculation.",
                    CalcArgs, calculator, timeout_s=5))
    r.register(Tool("list_ct_studies", "List the chest CT studies in the archive (the worklist).", NoArgs,
                    list_ct_studies, scopes=S({"study:read"})))
    r.register(Tool("get_series_metadata", "DICOM header summary of a patient's CT series: scanner, slice thickness, "
                    "kernel, contrast, number of slices.", PatientArg, get_series_metadata, scopes=S({"study:read"})))
    r.register(Tool("check_series_quality", "Check whether a CT series is suitable for automated nodule analysis. "
                    "Returns passed, issues and warnings.", PatientArg, check_series_quality, scopes=S({"study:read"})))
    r.register(Tool("run_nodule_detector", "Run the nodule candidate detector on a patient's CT. Slow (seconds). "
                    "Returns ranked, UNREVIEWED candidates with measurements.", PatientArg, run_nodule_detector,
                    scopes=S({"study:analyze"}), timeout_s=120, max_retries=1))
    r.register(Tool("get_confirmed_findings", "Findings for a patient after radiologist review of the detector "
                    "candidates, with measurements. Use these, not raw candidates, for any report.", PatientArg,
                    get_confirmed_findings, scopes=S({"study:analyze"}), timeout_s=120, max_retries=1))
    r.register(Tool("lung_rads_category", "Deterministic Lung-RADS v2022 category and management for ONE nodule. "
                    "Always use this instead of working the category out yourself.", LungRadsArgs, lung_rads_category,
                    scopes=S({"kb:read"}), timeout_s=5))
    r.register(Tool("search_guideline", "Search the guideline knowledge base. Returns text chunks with chunk_id "
                    "values that must be cited.", GuidelineArgs, search_guideline, scopes=S({"kb:read"}), timeout_s=10))
    r.register(Tool("search_similar_cases", "Find similar published case reports (PMC-Patients) for a clinical "
                    "description. Returns case_id values and excerpts.", CaseSearchArgs, search_similar_cases,
                    scopes=S({"cases:search"}), timeout_s=30, max_retries=2))
    r.register(Tool("save_draft_report", "Save a validated draft CT report for a patient.", DraftArgs,
                    save_draft_report, scopes=S({"report:draft"}), side_effect=True, idempotent=False, max_retries=0))
    r.register(Tool("sign_report", "Sign a draft report so it becomes final and visible to the patient. Needs "
                    "human approval.", ReportIdArg, sign_report, scopes=S({"report:sign"}), side_effect=True,
                    idempotent=False, requires_approval=True, max_retries=0))
    # No role scope here on purpose: access is decided per record by _may_read_report
    # (doctors any, a patient only their own signed report, company staff none).
    r.register(Tool("get_report", "Read a patient's CT report.", PatientArg, get_report,
                    authorize=_may_read_report))
    r.register(Tool("get_my_report", "Read the signed CT report of the patient you are talking to.", NoArgs,
                    get_my_report, scopes=S({"self:read"})))
    r.register(Tool("explain_term", "Look up a medical term in the patient glossary.", TermArg, explain_term,
                    scopes=S({"self:read"}), timeout_s=10))
    r.register(Tool("request_callback", "Ask the care team to contact the patient. Use when the patient asks for "
                    "advice you may not give, or reports symptoms.", CallbackArgs, request_callback,
                    scopes=S({"self:read"}), side_effect=True, idempotent=False, max_retries=0))
    r.register(Tool("dataset_overview", "Summary of the local CT dataset and the LIDC-IDRI annotation database.",
                    NoArgs, dataset_overview, scopes=S({"dataset:read"}), timeout_s=120))
    r.register(Tool("deid_audit", "Audit the DICOM headers of one study for identifying fields. Returns field "
                    "names only, never values.", PatientArg, deid_audit, scopes=S({"dataset:read"})))
    r.register(Tool("build_cohort", "Count scans in the LIDC-IDRI annotation database that match a cohort definition.",
                    CohortQuery, build_cohort, scopes=S({"dataset:read"}), timeout_s=120))
    r.register(Tool("detector_vs_readers", "Compare detector output with radiologist annotations for one study: "
                    "detected, missed and false-positive counts.", PatientArg, detector_vs_readers,
                    scopes=S({"dataset:read"}), timeout_s=120))
    r.register(Tool("ops_run_stats", "Aggregate agent run statistics for a time window: runs, latency, tokens, "
                    "tool error rate.", HoursArg, ops_run_stats, scopes=S({"ops:read"})))
    r.register(Tool("ops_recent_failures", "List recent runs that did not complete, with error kinds only.",
                    LimitArg, ops_recent_failures, scopes=S({"ops:read"})))
    r.register(Tool("node_health", "Health of this deployment node: model server, retrieval service, GPU, disk.",
                    NoArgs, node_health, scopes=S({"ops:read"}), timeout_s=30))
    return r

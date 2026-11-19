"""Tests of the domain rules and the access model. No model is called. Tests that need the CT
studies are skipped when data/lidc has not been downloaded."""
from __future__ import annotations

import pytest

from dwagent.agent.guards import grounding_check
from dwagent.agent.loop import AgentState
from dwagent.catalog import get_registry
from dwagent.config import EgressBlocked, assert_egress_allowed, get_settings
from dwagent.ct import lungrads
from dwagent.llm.base import ToolCall
from dwagent.profiles import AGENTS, principal
from dwagent.rag import kb
from dwagent.runtime.trace import Tracer
from dwagent.store import Database
from dwagent.structured.rules import validate_explanation, validate_report
from dwagent.structured.schemas import CTReport, PatientExplanation
from dwagent.tools.executor import AwaitingApproval, execute_tool
from dwagent.tools.registry import ToolContext

HAVE_CT = (get_settings().lidc_dir / "manifest.json").exists()
needs_ct = pytest.mark.skipif(not HAVE_CT, reason="run scripts/download_lidc.py first")


def ctx_for(role: str, patient_id: str | None = None) -> ToolContext:
    db = Database(":memory:")
    tr = Tracer(db)
    return ToolContext(principal(f"{role}.test", role, patient_id), db, tr, run_id=tr.start_run("test", "t"))


def call(ctx: ToolContext, name: str, **args):
    return execute_tool(get_registry(), ToolCall("c", name, args), ctx, sleep=lambda s: None)


# ---------------------------------------------------------------- Lung-RADS v2022
@pytest.mark.parametrize("ntype,mm,solid,kw,expected", [
    ("solid", 5.9, None, {}, "2"), ("solid", 6.0, None, {}, "3"), ("solid", 7.9, None, {}, "3"),
    ("solid", 8.0, None, {}, "4A"), ("solid", 14.9, None, {}, "4A"), ("solid", 15.0, None, {}, "4B"),
    ("part_solid", 5.9, 2.0, {}, "2"), ("part_solid", 10.0, 5.9, {}, "3"), ("part_solid", 10.0, 6.0, {}, "4A"),
    ("part_solid", 12.0, 8.0, {}, "4B"), ("ground_glass", 29.9, None, {}, "2"), ("ground_glass", 30.0, None, {}, "3"),
    ("solid", 3.9, None, {"is_baseline": False, "is_new": True}, "2"),
    ("solid", 4.0, None, {"is_baseline": False, "is_new": True}, "3"),
    ("solid", 6.0, None, {"is_baseline": False, "is_new": True}, "4A"),
    ("solid", 8.0, None, {"is_baseline": False, "is_growing": True}, "4B"),
    ("solid", 7.0, None, {"is_baseline": False, "is_growing": True}, "4A"),
    ("solid", 20.0, None, {"benign_calcification": True}, "1"),
    ("solid", 7.0, None, {"suspicious_features": True}, "4X"),
    ("solid", 5.0, None, {"suspicious_features": True}, "2"),      # the 4X modifier applies to 3 and 4 only
])
def test_lung_rads_thresholds(ntype, mm, solid, kw, expected):
    assert lungrads.categorize(ntype, mm, solid, **kw)["category"] == expected


def test_lung_rads_rejects_inconsistent_input():
    for bad in (dict(nodule_type="part_solid", mean_diameter_mm=8), dict(nodule_type="solid", mean_diameter_mm=0),
                dict(nodule_type="solid", mean_diameter_mm=5, is_new=True),
                dict(nodule_type="part_solid", mean_diameter_mm=5, solid_component_mm=6)):
        with pytest.raises(lungrads.LungRadsInputError):
            lungrads.categorize(**bad)
    assert lungrads.exam_category(["2", "4A", "3"]) == "4A" and lungrads.exam_category([]) == "1"


# ---------------------------------------------------------------- business rules
EVIDENCE = {"patient_id": "P1", "findings": [
    {"finding_id": "F1", "laterality": "left", "zone": "mid", "nodule_type": "solid", "mean_diameter_mm": 8.1,
     "long_axis_mm": 9.0, "short_axis_mm": 7.2, "solid_component_mm": None, "slice_index": 80,
     "benign_calcification": False, "spiculated": False}]}


def good_report(**over) -> CTReport:
    f = {"finding_id": "F1", "laterality": "left", "zone": "mid", "nodule_type": "solid", "mean_diameter_mm": 8.1,
         "solid_component_mm": None, "slice_index": 80, "lung_rads": "4A",
         "description": "Solid nodule in the left mid zone, mean diameter 8.1 mm."}
    base = dict(patient_id="P1", technique="Chest CT, 2.5 mm slices.", comparison="None available.", findings=[f],
                impression="One solid pulmonary nodule. Lung-RADS 4A.", exam_lung_rads="4A",
                recommendation=lungrads.MANAGEMENT["4A"],
                guideline_citations=["lung_rads_v2022#category-4a-suspicious"], limitations="")
    base.update(over)
    return CTReport.model_validate(base)


def test_valid_report_passes_and_each_rule_bites():
    assert validate_report(good_report(), EVIDENCE) == []
    f = good_report().findings[0].model_dump()
    cases = {
        "wrong category": good_report(findings=[{**f, "lung_rads": "3"}]),
        "wrong size": good_report(findings=[{**f, "mean_diameter_mm": 9.1, "description": "Solid nodule, mean diameter 8.1 mm in size."}]),
        "wrong side": good_report(findings=[{**f, "laterality": "right"}]),
        "invented finding": good_report(findings=[f, {**f, "finding_id": "F9"}]),
        "missing finding": good_report(findings=[]),
        "wrong exam category": good_report(exam_lung_rads="3"),
        "category missing from impression": good_report(impression="One solid pulmonary nodule, suspicious."),
        "wrong follow-up": good_report(recommendation="Low-dose CT in 12 months."),
        "invented measurement": good_report(impression="A 12.4 mm nodule. Lung-RADS 4A."),
        "invented citation": good_report(guideline_citations=["lung_rads_v2022#made-up"]),
        "no citation": good_report(guideline_citations=[]),
        "other patient": good_report(patient_id="P2"),
    }
    for name, rep in cases.items():
        assert validate_report(rep, EVIDENCE), f"rule did not catch: {name}"


def test_report_schema_rejects_bad_types_and_extra_fields():
    with pytest.raises(Exception):
        good_report(exam_lung_rads="5")
    with pytest.raises(Exception):
        good_report(diagnosis="lung cancer")


def test_patient_explanation_rules():
    report = good_report().model_dump()
    ok = PatientExplanation(summary="The scan found one solid nodule of 8.1 mm in the left lung.",
                            what_was_found=["One solid nodule, 8.1 mm."], terms=["patient_glossary#solid-nodule"],
                            next_step=report["recommendation"], questions_for_doctor=["What happens next?"],
                            urgent_escalation=False)
    assert validate_explanation(ok, report) == []
    assert validate_explanation(ok.model_copy(update={"next_step": "Come back in a year."}), report)
    assert validate_explanation(ok.model_copy(update={"summary": "You have cancer, a nodule of 8.1 mm was found."}), report)
    assert validate_explanation(ok.model_copy(update={"summary": "The scan found a nodule of 14 mm in the left lung."}), report)
    assert validate_explanation(ok.model_copy(update={"terms": ["lung_rads_v2022#category-4a-suspicious"]}), report)


# ---------------------------------------------------------------- grounding guard
def _state(answer: str, observations: list[str]) -> AgentState:
    msgs = [{"role": "tool", "name": "t", "content": o} for o in observations]
    return AgentState(run_id="r", task="t", messages=msgs, answer_candidate=answer)


def test_grounding_guard_catches_the_three_observed_failures():
    ctx = ctx_for("doctor")
    obs = ['{"ok": true, "result": {"category": "4A", "chunks": [{"chunk_id": "lung_rads_v2022#category-4a-suspicious"}]}}']
    assert grounding_check(_state("The exam is Lung-RADS 4A [lung_rads_v2022#category-4a-suspicious].", obs), ctx) == []
    assert "category 2" in grounding_check(_state("EXAM CATEGORY: 2", obs), ctx)[0]
    assert "category 3" in grounding_check(_state("For this nodule the Lung-RADS category is 3.", obs), ctx)[0]
    assert "not a real id" in grounding_check(_state("Category 4A applies [12345].", obs), ctx)[0]
    assert "not a guideline chunk" in grounding_check(_state("Lung-RADS 4A [lung_rads_v2022#category-9].", obs), ctx)[0]
    assert "did not do it" in grounding_check(_state("I will now proceed with the corrected parameters.", obs), ctx)[0]
    failed = ['{"ok": false, "error_kind": "invalid_arguments", "error": "x", "hint": null}']
    assert grounding_check(_state("The nodule is Lung-RADS 3.", failed), ctx)      # a failed call grounds nothing


# ---------------------------------------------------------------- knowledge base
def test_knowledge_base_separates_audiences():
    assert len(kb.load_chunks()) >= 30
    assert all(h["chunk_id"].startswith("patient_glossary#") for h in kb.search("what is a nodule", 3, audience="patient"))
    top = kb.search("solid nodule 8 mm baseline management", 3, audience="clinician")
    assert top and not any(h["chunk_id"].startswith("patient_glossary#") for h in top)


# ---------------------------------------------------------------- access model
def test_each_role_is_offered_only_its_own_tools():
    reg = get_registry()
    seen = {role: {t.name for t in reg.allowed_for(principal("u", role, "P"), [*cfg.tool_names, "remember"])}
            for role, cfg in AGENTS.items()}
    assert seen["patient"] == {"get_my_report", "explain_term", "request_callback", "calculator", "remember"}
    assert "sign_report" not in seen["doctor"] and "assess_nodules" in seen["doctor"]
    assert not seen["employee"] & {"get_report", "get_my_report", "get_confirmed_findings", "run_nodule_detector",
                                   "list_ct_studies", "sign_report", "save_draft_report"}
    assert not seen["doctor"] & {"ops_run_stats", "deid_audit", "build_cohort"}
    # Scope check on the full registry, ignoring what an agent is configured to offer.
    all_patient = {t.name for t in reg.allowed_for(principal("u", "patient", "P"))}
    assert not all_patient & {"run_nodule_detector", "list_ct_studies", "sign_report", "save_draft_report",
                              "ops_run_stats", "build_cohort", "search_similar_cases", "get_confirmed_findings"}


def test_executor_refuses_tools_outside_the_role_even_when_called_directly():
    assert call(ctx_for("patient", "P1"), "list_ct_studies").error_kind == "unauthorized"
    assert call(ctx_for("patient", "P1"), "ops_run_stats", hours=1).error_kind == "unauthorized"
    assert call(ctx_for("employee"), "get_report", patient_id="P1").error_kind == "unauthorized"
    assert call(ctx_for("patient", "P1"), "get_report", patient_id="P2").error_kind == "unauthorized"
    assert call(ctx_for("patient", "P1"), "get_report", patient_id="P1").ok           # own record, nothing signed yet
    assert call(ctx_for("doctor"), "ops_run_stats", hours=1).error_kind == "unauthorized"
    with pytest.raises(AwaitingApproval):
        call(ctx_for("doctor"), "sign_report", report_id="rep_x")


def test_calculator_is_arithmetic_only():
    c = ctx_for("doctor")
    assert call(c, "calculator", expression="(7.2 + 5.1) / 2").content["value"] == 6.15
    for bad in ("__import__('os').system('dir')", "open('x')", "2 ** 999", "1/0", "a + 1"):
        assert call(c, "calculator", expression=bad).error_kind == "invalid_arguments"


def test_memory_refuses_credentials():
    assert call(ctx_for("patient", "P1"), "remember", text="My portal password is hunter2 please keep it").error_kind == "invalid_arguments"
    assert call(ctx_for("patient", "P1"), "remember", text="I prefer short answers in plain words.").ok


def test_onprem_profile_blocks_external_hosts(monkeypatch):
    monkeypatch.setenv("DWAGENT_PROFILE", "onprem")
    assert_egress_allowed("http://127.0.0.1:11434")
    with pytest.raises(EgressBlocked):
        assert_egress_allowed("https://api.example.com/v1/chat")
    monkeypatch.setenv("DWAGENT_ALLOWED_HOSTS", "pacs.hospital.lan")
    assert_egress_allowed("http://pacs.hospital.lan:8042")


# ---------------------------------------------------------------- real CT data
@needs_ct
def test_quality_gate_and_reader_annotations_on_real_studies():
    from dwagent.ct import annotations, store
    studies = store.list_studies()
    assert len(studies) == 10
    assert all(store.check_quality(s["patient_id"])["passed"] for s in studies)
    assert annotations.reader_nodules("LIDC-IDRI-0028") == []
    nodules = annotations.reader_nodules("LIDC-IDRI-0005")
    assert [n["mean_diameter_mm"] for n in nodules] == [8.1, 6.4, 6.3]
    assert store.deid_audit("LIDC-IDRI-0005")["passed"]


@needs_ct
def test_template_report_for_every_real_study_passes_the_rules():
    from dwagent import clinical
    from dwagent.ct import store
    from dwagent.pipelines.ct_study import template_report
    for s in store.list_studies():
        pid = s["patient_id"]
        ev = {"patient_id": pid, "findings": clinical.review_findings(pid)["findings"]}
        rep = CTReport.model_validate(template_report({"meta": store.series_metadata(pid), "evidence": ev}, "template"))
        assert validate_report(rep, ev) == [], pid

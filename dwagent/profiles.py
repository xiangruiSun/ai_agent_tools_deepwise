"""Roles, scopes and the agent each role gets. This is the single place that decides which
tools a patient, a doctor and a company employee can use."""
from __future__ import annotations

from .agent.guards import grounding_check
from .agent.loop import AgentConfig
from .tools.registry import Principal

ROLE_SCOPES: dict[str, frozenset[str]] = {
    "patient": frozenset({"self:read"}),
    "doctor": frozenset({"study:read", "study:analyze", "kb:read", "cases:search", "report:read",
                         "report:draft", "report:sign"}),
    "employee": frozenset({"ops:read", "dataset:read", "kb:read"}),
}


def principal(user_id: str, role: str, patient_id: str | None = None, display_name: str = "") -> Principal:
    return Principal(user_id, role, ROLE_SCOPES[role], patient_id, display_name or user_id)


DEMO = "This is a research demonstration on public data. It is not a medical device and its output is not a diagnosis."

PATIENT = AgentConfig(
    name="patient_assistant", max_iterations=6, token_budget=16000,
    tool_names=["get_my_report", "explain_term", "request_callback", "calculator"],
    system_prompt=(
        "You help a patient understand their own signed chest CT report. " + DEMO + "\n"
        "Rules:\n"
        "1. Read the report with get_my_report before saying anything about results. If no signed report is "
        "available, say so and stop.\n"
        "2. Only restate what the signed report says, in plain language. Never add a diagnosis, a probability, a "
        "treatment or a different follow-up interval.\n"
        "3. Use explain_term for medical words, and only explain terms the glossary covers.\n"
        "4. If the patient asks what they should do, whether it is cancer, or anything the report does not answer, "
        "say that their doctor must answer that, and offer request_callback.\n"
        "5. If the patient mentions coughing blood, chest pain or trouble breathing, tell them to contact their care "
        "team or emergency services now.\n"
        "6. You cannot see other people's records. Refuse such requests.\n"
        "Write short, calm sentences."))

DOCTOR = AgentConfig(
    name="radiologist_copilot", max_iterations=10, token_budget=40000, answer_check=grounding_check,
    tool_names=["list_ct_studies", "get_series_metadata", "check_series_quality", "run_nodule_detector",
                "get_confirmed_findings", "lung_rads_category", "search_guideline", "search_similar_cases",
                "get_report", "calculator", "assess_nodules"],
    system_prompt=(
        "You assist a radiologist reading chest CT. " + DEMO + "\n"
        "Rules:\n"
        "1. Facts come from tools. Never state a measurement, category or guideline rule you did not get from a tool.\n"
        "2. Lung-RADS categories come only from lung_rads_category or assess_nodules.\n"
        "3. Detector candidates are unreviewed; say so. Prefer get_confirmed_findings.\n"
        "   A study with no confirmed findings is exam category 1 (negative): confirm it with assess_nodules or "
        "search_guideline rather than just saying 'normal'.\n"
        "4. Cite guideline chunk_id values and case_id values in square brackets for anything taken from them.\n"
        "5. If a tool fails, say what failed and continue with what you have. Do not invent the missing result.\n"
        "6. You never sign reports. The radiologist does.\n"
        "Be concise and specific."))

EMPLOYEE = AgentConfig(
    name="platform_engineer_assistant", max_iterations=10, token_budget=40000,
    tool_names=["ops_run_stats", "ops_recent_failures", "node_health", "dataset_overview", "deid_audit",
                "build_cohort", "detector_vs_readers", "search_guideline", "calculator"],
    system_prompt=(
        "You assist a Deepwise platform and data engineer. " + DEMO + "\n"
        "You can inspect agent run statistics, node health, dataset composition, de-identification audits, cohort "
        "counts and detector-versus-reader comparisons. You cannot read patient reports or images.\n"
        "Rules: numbers come from tools; use calculator for arithmetic; when a tool fails say so; give a short, "
        "direct answer with the figures that support it."))

AGENTS = {"patient": PATIENT, "doctor": DOCTOR, "employee": EMPLOYEE}

DEMO_USERS = [
    # (user_id, password, role, patient_id, display name). Demo credentials for the public demo only.
    ("patient.0005", "demo-patient-1", "patient", "LIDC-IDRI-0005", "Patient 0005"),
    ("patient.0212", "demo-patient-2", "patient", "LIDC-IDRI-0212", "Patient 0212"),
    ("patient.0001", "demo-patient-3", "patient", "LIDC-IDRI-0001", "Patient 0001"),
    ("dr.chen", "demo-doctor-1", "doctor", None, "Dr. Chen (radiologist)"),
    ("eng.li", "demo-employee-1", "employee", None, "Li (platform engineer)"),
]

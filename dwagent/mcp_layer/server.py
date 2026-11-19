"""MCP server "deepwise-ct": exposes the CT and guideline tools over the Model Context Protocol.

Run:  python -m dwagent.mcp_layer.server            (stdio transport)

Where each control lives
    Authentication   here, at the server: the caller presents a token (DWAGENT_MCP_TOKEN for
                     stdio; an Authorization header for HTTP transports). No token, no tools.
    Authorization    here, at the server, on every call: the token maps to a principal and the
                     call goes through the same executor as local tools (role scopes, then the
                     resource check such as "a patient may only read their own report").
    Execution safety here for timeouts, retries and input validation; at the HOST for which
                     servers and tools are allowed, per-call timeouts and human approval.
The server never trusts a role or patient id sent in tool arguments.
"""
from __future__ import annotations

import json
import os
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from ..catalog import get_registry, new_context
from ..llm.base import ToolCall
from ..profiles import principal
from ..runtime.trace import Tracer
from ..store import get_db
from ..tools.executor import execute_tool
from ..tools.registry import Principal

# Demo token table. In a hospital these are issued by the identity provider.
TOKENS: dict[str, Principal] = {
    "tok-doctor-demo": principal("dr.chen", "doctor"),
    "tok-patient-0005-demo": principal("patient.0005", "patient", "LIDC-IDRI-0005"),
    "tok-employee-demo": principal("eng.li", "employee"),
}

server = MCPServer("deepwise-ct", instructions="Chest CT study tools for the Deepwise agent demo. Research use only.")


def _caller() -> Principal | None:
    return TOKENS.get(os.environ.get("DWAGENT_MCP_TOKEN", ""))


def _run(tool: str, **arguments: Any) -> dict[str, Any]:
    who = _caller()
    if who is None:
        return {"ok": False, "error_kind": "unauthenticated", "error": "missing or unknown token"}
    db = get_db()
    tracer = Tracer(db)
    run_id = tracer.start_run("mcp_server", tool, who.user_id, who.role, {"transport": "stdio"})
    res = execute_tool(get_registry(), ToolCall(id="mcp", name=tool, arguments=arguments),
                       new_context(who, run_id, db, tracer))
    tracer.end_run(run_id, "completed" if res.ok else "failed")
    return json.loads(res.observation(limit=20000))


@server.tool(description="List the chest CT studies in the archive.")
def list_ct_studies() -> dict[str, Any]:
    return _run("list_ct_studies")


@server.tool(description="Check whether a patient's CT series is suitable for automated nodule analysis.")
def check_series_quality(patient_id: str) -> dict[str, Any]:
    return _run("check_series_quality", patient_id=patient_id)


@server.tool(description="Run the nodule candidate detector on a patient's CT. Returns unreviewed candidates.")
def run_nodule_detector(patient_id: str) -> dict[str, Any]:
    return _run("run_nodule_detector", patient_id=patient_id)


@server.tool(description="Deterministic Lung-RADS v2022 category and management for one nodule.")
def lung_rads_category(nodule_type: Literal["solid", "part_solid", "ground_glass"], mean_diameter_mm: float,
                       solid_component_mm: float | None = None, is_baseline: bool = True, is_new: bool = False,
                       is_growing: bool = False, benign_calcification: bool = False,
                       suspicious_features: bool = False) -> dict[str, Any]:
    return _run("lung_rads_category", nodule_type=nodule_type, mean_diameter_mm=mean_diameter_mm,
                solid_component_mm=solid_component_mm, is_baseline=is_baseline, is_new=is_new,
                is_growing=is_growing, benign_calcification=benign_calcification,
                suspicious_features=suspicious_features)


@server.tool(description="Search the guideline knowledge base. Returns chunks with chunk_id values to cite.")
def search_guideline(query: str, k: int = 3) -> dict[str, Any]:
    return _run("search_guideline", query=query, k=k)


@server.tool(description="Read a patient's CT report. Patients may read only their own signed report.")
def get_report(patient_id: str) -> dict[str, Any]:
    return _run("get_report", patient_id=patient_id)


if __name__ == "__main__":
    server.run("stdio")

# Deepwise Agent Tools

Role-scoped AI agent tools for a chest CT service: a patient, a radiologist and a company engineer
each sign in to the same portal and get a different agent with different tools.

**Research demonstration on public data. Not a medical device. Nothing it produces is a diagnosis.**
The nodule detector is a rule-based stand-in, and the "radiologist review" of its candidates is
simulated from the public LIDC reader annotations.

It is built on the earlier retrieval project
[deepwise_rag](https://github.com/xiangruiSun/deepwise_rag), which runs unchanged as a sidecar
service and is called by the agents as a tool.

## What is in it

| Layer | What it does | Code |
|---|---|---|
| Tool-calling loop | model → tool → observation → model, with nine explicit stop reasons | `dwagent/agent/loop.py` |
| Tool execution | validation, authorization, approval, idempotency, timeout, retry, trace | `dwagent/tools/executor.py` |
| Grounding guard | rejects answers whose categories or citations no tool returned | `dwagent/agent/guards.py` |
| Structured output | Pydantic schemas, business rules, bounded repair, template fallback | `dwagent/structured/` |
| Runtime | graph, conditional routing, checkpoint per node, resume, human sign-off | `dwagent/runtime/`, `dwagent/pipelines/ct_study.py` |
| Reasoning and planning | direct, CoT, self-reflection, Tree of Thoughts; planner / executor | `dwagent/agent/strategies.py`, `planner.py` |
| Skills | a business workflow as one tool | `dwagent/skills/` |
| Memory | working context, session history, per-user long-term memory | `dwagent/memory/` |
| MCP | server, client and host integration (MCP Python SDK 2.x) | `dwagent/mcp_layer/` |
| Multi-agent | planner, researcher, writer, critic; three orchestration patterns; blackboard | `dwagent/multi/` |
| CT layer | DICOM archive, quality gate, detector, reader annotations, Lung-RADS v2022 | `dwagent/ct/` |
| DICOM gateway | C-STORE receiver for the scanner or PACS | `dwagent/gateway/scp.py` |
| Portal | FastAPI + a no-build single page, three roles | `dwagent/web/` |

Requirements for each job task are in [`requirements/`](requirements/00_overview.txt). Measured
results on the real data are in [`verification/REPORT.md`](verification/REPORT.md).

## Run it

Needs Python 3.13, [Ollama](https://ollama.com) and about 12 GB of GPU memory for the default model.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
ollama pull qwen3:14b
.venv/Scripts/python scripts/download_lidc.py               # 10 CT studies, 0.69 GB, CC BY 3.0
```

Similar-case search uses the earlier project as a sidecar (optional; the agents report it as
unavailable when it is not running):

```bash
git clone https://github.com/xiangruiSun/deepwise_rag _ref/deepwise_rag
python -m venv _ref/.venv-rag && _ref/.venv-rag/Scripts/python -m pip install -r _ref/deepwise_rag/backend/requirements.lock
# download PMC-Patients into _ref/deepwise_rag/data/pmc_patients/raw/ as described in its docs/open_datasets.md,
# then in _ref/deepwise_rag/backend:  prepare-dataset pmc-patients, ingest, reindex --embedding lexical
bash scripts/start_rag.sh                                    # port 8100
```

Start the portal:

```bash
.venv/Scripts/python -m uvicorn dwagent.web.app:app --port 8200
```

Open http://127.0.0.1:8200 and pick a demo account on the sign-in page.

| Account | Role | Sees |
|---|---|---|
| `patient.0005`, `patient.0212`, `patient.0001` | Patient | own signed report and image, explanation, assistant |
| `dr.chen` | Radiologist | worklist, study view, pipeline, sign-off queue, copilot |
| `eng.li` | Deepwise engineer | operations, verification, architecture, audit log, assistant |

A fresh install has no signed reports. Sign in as `dr.chen`, open a study, run the pipeline and
approve it; the report then appears in that patient's account. `python scripts/verify.py t03` does
this for all ten studies.

### Putting it online from this machine

```bash
.venv/Scripts/python -m dwagent.web.public        # public mode, still 127.0.0.1:8200 only
```

Public mode generates a session secret and one random password per account into
`data/runtime/public_credentials.json` (not committed), stops sending passwords to the browser,
limits agent jobs per visitor and adds security headers. Point a tunnel at `http://127.0.0.1:8200`;
nothing else on the machine is exposed. `--rotate` issues new passwords. Details and limits are in
`requirements/15_portal_and_tradeoffs.txt`, part C.

## Verify it

```bash
.venv/Scripts/python -m pytest tests -q          # 60 tests: agent core with a scripted model, domain rules, access model
.venv/Scripts/python scripts/verify.py all       # every job task on the real data, about an hour
.venv/Scripts/python scripts/build_report.py     # verification/REPORT.md
```

## Data and licences

See [`data/README.md`](data/README.md). LIDC-IDRI is CC BY 3.0. PMC-Patients is CC BY-NC-SA 4.0:
non-commercial use only. No data is committed to this repository.

## Limits worth knowing before relying on anything here

- Ten studies and one 14-billion-parameter local model. The numbers show what happened on this
  data, not general performance.
- Lung-RADS is a screening system; LIDC-IDRI is not a screening cohort. All studies are treated as
  baseline scans.
- The detector misses pure ground-glass nodules by design and produces about ten false positives
  per scan.
- Demo accounts, one process, SQLite. See each requirement file's "Known limits".

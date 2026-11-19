/* Single-page portal. No framework and no build step, so the on-prem install needs no npm.
   All dynamic text goes through h() as text nodes: model output is never inserted as HTML. */
const $app = document.getElementById("app");
const state = { me: null, page: null, session: "s" + Date.now().toString(36), chat: [] };

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: "same-origin", headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(data.error || r.statusText); e.status = r.status; throw e; }
  return data;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms");
const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleString() : "—");
const pill = (text, kind) => h("span", { class: "pill " + (kind || "") }, text);
const catPill = (c) => pill("Lung-RADS " + c, c === "1" || c === "2" ? "ok" : c === "3" ? "warn" : "bad");
async function waitRun(runId, onTick) {
  for (;;) {
    const r = await api("/api/runs/" + runId);
    if (onTick) onTick(r);
    if (r.job !== "running") return r;
    await sleep(900);
  }
}

/* ---------------------------------------------------------------- login */
const ROLE_COPY = {
  patient: ["Patient", ["Read your own signed CT report", "Plain-language explanation, checked against the report", "Ask questions; the assistant cannot diagnose or advise treatment"]],
  doctor: ["Radiologist", ["Worklist, image viewer and the CT study pipeline", "Draft reports validated against measurements and Lung-RADS rules", "Approve or reject sign-off; copilot with guideline and case search"]],
  employee: ["Deepwise engineer", ["Agent run statistics, traces without patient content, audit log", "Dataset governance: de-identification audit, cohorts, hard examples", "Verification evidence and system architecture"]],
};
async function renderLogin(msg) {
  const accounts = await api("/api/demo-accounts");
  const oneClick = accounts.some((a) => a.password);          // local demo only; never on the public site
  const pick = (a) => { user.value = a.username; if (a.password) { pass.value = a.password; submit(); } else { pass.value = ""; pass.focus(); } };
  const user = h("input", { id: "u", autocomplete: "username" }), pass = h("input", { id: "p", type: "password", autocomplete: "current-password" });
  const err = h("div", { class: "err", role: "alert" }, msg || "");
  const submit = async (ev) => {
    ev && ev.preventDefault();
    try { state.me = await api("/api/login", { method: "POST", body: { username: user.value.trim(), password: pass.value } }); start(); }
    catch (e) { err.textContent = e.message; }
  };
  $app.replaceChildren(h("div", { class: "login" },
    h("h1", {}, "Deepwise Agent Tools"),
    h("p", { class: "lead" }, "A research demonstration of role-scoped AI agents for chest CT: tool calling, validated structured output, a checkpointed runtime with human sign-off, skills, MCP and multi-agent workflows. It runs on public data (LIDC-IDRI CT studies and PMC-Patients case reports) with a local model."),
    h("div", { class: "banner" }, "Not a medical device. Nothing on this site is a diagnosis or medical advice. The detector is a rule-based stand-in, and the “radiologist review” is simulated from the public LIDC reader annotations."),
    h("div", { class: "roles" }, Object.entries(ROLE_COPY).map(([role, [title, points]]) => h("div", { class: "role" },
      h("h2", {}, title), h("ul", {}, points.map((p) => h("li", {}, p))),
      accounts.filter((a) => a.role === role).map((a) => h("button", { class: "acct", onclick: () => pick(a) }, (oneClick ? "Sign in as " : "Use account: ") + a.name))))),
    h("div", { class: "card" }, h("h2", {}, "Sign in"),
      h("form", { class: "loginform", onsubmit: submit },
        h("div", {}, h("label", { for: "u" }, "Username"), user), h("div", {}, h("label", { for: "p" }, "Password"), pass),
        h("button", { class: "primary", type: "submit" }, "Sign in")), err,
      h("p", { class: "small muted" }, oneClick ? "Demo accounts are open on this local install because it holds only public research data."
        : "Pick an account above, then enter the password you were given by the person who shared this demo."))));
}

/* ---------------------------------------------------------------- shell */
const PAGES = {
  patient: [["report", "My report", pagePatientReport], ["chat", "Ask the assistant", pageChat], ["tools", "What the assistant can do", pageTools]],
  doctor: [["worklist", "Worklist", pageWorklist], ["approvals", "Sign-off queue", pageApprovals], ["chat", "Copilot", pageChat], ["tools", "Tools and limits", pageTools]],
  employee: [["ops", "Operations", pageOps], ["verify", "Verification", pageVerification], ["arch", "Architecture", pageArchitecture], ["audit", "Audit log", pageAudit], ["chat", "Assistant", pageChat], ["tools", "Tools and limits", pageTools]],
};
function start() { state.chat = []; go(PAGES[state.me.role][0][0]); }
function go(page, arg) {
  state.page = page;
  const pages = PAGES[state.me.role];
  const main = h("main", { class: "main" });
  $app.replaceChildren(h("div", { class: "shell" },
    h("aside", { class: "side" },
      h("div", { class: "brand" }, "Deepwise Agent Tools", h("small", {}, "research demo · public data")),
      h("nav", { class: "nav", "aria-label": "Main" }, pages.map(([id, label]) => h("button", { class: id === page ? "on" : "", onclick: () => go(id) }, label))),
      h("div", { class: "who" }, h("b", {}, state.me.name), state.me.role + (state.me.patient_id ? " · " + state.me.patient_id : ""),
        h("button", { onclick: async () => { await api("/api/logout", { method: "POST" }); state.me = null; renderLogin(); } }, "Sign out"))),
    main));
  main.append(h("div", { class: "banner" }, "Research demonstration on public data. Not a medical device; not a diagnosis."));
  const entry = pages.find((p) => p[0] === page);
  const loading = h("p", { class: "muted" }, "Loading…");
  main.append(loading);
  entry[2](main, arg).then(() => loading.remove()).catch((e) => { if (e.status === 401) renderLogin("Your session ended. Please sign in again."); else main.append(h("div", { class: "card" }, "Could not load this page: " + e.message)); });
}
const head = (title, sub, ...actions) => h("div", { class: "head" }, h("div", {}, h("h1", {}, title), h("div", { class: "muted" }, sub)), h("div", { class: "row" }, actions));

/* ---------------------------------------------------------------- trace */
function traceView(run) {
  const s = run.summary || {};
  const steps = (run.events || []).map((e) => h("div", { class: "step" },
    h("span", { class: "k " + e.kind }, e.kind),
    h("div", {}, h("span", { class: "mono" }, e.name), " ", e.ok ? null : pill(e.error_kind || "error", "bad"), e.attempts > 1 ? pill(e.attempts + " attempts", "warn") : null,
      e.kind === "tool" && e.input ? h("div", { class: "io" }, "in: " + e.input) : null,
      e.kind === "tool" && e.output ? h("div", { class: "io" }, "out: " + e.output.slice(0, 320)) : null),
    h("span", { class: "muted" }, fmtMs(e.latency_ms) + (e.kind === "llm" ? " · " + (e.prompt_tokens + e.completion_tokens) + " tok" : ""))));
  return h("details", { class: "trace" },
    h("summary", {}, `${s.llm_calls || 0} model calls · ${s.tool_calls || 0} tool calls · ${s.total_tokens || 0} tokens · ${fmtMs(s.wall_ms || 0)}` + (s.tool_errors ? ` · ${s.tool_errors} tool errors` : "")),
    steps);
}

/* ---------------------------------------------------------------- chat (every role) */
const SUGGEST = {
  patient: ["What did my scan show?", "What does Lung-RADS mean?", "What is a solid nodule?", "Do I have cancer?", "Please remember that I prefer short answers."],
  doctor: ["Assess patient LIDC-IDRI-0005 and cite the guideline.", "Which study in the worklist has the most suspicious exam category?", "Find similar published cases for a 9 mm solid nodule.", "What category is a new 5 mm solid nodule at follow-up?"],
  employee: ["Is this node healthy?", "Tool error rate and error kinds in the last 24 hours?", "How many scans have a solid nodule of at least 10 mm?", "Run the de-identification audit on LIDC-IDRI-0003.", "Show me the report of patient LIDC-IDRI-0005."],
};
async function pageChat(main) {
  const msgs = h("div", { class: "msgs", "aria-live": "polite" });
  const box = h("textarea", { rows: 2, placeholder: "Type a question", "aria-label": "Message" });
  const send = h("button", { class: "primary" }, "Send");
  const draw = () => msgs.replaceChildren(...state.chat.map((m) => h("div", { class: "msg " + m.role }, m.text,
    m.run ? traceView(m.run) : null, m.meta ? h("div", { class: "small muted" }, m.meta) : null)));
  const ask = async (text) => {
    text = (text || box.value).trim();
    if (!text || send.disabled) return;
    box.value = ""; send.disabled = true;
    state.chat.push({ role: "user", text }); const pending = { role: "sys", text: "Working…" }; state.chat.push(pending); draw();
    try {
      const { run_id } = await api("/api/chat", { method: "POST", body: { session_id: state.session, message: text } });
      const run = await waitRun(run_id, (r) => { pending.text = `Working… ${(r.events || []).length} steps so far`; draw(); });
      const res = run.result || {};
      state.chat.pop();
      state.chat.push({ role: "bot", text: res.answer || res.error || "No answer.", run,
        meta: res.stop_reason && res.stop_reason !== "final_answer" ? "Stopped: " + res.stop_reason.replace(/_/g, " ") : (res.memories_used || []).length ? "Used memory: " + res.memories_used.join(" · ") : "" });
    } catch (e) { state.chat.pop(); state.chat.push({ role: "sys", text: "Request failed: " + e.message }); }
    send.disabled = false; draw(); box.focus();
  };
  send.onclick = () => ask();
  box.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  main.append(head({ patient: "Ask the assistant", doctor: "Radiologist copilot", employee: "Platform assistant" }[state.me.role],
      `Agent ${state.me.agent} · up to ${state.me.limits.max_iterations} steps and ${state.me.limits.token_budget} tokens per question`,
      h("button", { onclick: () => { state.chat = []; state.session = "s" + Date.now().toString(36); draw(); } }, "New conversation")),
    h("div", { class: "card chat" }, msgs, h("div", { class: "suggest" }, SUGGEST[state.me.role].map((s) => h("button", { onclick: () => ask(s) }, s))),
      h("div", { class: "composer" }, box, send)));
  draw();
}

/* ---------------------------------------------------------------- tools page */
async function pageTools(main) {
  const mem = await api("/api/memory");
  const memList = h("div", {});
  const drawMem = (items) => memList.replaceChildren(items.length ? h("table", {}, h("tbody", {}, items.map((m) => h("tr", {}, h("td", {}, m.text),
    h("td", { class: "num" }, h("button", { onclick: async () => { await api("/api/memory/" + m.id, { method: "DELETE" }); drawMem(await api("/api/memory")); } }, "Forget")))))) : h("p", { class: "muted" }, "Nothing stored. Ask the assistant to remember a preference."));
  drawMem(mem);
  main.append(head("Tools and limits", "What the agent for your role is allowed to call. The server checks this again on every call."),
    h("div", { class: "card scroll" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Kind"), h("th", {}, "What it does"), h("th", {}, "Flags"))),
      h("tbody", {}, state.me.tools.map((t) => h("tr", {}, h("td", { class: "mono" }, t.name), h("td", {}, pill(t.kind, t.kind === "skill" ? "brand" : "")),
        h("td", {}, t.description), h("td", {}, t.writes ? pill("writes", "warn") : null, " ", t.needs_approval ? pill("needs approval", "bad") : null)))))),
    h("div", { class: "card" }, h("h2", {}, "Long-term memory"), h("p", { class: "muted small" }, "Preferences the assistant stored for you. Only you can see or delete them."), memList),
    h("div", { class: "card" }, h("h2", {}, "Your access scopes"), h("div", { class: "row" }, state.me.scopes.map((s) => pill(s, "brand")))));
}

/* ---------------------------------------------------------------- patient */
function reportView(rep) {
  return h("dl", { class: "report" },
    h("dt", {}, "Technique"), h("dd", {}, rep.technique), h("dt", {}, "Comparison"), h("dd", {}, rep.comparison),
    h("dt", {}, "Findings"), h("dd", {}, rep.findings.length ? h("table", {}, h("tbody", {}, rep.findings.map((f) => h("tr", {},
      h("td", { class: "mono" }, f.finding_id), h("td", {}, f.description), h("td", {}, catPill(f.lung_rads)))))) : "No pulmonary nodule identified."),
    h("dt", {}, "Impression"), h("dd", {}, rep.impression, " ", catPill(rep.exam_lung_rads)),
    h("dt", {}, "Recommendation"), h("dd", {}, rep.recommendation),
    h("dt", {}, "Guideline references"), h("dd", { class: "mono" }, rep.guideline_citations.join(", ")),
    rep.limitations ? [h("dt", {}, "Limitations"), h("dd", {}, rep.limitations)] : null);
}
async function pagePatientReport(main) {
  const d = await api("/api/my/report");
  main.append(head("My CT report", "Signed reports only. Drafts are never shown to patients."));
  if (!d.available) { main.append(h("div", { class: "card" }, "Your report has not been signed by a radiologist yet. Please check again later.")); return; }
  const out = h("div", {});
  const btn = h("button", { class: "primary" }, "Explain this report in plain language");
  btn.onclick = async () => {
    btn.disabled = true; out.replaceChildren(h("p", { class: "muted" }, "Writing and checking the explanation…"));
    try {
      const { run_id } = await api("/api/my/explanation", { method: "POST" });
      const run = await waitRun(run_id); const r = run.result || {};
      if (!r.valid) out.replaceChildren(h("p", {}, "The explanation did not pass the safety checks, so it is not shown. Please ask your doctor to go through the report with you."), traceView(run));
      else { const e = r.explanation; out.replaceChildren(
        h("p", {}, e.summary), h("h3", {}, "What was found"), h("ul", {}, e.what_was_found.map((x) => h("li", {}, x))),
        h("h3", {}, "Next step (copied from your report)"), h("p", {}, e.next_step),
        h("h3", {}, "Questions you could ask your doctor"), h("ul", {}, e.questions_for_doctor.map((x) => h("li", {}, x))),
        h("p", { class: "small muted" }, `Checked against the report in ${r.attempts.length} attempt(s): ` + r.attempts.map((a) => a.stage).join(" → ")), traceView(run)); }
    } catch (e) { out.replaceChildren(h("p", {}, "Could not create the explanation: " + e.message)); }
    btn.disabled = false;
  };
  const k = d.key_slices[0];
  main.append(h("div", { class: "grid2" },
    h("div", { class: "card" }, h("h2", {}, "Report"), h("p", { class: "small muted" }, `Signed by ${d.signed_by} on ${fmtTime(d.signed)}`), reportView(d.report)),
    h("div", {}, k ? h("div", { class: "card" }, h("h2", {}, "Image showing finding " + k.finding_id),
        h("div", { class: "viewer" }, h("img", { src: `/api/studies/${d.patient_id}/slice/${k.slice_index}.png`, alt: `CT image ${k.slice_index} with finding ${k.finding_id} circled` }))) : null,
      h("div", { class: "card" }, h("h2", {}, "Plain-language explanation"), h("p", { class: "muted small" }, "The explanation may only restate your signed report. It is checked before you see it."), btn, out))));
}

/* ---------------------------------------------------------------- doctor */
async function pageWorklist(main, openId) {
  if (openId) return pageStudy(main, openId);
  const rows = await api("/api/worklist");
  main.append(head("Worklist", `${rows.length} chest CT studies from LIDC-IDRI`),
    h("div", { class: "card scroll" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Patient"), h("th", { class: "num" }, "Slices"), h("th", { class: "num" }, "Thickness"), h("th", {}, "Report"), h("th", {}, "Exam"), h("th", {}, ""))),
      h("tbody", {}, rows.map((r) => h("tr", {}, h("td", { class: "mono" }, r.patient_id), h("td", { class: "num" }, r.n_slices), h("td", { class: "num" }, r.slice_thickness_mm + " mm"),
        h("td", {}, pill(r.report_status, r.report_status === "signed" ? "ok" : r.report_status === "draft" ? "warn" : "")),
        h("td", {}, r.exam_lung_rads ? catPill(r.exam_lung_rads) : "—"), h("td", { class: "num" }, h("button", { onclick: () => go("worklist", r.patient_id) }, "Open"))))))));
}
function pipelineView(p, onDecide) {
  if (!p) return h("div", {});
  const hist = (p.history || []).filter((x) => x.status === "ok");
  const kids = [h("div", { class: "stepper" }, hist.map((x) => h("span", { class: "node ok", title: fmtMs(x.ms) }, x.node)),
    p.status === "interrupted" ? h("span", { class: "node wait" }, "signoff: waiting for you") : null,
    p.status === "failed" ? h("span", { class: "node err" }, p.next_node + ": failed") : null)];
  if (p.outcome === "rejected_quality") kids.push(h("p", {}, pill("Series rejected", "bad"), " ", (p.quality.issues || []).join("; "), ". No detector was run."));
  if (p.draft) kids.push(h("p", { class: "small muted" }, p.draft.source === "model" ? `Draft written by the model, valid after ${p.draft.attempts} attempt(s).`
    : p.draft.source === "template" ? "Template report: no confirmed findings, no model call." : `Model draft failed validation ${p.draft.attempts} times; the deterministic template was used instead.`));
  if (p.critical) kids.push(h("p", {}, pill("Critical result", "bad"), " Category 4B/4X must be acknowledged at sign-off."));
  if (p.report) kids.push(reportView(p.report));
  if (p.status === "interrupted" && p.waiting) kids.push(h("div", { class: "card approval" }, h("h2", {}, "Sign-off required"),
    h("p", {}, "The pipeline is paused at a checkpoint. Signing makes the report visible to the patient."),
    h("div", { class: "row" }, h("button", { class: "primary", onclick: () => onDecide(p.waiting.approval_id, "approved") }, "Approve and sign"),
      h("button", { class: "danger", onclick: () => onDecide(p.waiting.approval_id, "rejected") }, "Reject"))));
  if (p.outcome === "signed") kids.push(h("p", {}, pill("Signed", "ok"), " The report is now visible to the patient."));
  if (p.outcome === "signoff_rejected") kids.push(h("p", {}, pill("Rejected", "bad"), " The draft was not signed."));
  return h("div", {}, kids);
}
async function pageStudy(main, pid) {
  const d = await api("/api/studies/" + pid);
  let k = d.review.findings.length ? d.review.findings[0].slice_index : Math.floor(d.meta.n_slices / 2);
  const img = h("img", { alt: "CT image" }), label = h("span", { class: "muted small" });
  const show = () => { img.src = `/api/studies/${pid}/slice/${k}.png`; img.alt = `CT image ${k} of ${d.meta.n_slices}`; label.textContent = `Image ${k} of ${d.meta.n_slices - 1}`; slider.value = k; };
  const slider = h("input", { type: "range", min: 0, max: d.meta.n_slices - 1, "aria-label": "Image number", oninput: (e) => { k = +e.target.value; show(); } });
  const pipe = h("div", {});
  const decide = async (id, decision) => { pipe.replaceChildren(h("p", { class: "muted" }, "Resuming from checkpoint…")); const r = await api("/api/approvals/" + id, { method: "POST", body: { decision } }); pipe.replaceChildren(pipelineView(r, decide)); };
  const runBtn = h("button", { class: "primary" }, "Run CT study pipeline");
  runBtn.onclick = async () => {
    runBtn.disabled = true; pipe.replaceChildren(h("p", { class: "muted" }, "Running…"));
    try { const { run_id } = await api(`/api/studies/${pid}/pipeline`, { method: "POST" });
      const run = await waitRun(run_id, (r) => pipe.replaceChildren(h("p", { class: "muted" }, `Running… ${(r.events || []).filter((e) => e.kind === "node").map((e) => e.name.split(".")[1]).join(" → ")}`)));
      pipe.replaceChildren(pipelineView(run.result, decide), traceView(run)); } catch (e) { pipe.replaceChildren(h("p", {}, "Pipeline failed: " + e.message)); }
    runBtn.disabled = false;
  };
  main.append(head(pid, `${d.meta.Manufacturer} ${d.meta.ManufacturerModelName} · ${d.meta.SliceThickness} mm · ${d.meta.ConvolutionKernel} kernel`, h("button", { onclick: () => go("worklist") }, "Back to worklist")),
    h("div", { class: "grid2" },
      h("div", { class: "card" }, h("div", { class: "viewer" }, img), h("div", { class: "row", style: "margin-top:8px" }, slider), label),
      h("div", {},
        h("div", { class: "card" }, h("h2", {}, "Acquisition check"), h("p", {}, pill(d.quality.passed ? "Passed" : "Failed", d.quality.passed ? "ok" : "bad"), " ", [...d.quality.issues, ...d.quality.warnings].join("; ") || "No issues."),
          h("h2", {}, "Confirmed findings"), h("p", { class: "small muted" }, `Review ${d.review.review}: ${d.review.candidates_reviewed} detector candidates reviewed, ${d.review.candidates_rejected} rejected, ${d.review.findings_added_by_reader} added by readers.`),
          d.review.findings.length ? h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Id"), h("th", {}, "Location"), h("th", {}, "Type"), h("th", { class: "num" }, "Mean diameter"), h("th", {}, "Source"), h("th", {}, ""))),
            h("tbody", {}, d.review.findings.map((f) => h("tr", {}, h("td", { class: "mono" }, f.finding_id), h("td", {}, f.laterality + " " + f.zone), h("td", {}, f.nodule_type.replace("_", "-")), h("td", { class: "num" }, f.mean_diameter_mm + " mm"),
              h("td", {}, pill(f.source === "detector_confirmed" ? "detector" : "reader added", f.source === "detector_confirmed" ? "brand" : "warn")), h("td", {}, h("button", { onclick: () => { k = f.slice_index; show(); } }, "View")))))) : h("p", {}, "No confirmed findings.")))),
    h("div", { class: "card" }, h("div", { class: "head" }, h("div", {}, h("h2", {}, "Pipeline"), h("div", { class: "muted small" }, "intake → quality gate → detect → review → categorise → draft → critical check → save → sign-off")), runBtn),
      pipe, d.report && !pipe.childNodes.length ? h("div", {}, h("p", { class: "small muted" }, `Latest report: ${d.report.status}` + (d.report.signed_by ? ` by ${d.report.signed_by}` : "")), reportView(d.report.body)) : null));
  show();
}
async function pageApprovals(main) {
  const rows = await api("/api/approvals");
  main.append(head("Sign-off queue", "Runs paused at a checkpoint, waiting for a radiologist"));
  if (!rows.length) { main.append(h("div", { class: "card" }, "Nothing is waiting. Run the pipeline on a study from the worklist.")); return; }
  for (const a of rows) {
    const card = h("div", { class: "card approval" });
    const decide = async (decision) => { card.replaceChildren(h("p", { class: "muted" }, "Resuming…")); const r = await api("/api/approvals/" + a.id, { method: "POST", body: { decision } }); card.replaceChildren(pipelineView(r, () => {})); };
    card.append(h("h2", {}, a.report ? a.report.patient_id : a.tool), h("p", { class: "small muted" }, `Requested ${fmtTime(a.created)} · run ${a.run_id}`),
      a.report ? reportView(a.report.body) : null, h("div", { class: "row", style: "margin-top:12px" }, h("button", { class: "primary", onclick: () => decide("approved") }, "Approve and sign"), h("button", { class: "danger", onclick: () => decide("rejected") }, "Reject")));
    main.append(card);
  }
}

/* ---------------------------------------------------------------- employee */
const tile = (v, l) => h("div", { class: "tile" }, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
async function pageOps(main) {
  const [o, runs] = await Promise.all([api("/api/ops/overview"), api("/api/ops/runs")]);
  const s = o.stats, hl = o.health;
  main.append(head("Operations", `Last 7 days on this node · profile ${hl.profile} · ${hl.healthy ? "healthy" : "attention needed"}`),
    h("div", { class: "grid3", style: "margin-bottom:16px" }, tile(s.runs, "agent runs"), tile(s.llm_calls, "model calls"), tile(s.tokens.toLocaleString(), "tokens"),
      tile(fmtMs(s.llm_latency_ms_p50), "model latency p50"), tile(fmtMs(s.llm_latency_ms_p95), "model latency p95"), tile((s.tool_error_rate * 100).toFixed(1) + "%", "tool error rate"), tile(s.tool_retries, "tool retries")),
    h("div", { class: "grid2" },
      h("div", { class: "card" }, h("h2", {}, "Node health"), h("table", {}, h("tbody", {},
        h("tr", {}, h("td", {}, "Model server"), h("td", {}, pill(hl.model_server.ok ? "ok" : "down", hl.model_server.ok ? "ok" : "bad"), " ", hl.model_server.configured_model || "")),
        h("tr", {}, h("td", {}, "Case retrieval service"), h("td", {}, pill(hl.retrieval_service.ok ? "ok" : "down", hl.retrieval_service.ok ? "ok" : "bad"))),
        h("tr", {}, h("td", {}, "GPU"), h("td", {}, hl.gpu.ok ? `${hl.gpu.name} · ${hl.gpu.memory_used_mb} / ${hl.gpu.memory_total_mb} MB` : "not available")),
        h("tr", {}, h("td", {}, "Disk free"), h("td", {}, `${hl.disk.free_gb} of ${hl.disk.total_gb} GB`)),
        h("tr", {}, h("td", {}, "CT archive"), h("td", {}, hl.ct_archive.studies + " studies")),
        h("tr", {}, h("td", {}, "Pending approvals"), h("td", {}, hl.database.pending_approvals))))),
      h("div", { class: "card" }, h("h2", {}, "Runs by status"), h("div", { class: "row" }, Object.entries(s.runs_by_status).map(([k, v]) => pill(`${k}: ${v}`, k === "completed" ? "ok" : k === "failed" ? "bad" : "warn"))),
        h("h2", { style: "margin-top:14px" }, "Errors by kind"), h("div", { class: "row" }, Object.keys(s.errors_by_kind).length ? Object.entries(s.errors_by_kind).map(([k, v]) => pill(`${k}: ${v}`, "bad")) : h("span", { class: "muted" }, "None")),
        h("h2", { style: "margin-top:14px" }, "Dataset"), h("p", { class: "small" }, `${o.dataset.local_ct_studies} local CT studies (${(o.dataset.local_bytes / 1e9).toFixed(2)} GB, ${o.dataset.license}); annotation database: ${o.dataset.annotation_database.scans} scans, ${o.dataset.annotation_database.annotations} reader annotations.`))),
    h("div", { class: "card scroll" }, h("h2", {}, "Recent runs"), h("p", { class: "small muted" }, "Timings and counts only. Prompts, answers and tool payloads are not shown to staff."),
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Started"), h("th", {}, "Agent"), h("th", {}, "Role"), h("th", {}, "Status"), h("th", { class: "num" }, "Model calls"), h("th", { class: "num" }, "Tool calls"), h("th", { class: "num" }, "Tokens"), h("th", { class: "num" }, "Tool errors"), h("th", { class: "num" }, "Wall"))),
        h("tbody", {}, runs.map((r) => h("tr", {}, h("td", {}, fmtTime(r.started)), h("td", { class: "mono" }, r.agent), h("td", {}, r.role), h("td", {}, pill(r.status, r.status === "completed" ? "ok" : r.status === "failed" ? "bad" : "warn")),
          h("td", { class: "num" }, r.llm_calls), h("td", { class: "num" }, r.tool_calls), h("td", { class: "num" }, r.total_tokens), h("td", { class: "num" }, r.tool_errors), h("td", { class: "num" }, fmtMs(r.wall_ms))))))));
}
async function pageVerification(main) {
  const rows = await api("/api/ops/verification");
  main.append(head("Verification evidence", "Each job task was run on the real data with the local model. These are the recorded results."));
  if (!rows.length) main.append(h("div", { class: "card" }, "No evidence files yet. Run: python scripts/verify.py all"));
  for (const r of rows) main.append(h("div", { class: "card" }, h("div", { class: "head" }, h("h2", { class: "mono" }, r.task), h("span", { class: "muted small" }, `${r.ran_at} · ${r.model}`)),
    h("pre", { class: "mono", style: "white-space:pre-wrap;margin:0" }, JSON.stringify(r.summary, null, 1))));
}
async function pageArchitecture(main) {
  const a = await api("/api/ops/architecture");
  main.append(head("Architecture", `Model ${a.model} · deployment profile ${a.profile}`),
    h("div", { class: "card" }, h("h2", {}, "CT study pipeline (runtime graph)"), h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Node"), h("th", {}, "Next"))),
      h("tbody", {}, a.pipeline.nodes.map((n) => h("tr", {}, h("td", { class: "mono" }, n), h("td", { class: "mono" }, (a.pipeline.edges[n] || []).join("  |  ").replace("__end__", "END"))))))),
    h("div", { class: "card scroll" }, h("h2", {}, "Tool execution policy"), h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Tool"), h("th", {}, "Scopes required"), h("th", { class: "num" }, "Timeout"), h("th", { class: "num" }, "Retries"), h("th", {}, "Flags"), h("th", {}, "Offered to"))),
      h("tbody", {}, a.tools.map((t) => h("tr", {}, h("td", { class: "mono" }, t.name), h("td", { class: "mono" }, t.scopes.join(", ") || "—"), h("td", { class: "num" }, t.timeout_s + " s"), h("td", { class: "num" }, t.max_retries),
        h("td", {}, t.writes ? pill("writes", "warn") : null, " ", t.needs_approval ? pill("needs approval", "bad") : null),
        h("td", {}, Object.entries(a.roles).filter(([, names]) => names.includes(t.name)).map(([r]) => pill(r, "brand")))))))));
}
async function pageAudit(main) {
  const rows = await api("/api/ops/audit");
  main.append(head("Audit log", "Logins, tool calls and denied requests"),
    h("div", { class: "card scroll" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Time"), h("th", {}, "User"), h("th", {}, "Role"), h("th", {}, "Action"), h("th", {}, "Result"), h("th", {}, "Note"))),
      h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, fmtTime(r.at)), h("td", { class: "mono" }, r.user_id), h("td", {}, r.role), h("td", { class: "mono" }, r.action), h("td", {}, pill(r.allowed ? "allowed" : "denied", r.allowed ? "ok" : "bad")), h("td", { class: "small muted" }, r.allowed ? "" : r.detail)))))));
}

(async () => { try { state.me = await api("/api/me"); start(); } catch { renderLogin(); } })();

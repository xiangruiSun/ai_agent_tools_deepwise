"""The tool-calling agent loop (ReAct): model response -> tool execution -> observation -> next
model response, with explicit stopping conditions.

Stop reasons
    final_answer       the model answered without requesting a tool (the normal ending)
    max_iterations     the step limit was reached
    token_budget       the run used its token allowance
    wall_clock         the run exceeded its time allowance
    loop_detected      the same tool call was repeated without progress
    tool_failures      too many consecutive tool failures
    awaiting_approval  a tool needs a human decision; the run can be resumed afterwards
    llm_error          the model could not be reached or timed out
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..llm.base import LLMClient, LLMError, ToolCall
from ..memory.working import fit_context
from ..tools.executor import AwaitingApproval, call_key, execute_tool
from ..tools.registry import ToolContext, ToolRegistry


@dataclass
class AgentConfig:
    name: str = "agent"
    system_prompt: str = "You are a careful assistant. Use the tools to get facts; do not guess."
    tool_names: list[str] | None = None
    max_iterations: int = 8
    token_budget: int = 24000
    wall_clock_s: float = 600.0
    max_consecutive_tool_failures: int = 3
    loop_repeat_limit: int = 3
    context_tokens: int = 6000
    temperature: float = 0.0
    finalize_on_limit: bool = True
    # Optional check of a proposed final answer: returns a list of problems (empty = accept).
    answer_check: Callable[["AgentState", ToolContext], list[str]] | None = None
    max_answer_repairs: int = 2


class AgentState(BaseModel):
    run_id: str
    task: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status: str = "running"          # running | completed | stopped | awaiting_approval | failed
    stop_reason: str = ""
    answer: str = ""
    call_counts: dict[str, int] = Field(default_factory=dict)
    consecutive_failures: int = 0
    pending_calls: list[dict[str, Any]] = Field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    elapsed_s: float = 0.0
    tool_log: list[dict[str, Any]] = Field(default_factory=list)
    answer_candidate: str = ""
    answer_repairs: int = 0
    guard_log: list[list[str]] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def run_agent(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, task: str | None = None,
              config: AgentConfig | None = None, state: AgentState | None = None,
              on_step: Callable[[AgentState], None] | None = None) -> AgentState:
    cfg = config or AgentConfig()
    tools = registry.allowed_for(ctx.principal, cfg.tool_names)
    allowed = {t.name for t in tools}
    schemas = [t.schema() for t in tools]
    if state is None:
        if task is None:
            raise ValueError("task is required for a new run")
        state = AgentState(run_id=ctx.run_id, task=task, messages=[
            {"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": task}])
    state.status, state.stop_reason, state.pending_approval = "running", "", None
    t_start = time.perf_counter() - state.elapsed_s

    def stop(status: str, reason: str, answer: str = "") -> AgentState:
        state.status, state.stop_reason = status, reason
        if answer:
            state.answer = answer
        state.elapsed_s = time.perf_counter() - t_start
        if on_step:
            on_step(state)
        return state

    def run_calls(calls: list[dict[str, Any]]) -> bool:
        """Execute tool calls and append observations. Returns False when the run must pause."""
        for i, c in enumerate(calls):
            call = ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            try:
                res = execute_tool(registry, call, ctx, allowed=allowed)
            except AwaitingApproval as wait:
                state.pending_calls = calls[i:]
                state.pending_approval = {"approval_id": wait.approval_id, "tool": wait.tool, "arguments": wait.arguments}
                return False
            state.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                   "content": res.observation()})
            state.tool_log.append({"tool": call.name, "ok": res.ok, "error_kind": res.error_kind,
                                   "attempts": res.attempts, "cached": res.cached,
                                   "latency_ms": round(res.latency_ms, 1)})
            state.consecutive_failures = 0 if res.ok else state.consecutive_failures + 1
            sig = call_key(call.name, call.arguments)
            state.call_counts[sig] = state.call_counts.get(sig, 0) + 1
            if state.call_counts[sig] == cfg.loop_repeat_limit - 1 and cfg.loop_repeat_limit > 2:
                state.messages.append({"role": "user", "content":
                                       f"You have already called {call.name} with these exact arguments. The result "
                                       "will not change. Use it, try something different, or give the final answer."})
        state.pending_calls = []
        return True

    # Resume: finish the tool calls that were waiting (for example after a human approval).
    if state.pending_calls and not run_calls(state.pending_calls):
        return stop("awaiting_approval", "awaiting_approval")

    while True:
        state.elapsed_s = time.perf_counter() - t_start
        limit = None
        if any(n >= cfg.loop_repeat_limit for n in state.call_counts.values()):
            limit = "loop_detected"
        elif state.consecutive_failures >= cfg.max_consecutive_tool_failures:
            limit = "tool_failures"
        elif state.iterations >= cfg.max_iterations:
            limit = "max_iterations"
        elif state.total_tokens >= cfg.token_budget:
            limit = "token_budget"
        elif state.elapsed_s >= cfg.wall_clock_s:
            limit = "wall_clock"
        if limit:
            return stop("stopped", limit, _wrap_up(llm, state, cfg, ctx, limit) if cfg.finalize_on_limit else "")

        context = fit_context(state.messages, cfg.context_tokens, anchor_task=state.task, iteration=state.iterations)
        try:
            with ctx.tracer.span(ctx.run_id, "llm", f"{cfg.name}.step{state.iterations + 1}",
                                 input={"messages": len(context), "tools": sorted(allowed)}) as sp:
                resp = llm.chat(context, tools=schemas or None, temperature=cfg.temperature)
                sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
                sp.output = {"content": resp.content[:800], "tool_calls": [c.name for c in resp.tool_calls]}
        except LLMError as exc:
            return stop("failed", "llm_error", f"The model call failed ({exc.kind}): {exc}")
        state.iterations += 1
        state.prompt_tokens += resp.usage.prompt_tokens
        state.completion_tokens += resp.usage.completion_tokens
        state.messages.append(resp.as_message())

        if not resp.tool_calls:                       # stopping condition: a plain answer
            state.answer_candidate = resp.content.strip()
            if cfg.answer_check is not None and state.answer_repairs < cfg.max_answer_repairs:
                with ctx.tracer.span(ctx.run_id, "guard", f"{cfg.name}.answer_check") as sp:
                    problems = cfg.answer_check(state, ctx)
                    sp.output = {"problems": problems}
                    if problems:
                        sp.fail("ungrounded_answer", "; ".join(problems)[:400])
                if problems:
                    state.answer_repairs += 1
                    state.guard_log.append(problems)
                    state.messages.append({"role": "user", "content":
                                           "Your answer cannot be accepted yet:\n"
                                           + "\n".join(f"{i}. {p}" for i, p in enumerate(problems, 1))
                                           + "\nFix these, using tools if needed, then write your COMPLETE answer "
                                             "again, not only the corrected part."})
                    continue
            if cfg.answer_check is not None:
                left = cfg.answer_check(state, ctx)
                if left:      # repairs exhausted: never pass an unverified statement off as verified
                    return stop("completed", "final_answer_unverified", state.answer_candidate
                                + "\n\n[Not verified against tool results: " + " | ".join(left)[:500] + "]")
            return stop("completed", "final_answer", state.answer_candidate)
        calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls]
        if not run_calls(calls):
            return stop("awaiting_approval", "awaiting_approval")
        if on_step:
            on_step(state)


def _wrap_up(llm: LLMClient, state: AgentState, cfg: AgentConfig, ctx: ToolContext, reason: str) -> str:
    """One last call without tools so a limited run still returns something honest."""
    note = {"role": "user", "content":
            f"Stop now ({reason.replace('_', ' ')}). Using only the observations above, give the best answer you "
            "can to the task. State clearly which parts could not be completed or verified."}
    try:
        with ctx.tracer.span(ctx.run_id, "llm", f"{cfg.name}.wrap_up") as sp:
            resp = llm.chat(fit_context(state.messages + [note], cfg.context_tokens), tools=None, temperature=0.0)
            sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
        state.prompt_tokens += resp.usage.prompt_tokens
        state.completion_tokens += resp.usage.completion_tokens
        return resp.content.strip()
    except LLMError:
        return ""


def tool_messages(state: AgentState) -> list[dict[str, Any]]:
    """Parsed tool observations, in order. Used by validators and by the evaluation scripts."""
    out = []
    for m in state.messages:
        if m["role"] == "tool":
            try:
                out.append({"name": m.get("name"), **json.loads(m["content"])})
            except json.JSONDecodeError:
                out.append({"name": m.get("name"), "ok": False, "raw": m["content"]})
    return out

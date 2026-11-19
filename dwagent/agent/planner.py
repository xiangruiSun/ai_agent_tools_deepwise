"""Planner / Executor: the model plans once, code executes.

The planner writes a Plan: steps that each call one tool, with dependencies between them.
The plan is validated before anything runs, which is where hallucinated tasks and goal drift
are caught: a step naming a tool that does not exist or is not allowed for this user, a
dependency on a step that is not in the plan, a cycle, or too many steps all reject the plan.
The executor then runs the dependency graph level by level (independent steps in parallel)
through the normal tool executor. A failed step skips its dependents; one bounded re-plan is
allowed. A final model call writes the answer from the step results only.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..llm.base import LLMClient, ToolCall
from ..structured.generate import generate_structured
from ..tools.executor import execute_tool
from ..tools.registry import ToolContext, ToolRegistry

MAX_STEPS = 8


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^s[0-9]+$")
    purpose: str = Field(min_length=5, max_length=200)
    tool: str
    arguments: dict[str, Any]
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_STEPS)


def validate_plan(plan: Plan, allowed_tools: set[str], registry: ToolRegistry) -> list[str]:
    errors = []
    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        errors.append("step ids must be unique")
    for s in plan.steps:
        if s.tool not in allowed_tools:
            errors.append(f"{s.id}: tool {s.tool!r} does not exist or is not available; choose from {sorted(allowed_tools)}")
            continue
        try:
            registry.get(s.tool).args_model.model_validate(s.arguments)   # type: ignore[union-attr]
        except Exception as exc:
            errors.append(f"{s.id}: arguments do not match {s.tool}: {str(exc)[:200]}")
        for d in s.depends_on:
            if d not in ids:
                errors.append(f"{s.id}: depends on unknown step {d!r}")
    if not errors and topological_levels(plan) is None:
        errors.append("the dependencies contain a cycle")
    return errors


def topological_levels(plan: Plan) -> list[list[PlanStep]] | None:
    remaining = {s.id: s for s in plan.steps}
    done: set[str] = set()
    levels = []
    while remaining:
        ready = [s for s in remaining.values() if set(s.depends_on) <= done]
        if not ready:
            return None
        levels.append(ready)
        for s in ready:
            done.add(s.id)
            del remaining[s.id]
    return levels


@dataclass
class PlanRun:
    status: str                       # completed | plan_rejected | failed
    answer: str = ""
    plan: dict[str, Any] | None = None
    results: dict[str, Any] = field(default_factory=dict)
    plan_attempts: int = 0
    replans: int = 0
    tokens: int = 0
    plan_errors: list[list[str]] = field(default_factory=list)


def plan_and_execute(llm: LLMClient, registry: ToolRegistry, ctx: ToolContext, goal: str,
                     tool_names: list[str], max_replans: int = 1) -> PlanRun:
    tools = registry.allowed_for(ctx.principal, tool_names)
    allowed = {t.name for t in tools}
    catalogue = "\n".join(f"- {t.name}: {t.description} Arguments schema: "
                          f"{json.dumps(t.schema()['function']['parameters'].get('properties', {}))}" for t in tools)
    out = PlanRun(status="failed")
    feedback = ""
    for round_no in range(max_replans + 1):
        res = generate_structured(
            llm, Plan,
            [{"role": "system", "content":
              "You are a planner. Break the goal into the smallest set of tool calls that achieves it. Each step "
              "calls exactly one tool with literal arguments. Use depends_on only when a step truly needs an earlier "
              f"step to have run. Use at most {MAX_STEPS} steps and only these tools:\n{catalogue}"},
             {"role": "user", "content": f"Goal: {goal}{feedback}"}],
            rules=lambda p: validate_plan(p, allowed, registry), max_attempts=3, ctx=ctx, span_name="planner")
        out.plan_attempts += len(res.attempts)
        out.tokens += res.tokens
        out.plan_errors += [a.errors for a in res.attempts if a.errors]
        if res.value is None:
            out.status = "plan_rejected"
            return out
        plan = res.value
        out.plan = plan.model_dump()
        results: dict[str, Any] = {}
        failed: set[str] = set()
        for level in topological_levels(plan) or []:
            runnable = [s for s in level if not (set(s.depends_on) & failed)]
            for s in level:
                if s not in runnable:
                    results[s.id] = {"ok": False, "skipped": "a step it depends on failed"}
                    failed.add(s.id)
            with cf.ThreadPoolExecutor(max_workers=max(1, len(runnable))) as pool:
                futs = {s.id: pool.submit(execute_tool, registry,
                                          ToolCall(id=f"plan.{s.id}", name=s.tool, arguments=s.arguments), ctx, allowed)
                        for s in runnable}
            for sid, fut in futs.items():
                r = fut.result()
                results[sid] = {"ok": r.ok, "result": r.content} if r.ok else {"ok": False, "error_kind": r.error_kind,
                                                                               "error": r.error}
                if not r.ok:
                    failed.add(sid)
        out.results = results
        if failed and round_no < max_replans:
            out.replans += 1
            feedback = ("\n\nYour previous plan had failing steps: " +
                        json.dumps({k: v for k, v in results.items() if not v["ok"]})[:800] +
                        "\nWrite a new plan that avoids these failures.")
            continue
        break
    with ctx.tracer.span(ctx.run_id, "llm", "planner.synthesise") as sp:
        resp = llm.chat([
            {"role": "system", "content": "Write the answer to the goal using only the step results. Cite chunk_id "
             "and case_id values in square brackets where you use them. If a step failed, say what is missing."},
            {"role": "user", "content": f"Goal: {goal}\n\nStep results:\n{json.dumps(out.results)[:9000]}"}])
        sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
    out.tokens += resp.usage.total
    out.answer = resp.content.strip()
    out.status = "completed"
    return out

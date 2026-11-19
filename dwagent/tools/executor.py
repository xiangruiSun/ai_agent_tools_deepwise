"""Tool execution engineering: argument validation, authorization, human approval, timeouts,
bounded retries with backoff, idempotency and tracing. Every failure is turned into an
observation the model can act on; nothing is raised into the agent loop except approval waits.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..llm.base import ToolCall
from ..store import dumps
from .registry import Tool, ToolContext, ToolError, ToolRegistry, has_scopes

_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: Any = None
    error_kind: str | None = None
    error: str | None = None
    hint: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0
    cached: bool = False

    def observation(self, limit: int = 6000) -> str:
        """The text the model reads back. Errors carry the kind and a hint so it can recover."""
        if self.ok:
            text = dumps({"ok": True, "result": self.content})
        else:
            text = dumps({"ok": False, "error_kind": self.error_kind, "error": self.error, "hint": self.hint})
        return text if len(text) <= limit else text[:limit] + '... [truncated]"}'


def call_key(name: str, arguments: dict[str, Any]) -> str:
    blob = json.dumps({"n": name, "a": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class AwaitingApproval(Exception):
    """Raised to pause the run until a human approves a tool call."""
    is_pause = True

    def __init__(self, approval_id: int, tool: str, arguments: dict[str, Any]):
        super().__init__(f"tool {tool} is waiting for human approval")
        self.approval_id, self.tool, self.arguments = approval_id, tool, arguments


def execute_tool(registry: ToolRegistry, call: ToolCall, ctx: ToolContext,
                 allowed: set[str] | None = None, sleep=time.sleep) -> ToolResult:
    with ctx.tracer.span(ctx.run_id, "tool", call.name, input=call.arguments) as sp:
        res = _execute(registry, call, ctx, allowed, sleep)
        sp.output, sp.attempts = (res.content if res.ok else {"error": res.error}), res.attempts
        if not res.ok:
            sp.fail(res.error_kind or "failed", res.error or "")
        return res


def _execute(registry: ToolRegistry, call: ToolCall, ctx: ToolContext, allowed: set[str] | None,
             sleep) -> ToolResult:
    t0 = time.perf_counter()

    def done(**kw: Any) -> ToolResult:
        return ToolResult(name=call.name, latency_ms=(time.perf_counter() - t0) * 1000, **kw)

    # 1. Tool discovery: the model may only call tools that exist and were offered to it.
    tool = registry.get(call.name)
    if tool is None or (allowed is not None and call.name not in allowed):
        offered = sorted(allowed) if allowed is not None else registry.names()
        return done(ok=False, error_kind="unknown_tool", error=f"no tool named {call.name!r}",
                    hint=f"available tools: {', '.join(offered)}")

    # 2. Authorization, part one: role scopes. Decided from the principal, not from the prompt.
    if not has_scopes(ctx.principal, tool.scopes):
        ctx.db.audit(ctx.principal.user_id, ctx.principal.role, f"tool:{tool.name}", "-", False, "missing scope")
        return done(ok=False, error_kind="unauthorized",
                    error=f"role {ctx.principal.role!r} may not use {tool.name}", hint="do not retry this tool")

    # 3. Schema validation of the arguments the model produced.
    try:
        args = tool.args_model.model_validate(call.arguments)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors())
        return done(ok=False, error_kind="invalid_arguments", error=problems,
                    hint="fix the arguments to match the tool schema and call again")

    # 4. Authorization, part two: the specific resource (for example, a patient's own record).
    if tool.authorize is not None:
        try:
            tool.authorize(args, ctx)
        except ToolError as exc:
            ctx.db.audit(ctx.principal.user_id, ctx.principal.role, f"tool:{tool.name}",
                         dumps(call.arguments)[:200], False, str(exc))
            return done(ok=False, error_kind="unauthorized", error=str(exc), hint="do not retry this tool")

    key = call_key(tool.name, args.model_dump(mode="json"))

    # 5. Human approval for consequential actions.
    if tool.requires_approval:
        row = ctx.db.one("SELECT * FROM approvals WHERE run_id=? AND call_key=?", (ctx.run_id, key))
        if row is None:
            aid = ctx.db.execute(
                "INSERT INTO approvals(run_id,call_key,tool,arguments,requested_by,status,created) VALUES (?,?,?,?,?,?,?)",
                (ctx.run_id, key, tool.name, dumps(call.arguments), ctx.principal.user_id, "pending", time.time()))
            raise AwaitingApproval(aid, tool.name, call.arguments)
        if row["status"] == "pending":
            raise AwaitingApproval(row["id"], tool.name, call.arguments)
        if row["status"] == "rejected":
            return done(ok=False, error_kind="rejected", error="a human reviewer rejected this action",
                        hint="do not retry; explain that the action was not approved")

    # 6. Idempotency: a write that already happened is never repeated; a read is memoised per run.
    if tool.side_effect:
        prior = ctx.db.one("SELECT result FROM idempotency WHERE key=?", (f"{ctx.run_id}:{key}",))
        if prior:
            return done(ok=True, content=json.loads(prior["result"]), cached=True)
    elif tool.idempotent and key in ctx.memo:
        return done(ok=True, content=ctx.memo[key], cached=True)

    # 7. Run with a timeout and bounded retries. Only retryable failures are retried, and a
    #    timed-out call is retried only when the tool is idempotent.
    attempts, last = 0, None
    while attempts <= tool.max_retries:
        attempts += 1
        future = _POOL.submit(tool.fn, args, ctx)
        try:
            out = future.result(timeout=tool.timeout_s)
            if tool.side_effect:
                ctx.db.execute("INSERT OR REPLACE INTO idempotency(key,tool,result,created) VALUES (?,?,?,?)",
                               (f"{ctx.run_id}:{key}", tool.name, dumps(out), time.time()))
            elif tool.idempotent:
                ctx.memo[key] = out
            ctx.db.audit(ctx.principal.user_id, ctx.principal.role, f"tool:{tool.name}",
                         dumps(call.arguments)[:200], True)
            return done(ok=True, content=out, attempts=attempts)
        except cf.TimeoutError:
            future.cancel()
            last = ToolError("timeout", f"{tool.name} did not finish within {tool.timeout_s}s",
                             retryable=tool.idempotent, hint="try a narrower request or continue without it")
        except ToolError as exc:
            last = exc
        except Exception as exc:  # a bug or an unexpected upstream failure
            last = ToolError("failed", f"{type(exc).__name__}: {exc}")
        if not last.retryable or attempts > tool.max_retries:
            break
        sleep(min(0.5 * 2 ** (attempts - 1), 4.0))
    assert last is not None
    return done(ok=False, error_kind=last.kind, error=str(last), hint=last.hint, attempts=attempts)

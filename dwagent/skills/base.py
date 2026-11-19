"""Skills: a business workflow packaged as one callable capability.

A tool is one atomic operation. A skill is a fixed sequence of tool calls with the business
rules between them written in code (gates, loops, how results combine). The model sees a
skill as a single tool, so it plans one step instead of many, and the order of operations
cannot drift. Each inner tool call still goes through the executor, so it is validated,
authorized, retried and traced exactly like a call the model made itself.
"""
from __future__ import annotations

from typing import Any

from ..llm.base import ToolCall
from ..tools.executor import execute_tool
from ..tools.registry import ToolContext, ToolError, ToolRegistry


class SkillRunner:
    def __init__(self, registry: ToolRegistry, ctx: ToolContext, skill: str):
        self.registry, self.ctx, self.skill = registry, ctx, skill
        self.steps: list[dict[str, Any]] = []

    def call(self, tool: str, **arguments: Any) -> Any:
        res = execute_tool(self.registry, ToolCall(id=f"{self.skill}.{len(self.steps) + 1}", name=tool,
                                                   arguments=arguments), self.ctx)
        self.steps.append({"tool": tool, "ok": res.ok, "error_kind": res.error_kind,
                           "latency_ms": round(res.latency_ms, 1)})
        if not res.ok:
            # Surface the inner failure with its kind so retry policy still applies one level up.
            raise ToolError(res.error_kind or "failed", f"{self.skill} step {tool} failed: {res.error}",
                            retryable=res.error_kind in ("transient", "timeout"), hint=res.hint)
        return res.content

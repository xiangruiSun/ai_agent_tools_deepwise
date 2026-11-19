"""Tool definitions: a name, a description, a Pydantic argument model (the JSON Schema the
model sees is generated from it) and the execution policy for the tool."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from ..runtime.trace import Tracer
from ..store import Database


@dataclass(frozen=True)
class Principal:
    """Who the agent is acting for. Authorization is decided from this, never from the prompt."""
    user_id: str
    role: str  # patient | doctor | employee | system
    scopes: frozenset[str]
    patient_id: str | None = None  # set for patients: the only record they may read
    display_name: str = ""


SYSTEM = Principal("system", "system", frozenset({"*"}), display_name="System")


class ToolError(Exception):
    """A tool failed in a way the model should be told about.

    kind: invalid_arguments | unknown_tool | unauthorized | timeout | failed | transient | needs_approval
    """

    def __init__(self, kind: str, message: str, retryable: bool = False, hint: str | None = None):
        super().__init__(message)
        self.kind, self.retryable, self.hint = kind, retryable, hint


class TransientToolError(ToolError):
    def __init__(self, message: str):
        super().__init__("transient", message, retryable=True)


@dataclass
class ToolContext:
    principal: Principal
    db: Database
    tracer: Tracer
    run_id: str = "adhoc"
    memo: dict[str, Any] = field(default_factory=dict)  # per-run cache for idempotent reads
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, ToolContext], Any]
    scopes: frozenset[str] = frozenset()
    timeout_s: float = 30.0
    max_retries: int = 2
    idempotent: bool = True            # safe to call twice with the same arguments
    side_effect: bool = False          # writes something; deduplicated by idempotency key
    requires_approval: bool = False    # a human must approve before it runs
    authorize: Callable[[Any, ToolContext], None] | None = None  # resource-level check
    schema_override: dict[str, Any] | None = None  # input schema supplied by a remote (MCP) server
    origin: str = "local"

    def schema(self) -> dict[str, Any]:
        params = _clean(self.schema_override or self.args_model.model_json_schema())
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}


def _clean(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop Pydantic titles so the schema the model reads is as small as possible."""
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def allowed_for(self, principal: Principal, names: list[str] | None = None) -> list[Tool]:
        """Tools this principal may even see. Hiding a tool is a convenience; the executor
        checks scopes again on every call."""
        pool = [self._tools[n] for n in (names or self.names()) if n in self._tools]
        return [t for t in pool if has_scopes(principal, t.scopes)]

    def schemas(self, principal: Principal, names: list[str] | None = None) -> list[dict[str, Any]]:
        return [t.schema() for t in self.allowed_for(principal, names)]

    def subset(self, names: list[str]) -> "ToolRegistry":
        r = ToolRegistry()
        for n in names:
            if n in self._tools:
                r._tools[n] = self._tools[n]
        return r


def has_scopes(principal: Principal, required: frozenset[str]) -> bool:
    return "*" in principal.scopes or required <= principal.scopes

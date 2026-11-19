"""MCP host side: connect to a server, discover its tools, and register them so the same agent
loop can call them like local tools.

    Host    the agent runtime (this process). Decides which servers to trust, which of their
            tools to offer to the model, applies timeouts and size limits, asks for approval.
    Client  one MCPConnection per server: holds the session, speaks the protocol.
    Server  owns the capability and enforces who may use it.

The official SDK is asynchronous; MCPConnection keeps one session open on a background event
loop so the synchronous agent loop can call it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any

from mcp import Client, StdioServerParameters
from pydantic import BaseModel, ConfigDict

from ..config import ROOT
from ..tools.registry import Tool, ToolContext, ToolError, ToolRegistry, TransientToolError


class AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class MCPConnection:
    def __init__(self, command: str | None = None, args: list[str] | None = None, token: str = "",
                 env: dict[str, str] | None = None, call_timeout_s: float = 120.0):
        self.params = StdioServerParameters(
            command=command or sys.executable, args=args or ["-m", "dwagent.mcp_layer.server"], cwd=str(ROOT),
            env={**os.environ, **(env or {}), "DWAGENT_MCP_TOKEN": token, "PYTHONPATH": str(ROOT)})
        self.call_timeout_s = call_timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-client")
        self._client: Client | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    async def _session(self) -> None:
        try:
            async with Client(self.params) as client:
                self._client, self._stop = client, asyncio.Event()
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # startup failure is reported to the caller of start()
            self._error = exc
            self._ready.set()

    def start(self, timeout_s: float = 60.0) -> "MCPConnection":
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._session(), self._loop)
        if not self._ready.wait(timeout_s) or self._error is not None:
            raise ConnectionError(f"could not start MCP server: {self._error or 'timeout'}")
        return self

    def _submit(self, coro: Any, timeout_s: float) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout_s)

    def server_info(self) -> dict[str, Any]:
        assert self._client is not None
        info = self._client.server_info
        return {"name": getattr(info, "name", None), "version": getattr(info, "version", None),
                "protocol_version": str(self._client.protocol_version)}

    def list_tools(self) -> list[dict[str, Any]]:
        assert self._client is not None
        res = self._submit(self._client.list_tools(), 30)
        return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema
                 if hasattr(t, "input_schema") else getattr(t, "inputSchema", {})} for t in res.tools]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None
        res = self._submit(self._client.call_tool(name, arguments), self.call_timeout_s)
        is_error = getattr(res, "is_error", None)
        if is_error is None:
            is_error = getattr(res, "isError", False)
        structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        text = "".join(getattr(c, "text", "") for c in (res.content or []))
        if is_error:
            return {"ok": False, "error_kind": "failed", "error": text[:500]}
        if structured is not None:
            return structured.get("result", structured) if isinstance(structured, dict) else {"result": structured}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"ok": True, "result": text}

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=15)
        self._loop.call_soon_threadsafe(self._loop.stop)


def mcp_registry(conn: MCPConnection, allow: set[str] | None = None, prefix: str = "") -> ToolRegistry:
    """Turn discovered MCP tools into registry tools. ``allow`` is the host's allowlist: a tool
    the server offers but the host has not approved is never shown to the model."""
    reg = ToolRegistry()
    for t in conn.list_tools():
        if allow is not None and t["name"] not in allow:
            continue

        def call(a: AnyArgs, ctx: ToolContext, _name: str = t["name"]) -> Any:
            try:
                out = conn.call_tool(_name, a.model_dump())
            except TimeoutError:
                raise ToolError("timeout", f"MCP tool {_name} timed out", retryable=True) from None
            except Exception as exc:
                raise TransientToolError(f"MCP transport error calling {_name}: {type(exc).__name__}") from None
            if isinstance(out, dict) and out.get("ok") is False:
                raise ToolError(out.get("error_kind") or "failed", str(out.get("error")), hint=out.get("hint"))
            return out.get("result", out) if isinstance(out, dict) else out

        reg.register(Tool(prefix + t["name"], t["description"], AnyArgs, call, timeout_s=conn.call_timeout_s,
                          max_retries=1, schema_override=t["input_schema"], origin="mcp"))
    return reg

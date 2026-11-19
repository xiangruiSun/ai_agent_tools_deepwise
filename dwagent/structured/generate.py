"""Structured generation with a bounded repair loop.

One attempt = one model call. The reply is parsed as JSON, validated against the Pydantic
schema, then against the business rules. Any failure is fed back to the model as a numbered
list and the model tries again, up to ``max_attempts``. The function never returns an object
that failed validation: it returns ``value=None`` and the full error history instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.base import LLMClient, LLMError
from ..tools.registry import ToolContext

T = TypeVar("T", bound=BaseModel)


@dataclass
class Attempt:
    number: int
    stage: str                      # json | schema | rules | ok | llm_error
    errors: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class StructuredResult(Generic[T]):
    value: T | None
    attempts: list[Attempt]

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def first_pass(self) -> bool:
        return bool(self.attempts) and self.attempts[0].stage == "ok"

    @property
    def tokens(self) -> int:
        return sum(a.prompt_tokens + a.completion_tokens for a in self.attempts)


def extract_json(text: str) -> Any:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def generate_structured(llm: LLMClient, model: type[T], messages: list[dict[str, Any]],
                        rules: Callable[[T], list[str]] | None = None, max_attempts: int = 3,
                        constrained: bool = True, ctx: ToolContext | None = None,
                        span_name: str = "structured") -> StructuredResult[T]:
    schema = model.model_json_schema()
    msgs = list(messages)
    if not constrained:
        msgs.append({"role": "user", "content": "Reply with one JSON object that matches this JSON Schema, and "
                     "nothing else:\n" + json.dumps(schema)})
    attempts: list[Attempt] = []
    for n in range(1, max_attempts + 1):
        att = Attempt(number=n, stage="ok")
        attempts.append(att)
        try:
            if ctx is not None:
                with ctx.tracer.span(ctx.run_id, "llm", f"{span_name}.attempt{n}") as sp:
                    resp = llm.chat(msgs, json_schema=schema if constrained else None, temperature=0.0)
                    sp.prompt_tokens, sp.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
            else:
                resp = llm.chat(msgs, json_schema=schema if constrained else None, temperature=0.0)
        except LLMError as exc:
            att.stage, att.errors = "llm_error", [f"{exc.kind}: {exc}"]
            break
        att.prompt_tokens, att.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
        att.latency_ms = resp.latency_ms
        errors: list[str] = []
        value: T | None = None
        try:
            data = extract_json(resp.content)
        except (json.JSONDecodeError, ValueError) as exc:
            att.stage, errors = "json", [f"the reply was not valid JSON: {exc}"]
        else:
            try:
                value = model.model_validate(data)
            except ValidationError as exc:
                att.stage = "schema"
                errors = [f"{'.'.join(map(str, e['loc'])) or 'object'}: {e['msg']}" for e in exc.errors()]
            else:
                if rules is not None:
                    errors = rules(value)
                    if errors:
                        att.stage = "rules"
        att.errors = errors
        if not errors and value is not None:
            return StructuredResult(value, attempts)
        # Feed the exact problems back so the next attempt can repair them.
        msgs = msgs + [{"role": "assistant", "content": resp.content},
                       {"role": "user", "content": "Your JSON was rejected. Fix every problem below and reply with "
                        "the complete corrected JSON object only.\n"
                        + "\n".join(f"{i}. {e}" for i, e in enumerate(errors[:12], 1))}]
    return StructuredResult(None, attempts)

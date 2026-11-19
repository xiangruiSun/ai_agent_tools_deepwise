"""Hermetic tests of the agent core with a scripted model: no GPU, no network, no CT data.
They pin down the failure-mode behaviour (loops, hallucinated tools, bad arguments, timeouts,
retries, limits, approval, authorization, idempotency) so it cannot regress silently.
"""
from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from dwagent.agent.loop import AgentConfig, run_agent
from dwagent.agent.planner import Plan, topological_levels, validate_plan
from dwagent.llm.base import LLMError, ToolCall
from dwagent.llm.scripted import ScriptedLLM, call, say
from dwagent.memory.stores import LongTermMemory, SessionMemory
from dwagent.memory.working import fit_context
from dwagent.multi.blackboard import Blackboard, Conflict
from dwagent.runtime.graph import END, Env, Graph, Interrupt, Limits
from dwagent.runtime.trace import Tracer
from dwagent.store import Database
from dwagent.structured.generate import extract_json, generate_structured
from dwagent.tools.executor import AwaitingApproval, execute_tool
from dwagent.tools.registry import Principal, Tool, ToolContext, ToolError, ToolRegistry, TransientToolError


class Q(BaseModel):
    x: int


@pytest.fixture()
def env():
    db = Database(":memory:")
    tracer = Tracer(db)
    who = Principal("u1", "doctor", frozenset({"read", "write"}))
    state = {"calls": 0, "writes": 0}
    reg = ToolRegistry()

    def double(a: Q, c: ToolContext):
        state["calls"] += 1
        return {"value": a.x * 2}

    def flaky(a: Q, c: ToolContext):
        state["calls"] += 1
        if state["calls"] < 3:
            raise TransientToolError("upstream 503")
        return {"value": a.x}

    def slow(a: Q, c: ToolContext):
        time.sleep(1.0)
        return {"value": a.x}

    def write(a: Q, c: ToolContext):
        state["writes"] += 1
        return {"written": a.x}

    def own_only(a: Q, c: ToolContext):
        if a.x != 7:
            raise ToolError("unauthorized", "only record 7 is yours")

    reg.register(Tool("double", "double x", Q, double, scopes=frozenset({"read"})))
    reg.register(Tool("flaky", "fails twice", Q, flaky, max_retries=2))
    reg.register(Tool("slow", "too slow", Q, slow, timeout_s=0.1, max_retries=1))
    reg.register(Tool("write", "writes", Q, write, side_effect=True, idempotent=False, max_retries=0))
    reg.register(Tool("approve_me", "needs approval", Q, write, requires_approval=True, side_effect=True, idempotent=False))
    reg.register(Tool("admin", "admin only", Q, double, scopes=frozenset({"admin"})))
    reg.register(Tool("record", "per-record access", Q, double, authorize=own_only))
    ctx = ToolContext(who, db, tracer, run_id=tracer.start_run("test", "t"))
    return reg, ctx, state


def run(reg, ctx, steps, **cfg):
    return run_agent(ScriptedLLM(steps), reg, ctx, "task", AgentConfig(**cfg))


def ex(reg, ctx, name, **args):
    return execute_tool(reg, ToolCall("c1", name, args), ctx, sleep=lambda s: None)


# ---------------------------------------------------------------- the loop and its stopping conditions
def test_loop_runs_tool_then_answers(env):
    reg, ctx, _ = env
    st = run(reg, ctx, [call("double", x=4), say("The answer is 8.")])
    assert (st.status, st.stop_reason, st.iterations) == ("completed", "final_answer", 2)
    assert '"value": 8' in st.messages[3]["content"]


def test_max_iterations_stops_with_wrap_up(env):
    reg, ctx, _ = env
    llm = ScriptedLLM([lambda m, t: call("double", x=len(m)) if t else say("Best effort: unfinished.")])
    st = run_agent(llm, reg, ctx, "task", AgentConfig(max_iterations=3))
    assert (st.status, st.stop_reason, st.iterations) == ("stopped", "max_iterations", 3)
    assert "unfinished" in st.answer


def test_infinite_loop_is_detected(env):
    reg, ctx, state = env
    st = run(reg, ctx, [call("double", x=1)], max_iterations=20, finalize_on_limit=False)
    assert st.stop_reason == "loop_detected" and st.iterations == 3
    assert state["calls"] == 1          # repeats were served from the per-run cache
    assert any("already called" in m["content"] for m in st.messages if m["role"] == "user")


def test_token_budget_stops_the_run(env):
    reg, ctx, _ = env
    llm = ScriptedLLM([lambda m, t: call("double", x=len(m))])
    st = run_agent(llm, reg, ctx, "task", AgentConfig(token_budget=100, finalize_on_limit=False, max_iterations=50))
    assert st.stop_reason == "token_budget" and st.total_tokens >= 100


def test_llm_error_fails_cleanly(env):
    reg, ctx, _ = env
    st = run(reg, ctx, [LLMError("timeout", "model timed out")])
    assert (st.status, st.stop_reason) == ("failed", "llm_error")


def test_consecutive_tool_failures_stop_the_run(env):
    reg, ctx, _ = env
    llm = ScriptedLLM([lambda m, t: call("double", x="bad" + str(len(m)))])
    st = run_agent(llm, reg, ctx, "task", AgentConfig(max_consecutive_tool_failures=3, finalize_on_limit=False))
    assert st.stop_reason == "tool_failures"


def test_answer_check_sends_answer_back_then_flags_it(env):
    reg, ctx, _ = env
    cfg = AgentConfig(answer_check=lambda s, c: ["unsupported"] if "guess" in s.answer_candidate else [], max_answer_repairs=1)
    st = run_agent(ScriptedLLM([say("my guess"), say("grounded answer")]), reg, ctx, "task", cfg)
    assert st.answer == "grounded answer" and st.answer_repairs == 1
    st = run_agent(ScriptedLLM([say("my guess")]), reg, ctx, "task", cfg)
    assert st.stop_reason == "final_answer_unverified" and "Not verified" in st.answer


# ---------------------------------------------------------------- tool execution engineering
def test_hallucinated_tool_is_reported_not_raised(env):
    reg, ctx, _ = env
    st = run(reg, ctx, [call("launch_rocket", x=1), say("That tool does not exist.")])
    obs = st.messages[3]["content"]
    assert "unknown_tool" in obs and "double" in obs and st.status == "completed"


def test_tool_outside_the_offered_set_is_refused(env):
    reg, ctx, _ = env
    st = run(reg, ctx, [call("write", x=1), say("ok")], tool_names=["double"])
    assert "unknown_tool" in st.messages[3]["content"]


def test_invalid_arguments_return_the_validation_message(env):
    reg, ctx, _ = env
    r = ex(reg, ctx, "double", x="not a number")
    assert (r.ok, r.error_kind) == (False, "invalid_arguments") and "x" in r.error


def test_transient_failure_is_retried_with_backoff(env):
    reg, ctx, state = env
    r = ex(reg, ctx, "flaky", x=5)
    assert r.ok and r.attempts == 3 and state["calls"] == 3


def test_timeout_is_reported_and_retried_once(env):
    reg, ctx, _ = env
    r = ex(reg, ctx, "slow", x=1)
    assert (r.ok, r.error_kind, r.attempts) == (False, "timeout", 2)


def test_side_effect_runs_once_per_run(env):
    reg, ctx, state = env
    first, second = ex(reg, ctx, "write", x=3), ex(reg, ctx, "write", x=3)
    assert first.ok and second.ok and second.cached and state["writes"] == 1


def test_missing_scope_is_unauthorized_and_audited(env):
    reg, ctx, _ = env
    r = ex(reg, ctx, "admin", x=1)
    assert r.error_kind == "unauthorized"
    assert ctx.db.one("SELECT allowed FROM audit WHERE action='tool:admin'")["allowed"] == 0


def test_record_level_authorization(env):
    reg, ctx, _ = env
    assert ex(reg, ctx, "record", x=7).ok
    assert ex(reg, ctx, "record", x=8).error_kind == "unauthorized"


def test_approval_gate_pauses_then_runs_or_refuses(env):
    reg, ctx, state = env
    with pytest.raises(AwaitingApproval):
        ex(reg, ctx, "approve_me", x=1)
    assert state["writes"] == 0
    ctx.db.execute("UPDATE approvals SET status='approved'")
    assert ex(reg, ctx, "approve_me", x=1).ok and state["writes"] == 1
    with pytest.raises(AwaitingApproval):
        ex(reg, ctx, "approve_me", x=2)
    ctx.db.execute("UPDATE approvals SET status='rejected' WHERE status='pending'")
    assert ex(reg, ctx, "approve_me", x=2).error_kind == "rejected"


def test_agent_pauses_for_approval_and_resumes(env):
    reg, ctx, state = env
    llm = ScriptedLLM([call("approve_me", x=9), say("done")])
    st = run_agent(llm, reg, ctx, "task", AgentConfig())
    assert st.status == "awaiting_approval" and st.pending_approval["tool"] == "approve_me"
    ctx.db.execute("UPDATE approvals SET status='approved'")
    st = run_agent(llm, reg, ctx, config=AgentConfig(), state=st)
    assert st.status == "completed" and state["writes"] == 1


def test_trace_records_latency_tokens_and_errors(env):
    reg, ctx, _ = env
    run(reg, ctx, [call("double", x="bad"), call("double", x=2), say("4")])
    s = ctx.tracer.summary(ctx.run_id)
    assert s["llm_calls"] == 3 and s["tool_calls"] == 2 and s["tool_errors"] == 1 and s["total_tokens"] == 180
    assert s["errors_by_kind"] == {"invalid_arguments": 1}
    assert "input" not in ctx.tracer.redacted_events(ctx.run_id)[0]


# ---------------------------------------------------------------- structured output
class Out(BaseModel):
    n: int


def test_structured_retry_feeds_errors_back():
    llm = ScriptedLLM([say("not json"), say('{"n": "x"}'), say('```json\n{"n": 3}\n```'), say('{"n": 4}')])
    res = generate_structured(llm, Out, [{"role": "user", "content": "go"}], rules=lambda o: ["n must be even"] if o.n % 2 else [],
                              max_attempts=4, constrained=False)
    assert res.ok and res.value.n == 4 and [a.stage for a in res.attempts] == ["json", "schema", "rules", "ok"]
    assert "n must be even" in llm.calls[3][-1]["content"]


def test_structured_gives_up_after_the_bound():
    res = generate_structured(ScriptedLLM([say("nope")]), Out, [{"role": "user", "content": "go"}], max_attempts=2, constrained=False)
    assert not res.ok and len(res.attempts) == 2


def test_extract_json_tolerates_prose():
    assert extract_json('Here you go: {"a": 1} hope it helps') == {"a": 1}


# ---------------------------------------------------------------- planning
def _plan(steps):
    return Plan.model_validate({"goal": "g", "steps": steps})


def test_plan_validation_rejects_hallucinated_tool_cycle_and_bad_dependency(env):
    reg, _, _ = env
    allowed = {"double"}
    ok = _plan([{"id": "s1", "purpose": "first step", "tool": "double", "arguments": {"x": 1}},
                {"id": "s2", "purpose": "second step", "tool": "double", "arguments": {"x": 2}, "depends_on": ["s1"]}])
    assert validate_plan(ok, allowed, reg) == []
    assert [[s.id for s in lvl] for lvl in topological_levels(ok)] == [["s1"], ["s2"]]
    bad_tool = _plan([{"id": "s1", "purpose": "invented tool", "tool": "order_biopsy", "arguments": {}}])
    assert "does not exist" in validate_plan(bad_tool, allowed, reg)[0]
    bad_args = _plan([{"id": "s1", "purpose": "wrong args", "tool": "double", "arguments": {"y": 1}}])
    assert "arguments do not match" in validate_plan(bad_args, allowed, reg)[0]
    bad_dep = _plan([{"id": "s1", "purpose": "bad dependency", "tool": "double", "arguments": {"x": 1}, "depends_on": ["s9"]}])
    assert "unknown step" in validate_plan(bad_dep, allowed, reg)[0]
    cycle = _plan([{"id": "s1", "purpose": "cycle a", "tool": "double", "arguments": {"x": 1}, "depends_on": ["s2"]},
                   {"id": "s2", "purpose": "cycle b", "tool": "double", "arguments": {"x": 1}, "depends_on": ["s1"]}])
    assert validate_plan(cycle, allowed, reg) == ["the dependencies contain a cycle"]


# ---------------------------------------------------------------- runtime graph
def test_graph_routes_checkpoints_interrupts_and_resumes(env):
    reg, ctx, _ = env
    seen = []
    g = Graph("t")
    g.node("a", lambda s, e: seen.append("a") or {"n": s["n"] + 1})
    g.node("gate", lambda s, e: (_ for _ in ()).throw(Interrupt("wait")) if not s.get("ok") else {"gated": True})
    g.node("big", lambda s, e: {"path": "big"}).node("small", lambda s, e: {"path": "small"})
    g.edge("a", "gate").route("gate", lambda s: "big" if s["n"] > 5 else "small", {"big", "small"}).edge("big", END).edge("small", END)
    r = g.run(Env(ctx), {"n": 9})
    assert (r.status, r.next_node, seen) == ("interrupted", "gate", ["a"])
    ctx.db.execute("UPDATE checkpoints SET state=json_set(state,'$.ok',1) WHERE seq=(SELECT MAX(seq) FROM checkpoints)")
    r = g.run(Env(ctx), resume=True)
    assert (r.status, r.state["path"], seen) == ("completed", "big", ["a"])     # node a did not run again


def test_graph_failure_keeps_checkpoint_and_step_limit(env):
    reg, ctx, _ = env
    box = {"fail": True}

    def node(s, e):
        if box["fail"]:
            raise RuntimeError("boom")
        return {"done": True}

    g = Graph("f").node("first", lambda s, e: {"x": 1}).node("second", node).edge("first", "second").edge("second", END)
    r = g.run(Env(ctx), {}, limits=Limits(max_node_attempts=1))
    assert (r.status, r.next_node) == ("failed", "second")
    box["fail"] = False
    assert g.run(Env(ctx), resume=True).state["done"] is True
    loop = Graph("l").node("a", lambda s, e: {}).edge("a", "a")
    ctx.run_id = ctx.tracer.start_run("test", "loop")
    assert g is not loop and loop.run(Env(ctx), {}, limits=Limits(max_steps=5)).detail == "max_steps"


# ---------------------------------------------------------------- memory and shared state
def test_memory_is_per_user_and_bounded():
    db = Database(":memory:")
    ltm = LongTermMemory(db)
    ltm.remember("alice", "Alice prefers answers in two sentences.")
    assert [m["text"] for m in ltm.recall("alice", "how long should answers be")] == ["Alice prefers answers in two sentences."]
    assert ltm.recall("bob", "answers") == []
    s = SessionMemory(db, "alice:s1")
    for i in range(10):
        s.append("user", f"q{i}"), s.append("assistant", f"a{i}")
    assert [m["content"] for m in s.history(max_turns=2)] == ["q8", "a8", "q9", "a9"]


def test_working_context_fits_budget_and_keeps_task():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "THE TASK"}]
    for i in range(12):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "name": "t", "arguments": {}}]},
                 {"role": "tool", "name": "t", "content": "x" * 2000}]
    fitted = fit_context(msgs, 1500, anchor_task="THE TASK", iteration=4)
    from dwagent.llm.base import messages_tokens
    assert messages_tokens(fitted) <= 1500 + 60 < messages_tokens(msgs) / 3      # budget plus the anchor message
    assert fitted[1]["content"] == "THE TASK" and fitted[-2]["content"] == "x" * 2000 and "Reminder" in fitted[-1]["content"]


def test_blackboard_detects_conflicting_writes():
    b = Blackboard()
    _, v = b.read("k")
    b.write("k", "first", "a", expected_version=v)
    with pytest.raises(Conflict):
        b.write("k", "second", "b", expected_version=v)     # b read before a wrote
    assert b.append("list", 1, "a") == 1 and b.append("list", 2, "b") == 2 and b.read("list")[0] == [1, 2]

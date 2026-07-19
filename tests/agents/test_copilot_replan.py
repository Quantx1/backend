import asyncio

import backend.ai.agents.copilot as cp
from backend.ai.agents.state import AgentState


def _state(plan, plan_source, results, follow_up=False):
    s = AgentState(inputs={"message": "x"}, user_id="u", graph_name="copilot")
    s.put("tool_planner", plan=plan, plan_source=plan_source, follow_up=follow_up)
    s.put("tool_caller", tool_results=results)
    return s


def test_needs_more_false_for_regex_fastpaths():
    assert cp.needs_more(_state([], "regex_greeting", [])) is False
    assert cp.needs_more(_state([], "regex_pure_knowledge", [])) is False


def test_needs_more_true_on_tool_error():
    s = _state([{"tool": "get_signal", "args": {}}], "llm",
               [{"tool": "get_signal", "args": {}, "result": {"error": "not found"}}])
    assert cp.needs_more(s) is True


def test_needs_more_true_when_planned_but_no_results():
    assert cp.needs_more(_state([{"tool": "get_signal", "args": {}}], "llm", [])) is True


def test_needs_more_true_on_follow_up_flag():
    s = _state([{"tool": "get_signal", "args": {}}], "llm",
               [{"tool": "get_signal", "args": {}, "result": {"signal": {"id": 1}}}],
               follow_up=True)
    assert cp.needs_more(s) is True


def test_needs_more_false_on_clean_success():
    s = _state([{"tool": "get_signal", "args": {}}], "llm",
               [{"tool": "get_signal", "args": {}, "result": {"signal": {"id": 1}}}])
    assert cp.needs_more(s) is False


def test_replan_loop_hard_caps_at_two_rounds(monkeypatch):
    calls = {"plan": 0, "call": 0}

    class _P:
        async def run(self, state):
            calls["plan"] += 1
            state.put("tool_planner", plan=[{"tool": "get_signal", "args": {}}],
                      plan_source="llm_replan", follow_up=False)

    class _C:
        async def run(self, state):
            calls["call"] += 1
            prior = state.get("tool_caller", "tool_results") or []
            state.put("tool_caller", tool_results=prior + [
                {"tool": "get_signal", "args": {}, "result": {"error": "still bad"}}])

    s = _state([{"tool": "get_signal", "args": {}}], "llm",
               [{"tool": "get_signal", "args": {}, "result": {"error": "bad"}}])
    monkeypatch.setattr(cp, "_replan_affordable", lambda: True)
    asyncio.run(cp._replan_loop(s, _P(), _C()))
    assert calls["plan"] == 2 and calls["call"] == 2


def test_replan_loop_skips_when_over_budget(monkeypatch):
    monkeypatch.setattr(cp, "_replan_affordable", lambda: False)

    class _P:
        async def run(self, state):
            raise AssertionError("planner must not run when over budget")

    class _C:
        async def run(self, state):
            raise AssertionError("caller must not run when over budget")

    s = _state([{"tool": "get_signal", "args": {}}], "llm",
               [{"tool": "get_signal", "args": {}, "result": {"error": "bad"}}])
    asyncio.run(cp._replan_loop(s, _P(), _C()))

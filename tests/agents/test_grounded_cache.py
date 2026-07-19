import backend.ai.agents.grounded as g
import backend.ai.agents.response_cache as rc


def setup_function(_):
    rc._L1.clear()


# grounded_reason discards implausibly short output (< 40 chars), so the mock
# answer must be a realistic length or it is filtered to None before caching.
_ANSWER = "The stock rallied on heavy volume with strong momentum this week."


def test_grounded_reason_serves_from_cache_without_calling_llm(monkeypatch):
    calls = {"n": 0}

    def _fake_complete_sync(*_a, **_k):
        calls["n"] += 1
        return _ANSWER

    monkeypatch.setattr("backend.ai.agents.llm.complete_sync", _fake_complete_sync)
    monkeypatch.setattr(rc, "_sb", lambda: None)

    out1 = g.grounded_reason({"x": 1}, "why?", cache_key="t:2026-06-10")
    out2 = g.grounded_reason({"x": 1}, "why?", cache_key="t:2026-06-10")
    assert out1 == _ANSWER
    assert out2 == _ANSWER
    assert calls["n"] == 1


def test_grounded_reason_returns_none_on_empty(monkeypatch):
    monkeypatch.setattr("backend.ai.agents.llm.complete_sync", lambda *a, **k: "")
    monkeypatch.setattr(rc, "_sb", lambda: None)
    assert g.grounded_reason({"x": 1}, "why?", cache_key="t2:2026-06-10") is None

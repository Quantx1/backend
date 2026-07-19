import backend.ai.agents.conversation_memory as cm


def test_load_memory_empty_without_db(monkeypatch):
    monkeypatch.setattr(cm, "_sb", lambda: None)
    assert cm.load_memory("u1") == {"summary": "", "turns_summarized": 0}


def test_load_memory_empty_for_blank_user():
    assert cm.load_memory("") == {"summary": "", "turns_summarized": 0}


def test_refresh_noop_below_threshold(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr("backend.ai.agents.llm.complete_sync",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or "x")
    cm.maybe_refresh_memory(user_id="u1", total_turns=3, recent_turns=[],
                            prev_summary="", prev_turns_summarized=0)
    assert called["n"] == 0


def test_refresh_fires_at_threshold(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr("backend.ai.agents.llm.complete_sync",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or "new summary")
    monkeypatch.setattr(cm, "_sb", lambda: None)
    cm.maybe_refresh_memory(user_id="u1", total_turns=6,
                            recent_turns=[{"role": "user", "content": "hi"}],
                            prev_summary="", prev_turns_summarized=0)
    assert called["n"] == 1

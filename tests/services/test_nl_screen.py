"""NL screener — free rule fast-path (deterministic). LLM path is integration-only."""
import pytest

import backend.ai.agents.response_cache as rc
import backend.services.screener_v2.nl_screen as nl_screen
from backend.services.screener_v2.nl_screen import (
    parse_rules, resolve_screen_query, scanner_label,
)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    """No test touches Supabase; L1 is cleared per test."""
    rc._L1.clear()
    monkeypatch.setattr(rc, "_sb", lambda: None)


def test_parse_rules_maps_common_vocab():
    ids = parse_rules("oversold stocks with rising volume")
    assert 9 in ids                       # oversold
    assert 8 in ids or 4 in ids           # volume


def test_resolve_strong_rule_match_skips_llm():
    r = resolve_screen_query("52 week high with a volume surge", allow_llm=False)
    assert r["source"] == "rules"
    assert 5 in r["scanner_ids"]
    assert 8 in r["scanner_ids"] or 4 in r["scanner_ids"]


def test_resolve_unrecognized_is_honest():
    r = resolve_screen_query("qwertz nonsense", allow_llm=False)
    assert r["scanner_ids"] == []


def test_scanner_label():
    assert "52 Week High" in scanner_label(5)
    assert scanner_label(99999).startswith("Scanner")


def test_llm_path_caches_persistently(monkeypatch):
    calls = {"n": 0}

    def _fake_resolve(_norm, *, user_id=None):
        calls["n"] += 1
        return [7, 12]

    monkeypatch.setattr(nl_screen, "llm_resolve", _fake_resolve)
    r1 = resolve_screen_query("qwertz nuanced multi concept ask")
    assert r1["source"] == "llm"
    assert r1["scanner_ids"] == [7, 12]
    r2 = resolve_screen_query("qwertz nuanced multi concept ask")
    assert r2["source"] == "cache"
    assert r2["scanner_ids"] == [7, 12]
    assert calls["n"] == 1


def test_empty_llm_result_is_not_cached(monkeypatch):
    calls = {"n": 0}

    def _fake_resolve(_norm, *, user_id=None):
        calls["n"] += 1
        return []

    monkeypatch.setattr(nl_screen, "llm_resolve", _fake_resolve)
    r1 = resolve_screen_query("qwertz nothing matches here")
    r2 = resolve_screen_query("qwertz nothing matches here")
    assert r1["source"] == "rules"
    assert r2["source"] == "rules"
    assert calls["n"] == 2   # no negative caching — the LLM retries


def test_strong_rule_match_never_consults_cache(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("cache_get must not run on the rule fast-path")

    monkeypatch.setattr(rc, "cache_get", _boom)
    r = resolve_screen_query("52 week high with a volume surge", allow_llm=False)
    assert r["source"] == "rules"
    assert 5 in r["scanner_ids"]

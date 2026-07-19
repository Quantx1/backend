"""Multi-source providers + multi-model sentiment ensemble + finance lexicon."""
import backend.ai.sentiment.news_providers as np
from backend.ai.sentiment.finance_lexicon import lexicon_label, lexicon_score
from backend.ai.sentiment.sentiment_ensemble import cross_check


# ── finance lexicon ─────────────────────────────────────────────────────────


def test_lexicon_bullish_bearish_neutral():
    assert lexicon_score("Reliance profit jumps, beats estimates") > 0
    assert lexicon_score("Yes Bank crashes on SEBI probe and fraud") < 0
    assert lexicon_score("Company holds annual general meeting") == 0.0
    assert lexicon_label("TCS wins record order") == "positive"
    assert lexicon_label("Infosys downgrade on weak guidance") == "negative"


# ── ensemble cross-check ─────────────────────────────────────────────────────


def test_cross_check_agreement_llm_and_lexicon(monkeypatch):
    # FinBERT not loaded → 2-model ensemble (llm + lexicon).
    monkeypatch.setattr("backend.ai.sentiment.sentiment_ensemble._finbert_scores", lambda titles: None)
    titles = ["ACME profit surges beats estimates", "ACME hit by fraud probe penalty"]
    llm = [1.0, -1.0]
    out = cross_check(titles, llm)
    assert out[0]["consensus"] == "positive" and out[0]["models_agree"] == 2
    assert out[1]["consensus"] == "negative" and out[1]["models_agree"] == 2
    assert out[0]["finbert"] is None
    assert out[0]["models_total"] == 2


def test_cross_check_disagreement(monkeypatch):
    monkeypatch.setattr("backend.ai.sentiment.sentiment_ensemble._finbert_scores", lambda titles: None)
    # LLM says positive but the headline lexicon is clearly negative → split.
    titles = ["ACME crashes on fraud probe"]
    out = cross_check(titles, [1.0])
    # one +1 (llm), one -1 (lexicon) → tie broken to positive (pos>=neg), agree=1
    assert out[0]["models_total"] == 2
    assert out[0]["models_agree"] == 1


def test_cross_check_finbert_votes_when_loaded(monkeypatch):
    monkeypatch.setattr("backend.ai.sentiment.sentiment_ensemble._finbert_scores",
                        lambda titles: [0.8 for _ in titles])
    out = cross_check(["ACME profit surges"], [1.0])
    assert out[0]["finbert"] == 0.8
    assert out[0]["models_total"] == 3 and out[0]["models_agree"] == 3


# ── provider fan-out + merge ────────────────────────────────────────────────


async def test_fetch_all_sources_merges_and_failopen(monkeypatch):
    async def _g(sym, days, lim):
        return [{"title": "g1", "source": "ET", "link": "a", "published": None, "provider": "google"}]

    async def _gd(sym, days, lim):
        raise RuntimeError("gdelt down")  # one provider fails → must not break

    async def _y(sym, days, lim):
        return [{"title": "y1", "source": "Yahoo", "link": "b", "published": None, "provider": "yahoo"}]

    monkeypatch.setattr(np, "_PROVIDERS", {"google": _g, "gdelt": _gd, "yahoo": _y})
    monkeypatch.setenv("NEWS_PROVIDERS", "google,gdelt,yahoo")
    rows = await np.fetch_all_sources("ACME")
    titles = {r["title"] for r in rows}
    assert titles == {"g1", "y1"}  # gdelt failed open, the other two merged


async def test_fetch_all_sources_empty_symbol():
    assert await np.fetch_all_sources("") == []


def test_enabled_providers_env(monkeypatch):
    monkeypatch.setenv("NEWS_PROVIDERS", "google, yahoo ,gdelt")
    assert np.enabled_providers() == ["google", "yahoo", "gdelt"]


def test_gdelt_date_parse():
    assert np._gdelt_date("20260614T101500Z") == "2026-06-14T10:15:00+00:00"
    assert np._gdelt_date("") == ""

"""News Digest — deterministic drivers (the grounded narrative is
integration-only). Pure tests over `build_drivers` with fabricated facts."""
from backend.services.news.news_digest import build_drivers


def test_drivers_full_facts():
    facts = {
        "symbol": "TCS",
        "mood": {"mean_score": 0.42, "label": "bullish", "headline_count": 6,
                 "positive": 4, "negative": 1, "neutral": 1,
                 "headlines": [], "sources": ["Economic Times"]},
        "mood_prior": {"mean_score": 0.27, "trade_date": "2026-06-10", "headline_count": 9},
        "price": {"ltp": 4012.0, "change_pct": 2.1},
    }
    txt = " ".join(build_drivers(facts))
    assert "6 headlines in the last 3 days: 4 positive / 1 neutral / 1 negative." in txt
    assert "News mood bullish (+0.42" in txt
    assert "Mood improving vs 2026-06-10 (+0.27 → +0.42)." in txt
    assert "Price +2.1% today — trading with the news." in txt


def test_drivers_deteriorating_and_diverging():
    facts = {
        "mood": {"mean_score": -0.3, "label": "bearish", "headline_count": 5,
                 "positive": 1, "negative": 3, "neutral": 1},
        "mood_prior": {"mean_score": 0.1, "trade_date": "2026-06-10"},
        "price": {"change_pct": 1.4},
    }
    txt = " ".join(build_drivers(facts))
    assert "Mood deteriorating vs 2026-06-10" in txt
    assert "diverging from the news" in txt


def test_drivers_steady_below_epsilon_and_neutral_no_price_line():
    facts = {
        "mood": {"mean_score": 0.05, "label": "neutral", "headline_count": 3,
                 "positive": 1, "negative": 1, "neutral": 1},
        "mood_prior": {"mean_score": 0.02, "trade_date": "2026-06-10"},
        "price": {"change_pct": -0.8},
    }
    txt = " ".join(build_drivers(facts))
    assert "Mood steady vs the prior day." in txt
    assert "the news" not in txt


def test_drivers_honest_empty():
    assert build_drivers({}) == []
    assert build_drivers({"symbol": "X", "price": {"change_pct": 3.0}}) == []
    assert build_drivers({"mood": {"headline_count": 0}}) == []

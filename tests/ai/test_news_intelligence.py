"""News Intelligence — pure dedup/enrich/aggregate + orchestrator (monkeypatched LLM)."""
from datetime import datetime, timedelta, timezone

import backend.services.news.news_intelligence as ni
from backend.ai.sentiment.news_dedup import (
    cluster_headlines, recency_weight, source_tier, source_weight, urgency,
)
from backend.ai.sentiment.news_enrich import IMPACT_WEIGHT, _coerce_row
from backend.services.news.news_intelligence import detect_thesis_change


def _now():
    return datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)


def _iso(hours_ago):
    return (_now() - timedelta(hours=hours_ago)).isoformat()


# ── source credibility ─────────────────────────────────────────────────────


def test_source_tiers_and_weights():
    assert source_tier("The Economic Times") == 1
    assert source_tier("Moneycontrol") == 1
    assert source_tier("Investing.com") == 2
    assert source_tier("RandomBlog") == 3
    assert source_tier("") == 3
    assert source_weight("Reuters") > source_weight("RandomBlog")


# ── urgency + recency ───────────────────────────────────────────────────────


def test_urgency_buckets():
    assert urgency(_iso(1), now=_now()) == "breaking"
    assert urgency(_iso(10), now=_now()) == "recent"
    assert urgency(_iso(30), now=_now()) == "today"
    assert urgency(_iso(100), now=_now()) == "older"
    assert urgency(None, now=_now()) == "older"


def test_recency_weight_decays():
    fresh = recency_weight(_iso(0), now=_now())
    old = recency_weight(_iso(48), now=_now())
    assert fresh > old
    assert recency_weight(None, now=_now()) == 0.5  # undated neutral


# ── clustering / dedup ──────────────────────────────────────────────────────


def test_cluster_merges_near_duplicates():
    heads = [
        {"title": "Reliance Q3 profit jumps beating estimates", "source": "Economic Times", "published": _iso(2)},
        {"title": "Reliance Q3 profit jumps, beats estimates", "source": "Moneycontrol", "published": _iso(3)},
        {"title": "Reliance Q3 profit rises beating estimates", "source": "Mint", "published": _iso(4)},
        {"title": "Tata Motors launches new EV model in India", "source": "Reuters", "published": _iso(1)},
    ]
    clusters = cluster_headlines(heads)
    # 3 near-dup Reliance stories collapse to 1; Tata is its own → 2 clusters
    assert len(clusters) == 2
    reliance = max(clusters, key=lambda c: c["member_count"])
    assert reliance["member_count"] == 3
    assert len(reliance["sources"]) == 3  # 3 distinct outlets corroborate


def test_cluster_representative_prefers_tier1_then_fresh():
    heads = [
        {"title": "Infosys wins big deal in Europe", "source": "RandomBlog", "published": _iso(1)},
        {"title": "Infosys wins big deal Europe", "source": "Reuters", "published": _iso(5)},
    ]
    clusters = cluster_headlines(heads)
    assert len(clusters) == 1
    assert clusters[0]["source"] == "Reuters"  # tier-1 beats fresher blog


# ── enrich coercion (no LLM) ────────────────────────────────────────────────


def test_coerce_row_valid():
    r = _coerce_row({"sentiment": "positive", "event_type": "earnings", "impact": "high"})
    assert r["label"] == "positive" and r["event_type"] == "earnings" and r["impact"] == "high"
    assert r["score"] == IMPACT_WEIGHT["high"]  # +1 * high weight


def test_coerce_row_negative_sign_and_unknown_enum():
    r = _coerce_row({"sentiment": "negative", "event_type": "zzz", "impact": "weird"})
    assert r["label"] == "negative" and r["event_type"] == "other" and r["impact"] == "low"
    assert r["score"] == -IMPACT_WEIGHT["low"]


def test_coerce_row_garbage_is_neutral():
    r = _coerce_row("not a dict")
    assert r["label"] == "neutral" and r["score"] == 0.0


# ── materiality-weighted mood ──────────────────────────────────────────────


def test_materiality_weighting_high_impact_dominates():
    # one high-impact bearish (tier-1, fresh) vs two low-impact bullish → bearish
    stories = [
        {"score": -1.0, "impact": "high", "source": "Reuters", "published": _iso(1)},
        {"score": 0.25, "impact": "low", "source": "RandomBlog", "published": _iso(1)},
        {"score": 0.25, "impact": "low", "source": "RandomBlog", "published": _iso(1)},
    ]
    mood = ni._materiality_weighted_mood(stories)
    assert mood is not None and mood < 0  # the material bearish story wins


# ── orchestrator analyze() with monkeypatched fetch + enrich ───────────────


async def test_analyze_full(monkeypatch):
    async def _fetch(sym, lookback_days=3, max_per_source=20):
        return [
            {"title": "ACME wins large defence order worth 5000 cr", "source": "Economic Times", "link": "u1", "published": _iso(2), "provider": "google"},
            {"title": "ACME bags defence order 5000 crore", "source": "Mint", "link": "u2", "published": _iso(3), "provider": "gdelt"},
            {"title": "ACME faces SEBI probe over disclosures", "source": "Reuters", "link": "u3", "published": _iso(1), "provider": "yahoo"},
        ]
    monkeypatch.setattr("backend.ai.sentiment.news_providers.fetch_all_sources", _fetch)

    def _enrich(titles, **k):
        # order-win bullish high; probe bearish high
        out = []
        for t in titles:
            if "SEBI" in t:
                out.append({"label": "negative", "score": -1.0, "confidence": 1.0, "event_type": "regulatory", "impact": "high"})
            else:
                out.append({"label": "positive", "score": 1.0, "confidence": 1.0, "event_type": "order_win", "impact": "high"})
        return out
    monkeypatch.setattr(ni, "enrich_headlines", _enrich)

    res = await ni.analyze("ACME", use_llm=True, use_narrative=False)
    assert res["available"] is True
    assert res["raw_headline_count"] == 3
    assert res["story_count"] == 2  # two order-win headlines deduped across sources
    assert res["mood_score"] is not None
    events = {e["event"] for e in res["event_breakdown"]}
    assert "Order win" in events and "Regulatory" in events
    assert res["impact_counts"]["high"] == 2
    assert res["top_story"] is not None
    # multi-source + multi-model surfaced
    assert set(res["providers"]) == {"google", "gdelt", "yahoo"}
    assert "llm" in res["models"] and "lexicon" in res["models"]
    assert res["stories"][0]["agreement"]["models_total"] >= 1


async def test_analyze_honest_empty(monkeypatch):
    async def _fetch(sym, lookback_days=3, max_per_source=20):
        return []
    monkeypatch.setattr("backend.ai.sentiment.news_providers.fetch_all_sources", _fetch)
    res = await ni.analyze("ZZZ")
    assert res["available"] is False
    assert res["mood_score"] is None and res["stories"] == []


# ── thesis-change detection ─────────────────────────────────────────────────


def _intel(stories):
    return {"available": True, "symbol": "ACME", "stories": stories}


def test_thesis_change_long_threatened_by_high_impact_bearish():
    intel = _intel([{"title": "SEBI probe", "label": "bearish", "impact": "high", "event_label": "Regulatory", "link": "u"}])
    alert = detect_thesis_change("LONG", intel)
    assert alert and alert["at_risk"] is True and alert["severity"] == "high"
    assert alert["position"] == "LONG"


def test_thesis_change_none_when_news_supports():
    intel = _intel([{"title": "order win", "label": "bullish", "impact": "high", "event_label": "Order win", "link": "u"}])
    assert detect_thesis_change("LONG", intel) is None


def test_thesis_change_short_threatened_by_bullish():
    intel = _intel([{"title": "order win", "label": "bullish", "impact": "medium", "event_label": "Order win", "link": "u"}])
    alert = detect_thesis_change("SHORT", intel)
    assert alert and alert["position"] == "SHORT" and alert["severity"] == "medium"


def test_thesis_change_ignores_low_impact():
    intel = _intel([{"title": "minor note", "label": "bearish", "impact": "low", "event_label": "General", "link": "u"}])
    assert detect_thesis_change("LONG", intel) is None

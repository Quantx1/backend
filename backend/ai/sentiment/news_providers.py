"""
Multi-source news aggregation — fan out across all the FREE/open news APIs,
merge into one stream, let the dedup-clusterer collapse cross-source repeats.

The audit's "single-source dependency" gap: live sentiment ran on Google News
RSS alone. This layer adds more free, no-key sources and merges them:

  * google   — Google News RSS (the existing canonical fetcher)
  * gdelt    — GDELT 2.0 DOC API (open, no key, global news graph)
  * yahoo    — Yahoo Finance per-ticker news (via yfinance, already a dep)
  * rss      — configured publisher RSS/Atom feeds (ET/Mint/MoneyControl/CNBC),
               filtered to headlines that mention the symbol

Every provider is independently toggleable (NEWS_PROVIDERS env), time-bounded,
and FAILS OPEN (returns [] on any error) so one flaky source never breaks the
feed. Output is the canonical headline shape {title, source, link, published}
— the same shape news_dedup.cluster_headlines + news_enrich consume, so adding
a source needs no downstream change.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDERS = "google,gdelt,yahoo"
_HTTP_TIMEOUT = 8.0


def enabled_providers() -> List[str]:
    raw = os.getenv("NEWS_PROVIDERS", _DEFAULT_PROVIDERS)
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


# ─────────────────────────────── providers ────────────────────────────────


async def _google(symbol: str, lookback_days: int, limit: int) -> List[Dict[str, Any]]:
    from .news_fetcher import fetch_headlines
    rows = await fetch_headlines(symbol, lookback_days=lookback_days, max_items=limit)
    for r in rows:
        r.setdefault("provider", "google")
    return rows


async def _gdelt(symbol: str, lookback_days: int, limit: int) -> List[Dict[str, Any]]:
    """GDELT 2.0 DOC API — open, no key. Article list for the symbol."""
    import httpx
    query = f'"{symbol}" (stock OR shares OR NSE OR Sensex)'
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(min(limit, 50)),
        "format": "json",
        "timespan": f"{max(1, lookback_days)}d",
        "sort": "DateDesc",
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return []
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("gdelt fetch failed for %s: %s", symbol, exc)
        return []
    out: List[Dict[str, Any]] = []
    for a in (data.get("articles") or [])[:limit]:
        title = (a.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "source": (a.get("domain") or "GDELT").strip(),
            "link": a.get("url"),
            "published": _gdelt_date(a.get("seendate")),
            "provider": "gdelt",
        })
    return out


def _gdelt_date(seendate: Any) -> Any:
    # GDELT seendate is like '20260614T101500Z' → ISO.
    s = str(seendate or "")
    if len(s) >= 15 and s[8] == "T":
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}+00:00"
    return seendate


async def _yahoo(symbol: str, lookback_days: int, limit: int) -> List[Dict[str, Any]]:
    """Yahoo Finance per-ticker news via yfinance (blocking → off-thread)."""
    def _pull() -> List[Dict[str, Any]]:
        try:
            import yfinance as yf
            tk = yf.Ticker(symbol if symbol.endswith(".NS") else f"{symbol}.NS")
            raw = getattr(tk, "news", None) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("yahoo news failed for %s: %s", symbol, exc)
            return []
        rows: List[Dict[str, Any]] = []
        for item in raw[:limit]:
            # yfinance shape varies by version: flat, or nested under 'content'.
            c = item.get("content") if isinstance(item, dict) else None
            if isinstance(c, dict):
                title = (c.get("title") or "").strip()
                link = ((c.get("canonicalUrl") or {}) or {}).get("url") or ((c.get("clickThroughUrl") or {}) or {}).get("url")
                source = ((c.get("provider") or {}) or {}).get("displayName") or "Yahoo Finance"
                published = c.get("pubDate") or c.get("displayTime")
            else:
                title = (item.get("title") or "").strip()
                link = item.get("link")
                source = item.get("publisher") or "Yahoo Finance"
                ts = item.get("providerPublishTime")
                published = _epoch_iso(ts) if ts else None
            if not title:
                continue
            rows.append({"title": title, "source": source, "link": link,
                         "published": published, "provider": "yahoo"})
        return rows
    return await asyncio.to_thread(_pull)


def _epoch_iso(ts: Any) -> Any:
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        return None


async def _publisher_rss_impl(symbol: str, lookback_days: int, limit: int) -> List[Dict[str, Any]]:
    """Configured publisher RSS/Atom feeds (ET/Mint/MoneyControl/CNBC), filtered
    to the symbol. Reuses the assistant's dual RSS/Atom parser. These are
    market-wide feeds → keep only headlines that mention the symbol token
    (high precision, low recall)."""
    import httpx
    try:
        from ...core.config import settings
        from ...services.assistant.news_context import NewsContextService
    except Exception as exc:  # noqa: BLE001
        logger.debug("publisher_rss import failed: %s", exc)
        return []
    feeds = [u.strip() for u in str(getattr(settings, "ASSISTANT_NEWS_FEEDS", "")).split(",") if u.strip()]
    if not feeds:
        return []
    svc = NewsContextService()
    sym = symbol.lower()
    out: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            results = await asyncio.gather(
                *[svc._fetch_feed(client, u) for u in feeds], return_exceptions=True
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("publisher_rss gather failed: %s", exc)
        return []
    for res in results:
        if isinstance(res, Exception):
            continue
        for art in res or []:
            title = (getattr(art, "title", "") or "").strip()
            if not title or sym not in title.lower():
                continue
            out.append({
                "title": title,
                "source": getattr(art, "source", "") or "RSS",
                "link": getattr(art, "url", None),
                "published": getattr(art, "published_at", None),
                "provider": "rss",
            })
    return out[:limit]


_PROVIDERS = {
    "google": _google,
    "gdelt": _gdelt,
    "yahoo": _yahoo,
    "rss": _publisher_rss_impl,
}


async def fetch_all_sources(
    symbol: str,
    *,
    lookback_days: int = 3,
    max_per_source: int = 20,
) -> List[Dict[str, Any]]:
    """Fan out across all enabled free providers concurrently and merge.

    Cross-source de-duplication is handled downstream by
    ``news_dedup.cluster_headlines`` (token-overlap), so duplicates across
    Google/GDELT/Yahoo/RSS collapse into single corroborated stories.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return []
    provs = [p for p in enabled_providers() if p in _PROVIDERS]
    if not provs:
        provs = ["google"]
    coros = [_PROVIDERS[p](sym, lookback_days, max_per_source) for p in provs]
    results = await asyncio.gather(*coros, return_exceptions=True)

    merged: List[Dict[str, Any]] = []
    by_provider: Dict[str, int] = {}
    for p, res in zip(provs, results):
        if isinstance(res, Exception) or not res:
            by_provider[p] = 0
            continue
        by_provider[p] = len(res)
        merged.extend(res)
    if by_provider:
        logger.info("news providers for %s: %s", sym, by_provider)
    return merged


__all__ = ["fetch_all_sources", "enabled_providers"]

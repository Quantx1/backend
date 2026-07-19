"""
Per-symbol news headline fetcher — Google News RSS (India edition).

Wraps Google News RSS (India edition). Standalone — no dependency on the
old ``services/sentiment_engine`` module (now deleted).

API::

    from backend.ai.sentiment import fetch_headlines

    headlines = await fetch_headlines("TCS", lookback_days=2)
    # [{"title": "...", "source": "Economic Times",
    #   "link": "https://...", "published": "2026-04-18T09:15:00"}]
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

_GNEWS_URL = (
    "https://news.google.com/rss/search?"
    "q={query}+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en"
)
_HTTP_TIMEOUT = 8.0
_MAX_ITEMS = 25


async def fetch_headlines(
    symbol: str,
    *,
    lookback_days: int = 2,
    max_items: int = _MAX_ITEMS,
) -> List[Dict]:
    """Fetch last-N-day Google News RSS headlines for a symbol."""
    query = quote_plus(f"{symbol} NSE stock")
    url = _GNEWS_URL.format(query=query, days=lookback_days)
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 QuantX/1.0"})
            resp.raise_for_status()
            xml = resp.text
    except Exception as e:
        logger.debug("fetch_headlines(%s): HTTP fail %s", symbol, e)
        return []

    try:
        root = ET.fromstring(xml)  # nosec B314 - RSS parsed from a fixed publisher allowlist
    except ET.ParseError as e:
        logger.debug("fetch_headlines(%s): XML parse fail %s", symbol, e)
        return []

    items: List[Dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src = ""
        src_el = item.find("source")
        if src_el is not None and src_el.text:
            src = src_el.text.strip()
        elif " - " in title:
            parts = title.rsplit(" - ", 1)
            if len(parts) == 2:
                title, src = parts[0].strip(), parts[1].strip()
        if not title:
            continue
        items.append({
            "title": _strip_html(title),
            "source": _strip_html(src) or "Google News",
            "link": link,
            "published": _parse_pubdate(pub),
        })
        if len(items) >= max_items:
            break
    return items


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _parse_pubdate(raw: str) -> Optional[str]:
    if not raw:
        return None
    try:
        dt = datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return raw


async def fetch_many(symbols: List[str], *, lookback_days: int = 2) -> Dict[str, List[Dict]]:
    """Fan out ``fetch_headlines`` across symbols. Bounded concurrency so
    we don't hammer Google News from a single IP."""
    sem = asyncio.Semaphore(5)

    async def _one(sym: str):
        async with sem:
            return sym, await fetch_headlines(sym, lookback_days=lookback_days)

    results = await asyncio.gather(*(_one(s) for s in symbols), return_exceptions=True)
    out: Dict[str, List[Dict]] = {}
    for item in results:
        if isinstance(item, Exception):
            continue
        sym, headlines = item
        out[sym] = headlines
    return out

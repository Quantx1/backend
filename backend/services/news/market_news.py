"""
Live market news — market-moving headlines for trading + finance, from FREE,
keyless RSS feeds. A curated mix of **Indian** market feeds (Economic Times
Markets, Livemint, Moneycontrol, Business Standard, BusinessLine) and **global**
finance feeds (CNBC Markets/Finance, Investing.com, MarketWatch).

No API key, no paid provider, no NSE licence needed — public news headlines
parsed with the Python stdlib. Fanned out in parallel, deduped, per-source
capped (so one feed can't dominate), and sorted newest-first. Dead/blocked feeds
just contribute nothing (honest-empty).
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 8.0
_PER_FEED = 8
_PER_SOURCE_CAP = 4  # no single source may flood the tape

# (url, region) — region ∈ {"India", "Global"}
_FEEDS: List[Tuple[str, str]] = [
    # ── Indian markets / finance ──
    ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "India"),
    ("https://www.livemint.com/rss/markets", "India"),
    ("https://www.moneycontrol.com/rss/business.xml", "India"),
    ("https://www.business-standard.com/rss/markets-106.rss", "India"),
    ("https://www.thehindubusinessline.com/markets/feeder/default.rss", "India"),
    # ── Global finance / markets ──
    ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "Global"),   # CNBC Markets
    ("https://www.cnbc.com/id/10000664/device/rss/rss.html", "Global"),    # CNBC Finance
    ("https://www.investing.com/rss/news_25.rss", "Global"),               # Investing breaking
    ("http://feeds.marketwatch.com/marketwatch/topstories/", "Global"),    # MarketWatch
]

_MEDIA_NS = "{http://search.yahoo.com/mrss/}"
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"
_IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"', re.I)


def _looks_image(url: str) -> bool:
    return bool(re.search(r"\.(jpg|jpeg|png|webp|gif)(\?|$)", url or "", re.I))


def _extract_image(item: ET.Element) -> Optional[str]:
    # media:content / media:thumbnail (incl. nested media:group)
    for tag in (f"{_MEDIA_NS}content", f"{_MEDIA_NS}thumbnail"):
        for el in item.iter(tag):
            url = el.get("url")
            if url:
                return url
    # RSS enclosure
    enc = item.find("enclosure")
    if enc is not None:
        url = enc.get("url") or ""
        if (enc.get("type", "").startswith("image")) or _looks_image(url):
            return url
    # first <img> inside description / content:encoded
    for tag in ("description", f"{_CONTENT_NS}encoded"):
        m = _IMG_RE.search(item.findtext(tag) or "")
        if m:
            return m.group(1)
    return None


_SOURCE_BY_HOST = {
    "economictimes": "Economic Times",
    "livemint": "Livemint",
    "moneycontrol": "Moneycontrol",
    "business-standard": "Business Standard",
    "thehindubusinessline": "BusinessLine",
    "cnbc": "CNBC",
    "investing": "Investing.com",
    "marketwatch": "MarketWatch",
    "reuters": "Reuters",
}


def _source_for(url: str) -> str:
    for host, name in _SOURCE_BY_HOST.items():
        if host in url:
            return name
    return "Markets"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _parse_pubdate(raw: str) -> Optional[str]:
    if not raw:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M %Z"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return None


async def _fetch_feed(client: httpx.AsyncClient, url: str, region: str) -> List[Dict[str, Any]]:
    try:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 QuantX/1.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)  # nosec B314 - fixed publisher allowlist
    except Exception as e:
        logger.debug("market_news: feed failed %s: %s", url, e)
        return []
    src = _source_for(url)
    out: List[Dict[str, Any]] = []
    for item in root.iter("item"):
        title = _strip_html(item.findtext("title") or "")
        if not title:
            continue
        desc = _strip_html(item.findtext("description") or "")
        if len(desc) > 180:
            desc = desc[:177].rsplit(" ", 1)[0] + "…"
        out.append({
            "title": title,
            "description": desc,
            "image": _extract_image(item),
            "source": src,
            "region": region,
            "link": (item.findtext("link") or "").strip(),
            "published": _parse_pubdate((item.findtext("pubDate") or "").strip()),
        })
        if len(out) >= _PER_FEED:
            break
    return out


async def fetch_market_news(limit: int = 15) -> List[Dict[str, Any]]:
    """Recent Indian + global market-moving headlines. Never raises — returns []
    on total failure (honest-empty)."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            results = await asyncio.gather(
                *[_fetch_feed(client, url, region) for url, region in _FEEDS],
                return_exceptions=True,
            )
    except Exception as e:
        logger.warning("market_news: gather failed: %s", e)
        return []

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    per_source: Dict[str, int] = {}
    for r in results:
        if not isinstance(r, list):
            continue
        for it in r:
            key = it["title"].lower()[:80]
            if key in seen:
                continue
            if per_source.get(it["source"], 0) >= _PER_SOURCE_CAP:
                continue
            seen.add(key)
            per_source[it["source"]] = per_source.get(it["source"], 0) + 1
            items.append(it)

    items.sort(key=lambda x: x.get("published") or "", reverse=True)

    # Interleave India + Global (each already newest-first) so BOTH regions lead
    # the tape — global feeds post more often and would otherwise crowd out the
    # Indian headlines a trader here needs most.
    india = [i for i in items if i["region"] == "India"]
    glob = [i for i in items if i["region"] == "Global"]
    merged: List[Dict[str, Any]] = []
    for a, b in zip_longest(india, glob):
        if a:
            merged.append(a)
        if b:
            merged.append(b)
    return merged[:limit]

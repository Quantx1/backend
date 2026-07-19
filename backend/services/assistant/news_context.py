"""
Financial news retrieval and ranking helpers.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx

from ...core.config import settings

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(element: Optional[ET.Element]) -> str:
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split()).strip()


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def _parse_published(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass(frozen=True)
class NewsArticle:
    title: str
    url: str
    source: str
    published_at: Optional[str]
    summary: str
    relevance_score: float

    def to_source(self) -> Dict[str, str]:
        payload: Dict[str, str] = {
            "title": self.title,
            "url": self.url,
            "source": self.source,
        }
        if self.published_at:
            payload["published_at"] = self.published_at
        return payload


class NewsContextService:
    """
    Pulls RSS/Atom feeds and returns query-relevant, deduplicated news items.
    """

    def __init__(self, feed_urls: Optional[List[str]] = None, timeout_seconds: Optional[float] = None):
        configured = [f.strip() for f in settings.ASSISTANT_NEWS_FEEDS.split(",") if f.strip()]
        self.feed_urls = feed_urls or configured
        self.timeout_seconds = timeout_seconds or settings.ASSISTANT_HTTP_TIMEOUT_SECONDS

    async def _fetch_feed(self, client: httpx.AsyncClient, url: str) -> List[NewsArticle]:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return self._parse_feed_xml(response.text, url)
        except Exception as exc:
            logger.warning("assistant_news_feed_fetch_failed feed=%s error=%s", url, type(exc).__name__)
            return []

    def _parse_feed_xml(self, xml_text: str, source_url: str) -> List[NewsArticle]:
        try:
            root = ET.fromstring(xml_text)  # nosec B314 - RSS parsed from a fixed publisher allowlist
        except ET.ParseError:
            return []

        articles: List[NewsArticle] = []

        # RSS format.
        for item in root.findall(".//item"):
            title = _safe_text(item.find("title"))
            link = _safe_text(item.find("link"))
            summary = _safe_text(item.find("description"))
            pub_raw = _safe_text(item.find("pubDate"))
            pub = _parse_published(pub_raw)
            source = _extract_domain(link or source_url)
            if title and link:
                articles.append(
                    NewsArticle(
                        title=title,
                        url=link,
                        source=source,
                        published_at=pub.isoformat() if pub else None,
                        summary=summary,
                        relevance_score=0.0,
                    )
                )

        # Atom format.
        atom_ns = "{http://www.w3.org/2005/Atom}"
        for entry in root.findall(f".//{atom_ns}entry"):
            title = _safe_text(entry.find(f"{atom_ns}title"))
            summary = _safe_text(entry.find(f"{atom_ns}summary")) or _safe_text(entry.find(f"{atom_ns}content"))
            updated_raw = _safe_text(entry.find(f"{atom_ns}updated"))
            link_el = entry.find(f"{atom_ns}link")
            link = ""
            if link_el is not None:
                link = link_el.attrib.get("href", "")
            pub = _parse_published(updated_raw)
            source = _extract_domain(link or source_url)
            if title and link:
                articles.append(
                    NewsArticle(
                        title=title,
                        url=link,
                        source=source,
                        published_at=pub.isoformat() if pub else None,
                        summary=summary,
                        relevance_score=0.0,
                    )
                )

        return articles

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
        tokens = [part for part in cleaned.split() if len(part) >= 3]
        return tokens

    @staticmethod
    def _hours_since(published_at: Optional[str]) -> float:
        if not published_at:
            return 72.0
        try:
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = _utc_now() - dt.astimezone(timezone.utc)
            return max(delta.total_seconds() / 3600, 0.0)
        except Exception:
            return 72.0

    def _score(self, article: NewsArticle, query_tokens: List[str]) -> float:
        corpus = f"{article.title} {article.summary}".lower()
        overlap = sum(1 for t in query_tokens if t in corpus)
        relevance = overlap / max(len(query_tokens), 1)
        recency_bonus = 1 / (1 + (self._hours_since(article.published_at) / 24))
        return (2.0 * relevance) + recency_bonus

    @staticmethod
    def _deduplicate(articles: Iterable[NewsArticle]) -> List[NewsArticle]:
        deduped: List[NewsArticle] = []
        seen: set[str] = set()
        for article in articles:
            key_raw = f"{article.url.strip().lower()}|{article.title.strip().lower()}"
            key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(article)
        return deduped

    def _rank(self, articles: List[NewsArticle], query: str) -> List[NewsArticle]:
        tokens = self._tokenize(query)
        scored = [
            NewsArticle(
                title=article.title,
                url=article.url,
                source=article.source,
                published_at=article.published_at,
                summary=article.summary,
                relevance_score=self._score(article, tokens),
            )
            for article in articles
        ]
        return sorted(scored, key=lambda item: item.relevance_score, reverse=True)

    async def get_relevant_news(self, query: str, limit: int = 5) -> List[NewsArticle]:
        if not self.feed_urls:
            return []

        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "QuantX-Assistant/1.0"}) as client:
            per_feed = await asyncio_gather([self._fetch_feed(client, url) for url in self.feed_urls])

        merged = [item for feed_items in per_feed for item in feed_items]
        deduped = self._deduplicate(merged)
        ranked = self._rank(deduped, query)
        return ranked[:limit]


async def asyncio_gather(tasks: List):
    import asyncio

    return await asyncio.gather(*tasks, return_exceptions=False)

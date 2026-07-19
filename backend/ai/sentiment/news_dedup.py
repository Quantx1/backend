"""
News dedup + signal-quality primitives — pure, deterministic, 0 LLM tokens.

The audit found the news pipeline treats the same wire story rewritten by 5
outlets as 5 independent headlines (inflating headline_count, skewing the
mean) and weights a pump blog the same as Reuters. These helpers fix both
without any new data source:

  * cluster_headlines  — near-duplicate clustering (token-set Jaccard) so a
    story is counted ONCE, with its outlet count as a corroboration signal.
  * source_weight      — outlet-credibility tiers (ET/Reuters > unknown blog).
  * recency_weight     — exponential freshness decay for materiality weighting.
  * urgency            — breaking / recent / today / older bucket from pubdate.

All pure functions — fully unit-testable, and they keep working when the LLM
is unavailable (the deterministic floor under the enriched layer).
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Tier-1 = established financial desks with editorial standards. Tier-2 =
# known-but-lesser outlets. Anything else falls to tier-3 (lowest weight).
# Matched case-insensitively as a substring of the Google-News source string.
_TIER1 = (
    "economic times", "et markets", "et now", "mint", "livemint", "business standard",
    "moneycontrol", "reuters", "bloomberg", "cnbc", "cnbctv18", "cnbc-tv18",
    "hindu businessline", "businessline", "financial express", "ndtv profit",
    "zee business", "bq prime", "bloomberg quint", "the hindu", "times of india",
    "hindustan times", "forbes", "outlook business",
)
_TIER2 = (
    "investing.com", "trade brains", "equitymaster", "5paisa", "angel one",
    "groww", "upstox", "icici direct", "motilal oswal", "goodreturns",
    "marketsmojo", "tickertape", "screener", "rediff", "financialexpress",
)

_SOURCE_WEIGHT = {1: 1.0, 2: 0.7, 3: 0.45}

# Overlap-coefficient above this (with a min shared-token guard) → same story.
# Overlap (|A∩B| / min(|A|,|B|)) beats raw Jaccard for short headlines because
# different outlets rewrite the SAME story at different lengths ("ACME wins
# large defence order worth 5000 cr" vs "ACME bags defence order 5000 crore").
DUP_THRESHOLD = 0.6
_MIN_SHARED_TOKENS = 3  # guard so 2-3 word headlines don't over-merge

_STOPWORDS = frozenset(
    "the a an of to in on for at by and or is are be as with from this that it "
    "its will has have was were after over amid into up down vs may can".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def source_tier(source: Optional[str]) -> int:
    s = (source or "").strip().lower()
    if not s:
        return 3
    if any(t in s for t in _TIER1):
        return 1
    if any(t in s for t in _TIER2):
        return 2
    return 3


def source_weight(source: Optional[str]) -> float:
    return _SOURCE_WEIGHT[source_tier(source)]


def _tokens(title: str) -> frozenset:
    return frozenset(
        w for w in _WORD_RE.findall((title or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _similar(a: frozenset, b: frozenset, *, threshold: float = DUP_THRESHOLD) -> bool:
    """Two headlines describe the same story?

    Overlap coefficient (|A∩B| / min size) with a minimum shared-token guard.
    For very short token sets (<MIN guard) fall back to strict Jaccard so tiny
    headlines that merely share a couple of common words don't over-merge.
    """
    if not a or not b:
        return False
    inter = len(a & b)
    if inter == 0:
        return False
    smaller = min(len(a), len(b))
    if smaller < _MIN_SHARED_TOKENS:
        return _jaccard(a, b) >= 0.7   # tiny headlines: require near-identical
    if inter < _MIN_SHARED_TOKENS:
        return False
    return (inter / smaller) >= threshold


def _parse_dt(published: Any) -> Optional[datetime]:
    if not published:
        return None
    if isinstance(published, datetime):
        return published if published.tzinfo else published.replace(tzinfo=timezone.utc)
    s = str(published).strip()
    # ISO first (the fetcher emits ISO when it can parse pubDate).
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            d = datetime.strptime(s, fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def hours_since(published: Any, *, now: Optional[datetime] = None) -> Optional[float]:
    d = _parse_dt(published)
    if d is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - d).total_seconds() / 3600.0)


def recency_weight(published: Any, *, now: Optional[datetime] = None, half_life_h: float = 24.0) -> float:
    """Exponential decay in [~0,1]; a story loses half its weight every
    ``half_life_h``. Undated stories get a neutral 0.5 (present but not fresh)."""
    h = hours_since(published, now=now)
    if h is None:
        return 0.5
    return math.exp(-math.log(2) * h / half_life_h)


def urgency(published: Any, *, now: Optional[datetime] = None) -> str:
    """breaking (<3h) / recent (<24h) / today (<48h) / older — deterministic."""
    h = hours_since(published, now=now)
    if h is None:
        return "older"
    if h < 3:
        return "breaking"
    if h < 24:
        return "recent"
    if h < 48:
        return "today"
    return "older"


def cluster_headlines(
    headlines: Sequence[Dict[str, Any]],
    *,
    threshold: float = DUP_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Greedily group near-duplicate stories by token-set Jaccard.

    Each headline dict carries {title, source, link, published}. Returns one
    cluster per unique story: the representative (most-credible, then newest)
    headline plus member_count, the unique sources, and the freshest pubdate —
    so a story counts ONCE and its outlet spread becomes a corroboration cue.
    """
    clusters: List[Dict[str, Any]] = []
    for h in headlines:
        title = str(h.get("title") or "").strip()
        if not title:
            continue
        toks = _tokens(title)
        placed = False
        for c in clusters:
            if _similar(toks, c["_tokens"], threshold=threshold):
                c["members"].append(h)
                c["sources"].add(str(h.get("source") or "").strip() or "unknown")
                placed = True
                break
        if not placed:
            clusters.append({
                "_tokens": toks,
                "members": [h],
                "sources": {str(h.get("source") or "").strip() or "unknown"},
            })

    out: List[Dict[str, Any]] = []
    for c in clusters:
        # Representative = highest source tier, then freshest (smallest age).
        def _rank(m):
            hs = hours_since(m.get("published"))
            return (source_tier(m.get("source")), hs if hs is not None else 1e9)
        rep = min(c["members"], key=_rank)
        out.append({
            "title": rep.get("title"),
            "source": rep.get("source"),
            "link": rep.get("link"),
            "published": rep.get("published"),
            "member_count": len(c["members"]),
            "sources": sorted(s for s in c["sources"] if s),
        })
    # Most-corroborated, then freshest, first.
    out.sort(key=lambda c: (-c["member_count"], hours_since(c.get("published")) if hours_since(c.get("published")) is not None else 1e9))
    return out


__all__ = [
    "DUP_THRESHOLD",
    "cluster_headlines",
    "source_tier",
    "source_weight",
    "recency_weight",
    "urgency",
    "hours_since",
]

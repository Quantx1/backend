"""Natural-language stock screener — hybrid AI agent.

Turns a trader's plain-English request ("oversold but in an uptrend with rising
volume", "stocks breaking out of a base with institutional accumulation") into
real scanner IDs, then the caller runs a confluence scan on them.

Cost-smart by design (founder rule): a FREE keyword fast-path handles the
obvious requests with 0 tokens; the LLM agent — the differentiator — only fires
for nuanced/multi-concept requests it can't cover, and its result is cached by
normalized query so repeats are free. The LLM routes through the free-first
OpenRouter chain.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Curated id -> name catalog (subset of the ~88 scanners that map to common
# trader vocabulary). Fed to the LLM and used for display labels.
CATALOG: Dict[int, str] = {
    1: "Breakout (Consolidation)", 2: "Top Gainers", 3: "Top Losers",
    4: "Volume Breakout", 5: "52 Week High", 6: "10 Day High", 7: "52 Week Low",
    8: "Volume Surge (>2.5x avg)", 9: "RSI Oversold (<30)", 10: "RSI Overbought (>70)",
    11: "MA Crossover (20 EMA)", 12: "Bullish Engulfing", 13: "Bearish Engulfing",
    14: "VCP (Volatility Contraction)", 15: "Bull Crossover (20/50 EMA)",
    16: "IPO Base Breakout", 17: "Bull Momentum", 19: "PSAR Reversal",
    21: "NR4 (Narrow Range)", 22: "NR7 (Narrow Range)", 26: "MACD Crossover",
    27: "MACD Bearish", 28: "Inside Bar", 29: "TTM Squeeze", 30: "Momentum Burst",
    31: "Trend Template (uptrend)", 32: "Super Trend", 33: "Pivot Breakout",
    34: "High Delivery % (institutional)", 35: "Bulk Deals", 36: "FII Net Buyers",
    37: "DII Net Buyers", 38: "FII+DII Buying", 40: "Long Buildup (price+ OI+)",
    41: "Short Buildup (price- OI+)", 42: "Short Covering (price+ OI-)",
    52: "Power Setup (high-conviction swing)", 53: "Squeeze Release",
    54: "MA Stack Bullish", 55: "Pre-Breakout Coil", 56: "Fresh Trend Start",
    57: "Oversold Bounce (uptrend)", 58: "Breakout w/ Volume (Stage-2)",
    59: "Pullback to EMA21", 60: "BB Squeeze Release", 61: "RS Leader (outperforming)",
    63: "MA Stack Bearish", 66: "Breakdown w/ Volume", 69: "RS Laggard", 70: "Bear Momentum",
}

# Free keyword fast-path: (regex, [scanner ids]). Multiple can fire.
_RULES = [
    (r"\boversold\b|rsi\s*(<|below|under)\s*[0-3]?\d\b", [9, 57]),
    (r"\boverbought\b|rsi\s*(>|above|over)\s*(7\d|8\d|9\d)\b", [10]),
    (r"rsi\s*(>|above|over)\s*[56]\d\b|strong\s+rsi", [17]),
    (r"volume\s*(surge|spike)|(high|rising|increas\w*|strong|heavy|above[- ]average)\s+volume|\d+\s*x\s*volume|volume\s*(>|above)", [8, 4]),
    (r"52[- ]?w(eek)?\s*high|yearly\s+high|all[- ]time\s+high", [5]),
    (r"10[- ]?day\s+high", [6]),
    (r"52[- ]?w(eek)?\s*low", [7]),
    (r"break(ing)?\s*out|breakout|breaking\s+(a\s+)?base", [1, 58]),
    (r"\bmomentum\b|momentum\s+leaders?", [17, 30]),
    (r"top\s+gainers?|rising\s+price|gaining", [2]),
    (r"top\s+losers?|falling|declin", [3]),
    (r"\bmacd\b", [26]),
    (r"golden\s+cross|ma\s+cross|moving\s+average\s+cross|above\s+\d+\s*(d|day|dma|ema|sma)|trend\s+template|uptrend|trending", [31, 54]),
    (r"\bvcp\b|volatility\s+contraction", [14]),
    (r"squeeze|ttm", [29, 53]),
    (r"engulfing", [12]),
    (r"reversal|reversing|bounce", [19, 57]),
    (r"inside\s+bar", [28]),
    (r"nr7|nr4|narrow\s+range|low\s+volatility|coil", [22, 55]),
    (r"delivery|institutional\s+(interest|accumulation|buying)", [34, 38]),
    (r"relative\s+strength|outperform|rs\s+leader|stronger\s+than\s+(nifty|market)", [61]),
    (r"\bfii\b", [36]), (r"\bdii\b", [37]),
    (r"long\s+build", [40]), (r"short\s+build", [41]), (r"short\s+cover", [42]),
    (r"pullback", [59]), (r"pivot", [33]),
]


def scanner_label(scanner_id: int) -> str:
    return CATALOG.get(scanner_id, f"Scanner {scanner_id}")


def parse_rules(text: str) -> List[int]:
    """Free keyword fast-path → scanner ids (order-preserving, de-duped)."""
    t = (text or "").lower()
    ids: List[int] = []
    for pattern, sids in _RULES:
        if re.search(pattern, t):
            for s in sids:
                if s not in ids:
                    ids.append(s)
    return ids


def _valid_ids() -> set:
    try:
        from ...data.screener.filters import SCANNER_FILTERS
        return set(SCANNER_FILTERS.keys())
    except Exception:
        return set(CATALOG.keys())


def llm_resolve(text: str, *, user_id: str = None) -> List[int]:
    """LLM agent: map a nuanced request to scanner ids. Free-first model,
    tight prompt, JSON-only output. Returns [] on any failure (caller falls
    back to the rule path)."""
    try:
        from ...ai.agents.llm import complete_sync, extract_json
        catalog = "\n".join(f"{i}: {n}" for i, n in CATALOG.items())
        system = (
            "You map a trader's plain-English stock-screen request to scanner IDs "
            "from the catalog. Reply with ONLY compact JSON {\"ids\": [int,...]} — the "
            "1-5 most relevant scanner IDs, no prose, no markdown."
        )
        prompt = f"Catalog (id: name):\n{catalog}\n\nRequest: {text!r}\nJSON:"
        reply = complete_sync(prompt, role="tool_planner", system=system,
                              temperature=0.0, feature="nl_screen", user_id=user_id)
        ids = (extract_json(reply) or {}).get("ids") or []
        valid = _valid_ids()
        out: List[int] = []
        for x in ids:
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if v in valid and v not in out:
                out.append(v)
        return out[:5]
    except Exception as e:
        logger.debug("nl_screen llm failed: %s", e)
        return []


_CACHE_TTL_S = 7 * 24 * 3600   # scanner-id resolution is stable


def _cache_key(norm: str) -> str:
    import hashlib
    return "nlscreen:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def resolve_screen_query(text: str, *, allow_llm: bool = True, user_id: str = None) -> Dict:
    """Resolve free text → scanner ids. Strong rule match (>=2 concepts) skips
    the LLM (free); otherwise the LLM agent resolves it and the result is
    cached persistently (L1 + Supabase llm_response_cache) by normalized query."""
    norm = " ".join((text or "").lower().split())
    rule_ids = parse_rules(norm)
    if len(rule_ids) >= 2:
        return {"scanner_ids": rule_ids[:6], "source": "rules", "query": text}
    from ...ai.agents.response_cache import cache_get, cache_set
    ck = _cache_key(norm)
    hit = cache_get(ck) or {}
    cached_ids = [int(x) for x in (hit.get("ids") or []) if str(x).lstrip("-").isdigit()]
    if cached_ids:
        return {"scanner_ids": cached_ids[:6], "source": "cache", "query": text}
    if allow_llm:
        llm_ids = llm_resolve(norm, user_id=user_id)
        if llm_ids:
            cache_set(ck, {"ids": llm_ids, "q": norm[:120]}, ttl_seconds=_CACHE_TTL_S,
                      surface="nl_screen", model="")
            return {"scanner_ids": llm_ids[:6], "source": "llm", "query": text}
    return {"scanner_ids": rule_ids[:6], "source": "rules", "query": text}

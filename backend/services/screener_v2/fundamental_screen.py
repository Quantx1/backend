"""
Fundamental screener engine (Phase 3, 2026-07-11).

The technical NL screener (nl_screen + confluence) runs scanner filters over the
LIVE indicator table (summary_df). Fundamentals live on a DIFFERENT data plane —
the nightly ``fundamentals_history`` snapshot (PE / ROE / ROCE / dividend yield /
sales & profit growth / promoter holding / market cap, sourced from screener.in
and cached). This module screens THAT plane: named presets ("low PE value",
"high ROCE quality", "quality compounders") plus a transparent 0-5 Quality Score.

HONESTY NOTE: a true Piotroski F-score (9 statement-level criteria) and Altman-Z
need income-statement / balance-sheet / cash-flow line items we do NOT ingest
(only headline ratios). We do NOT fabricate them. ``quality_score`` below is a
Piotroski-*spirit* composite computed transparently from the ratios we actually
have — named "Quality Score", never "Piotroski". ``debt_to_equity`` exists in the
schema but screener.in's headline ratios don't expose it, so it is currently
null for all rows; the "low debt" preset honest-empties until that column is fed
by a statements source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Columns we read from fundamentals_history (latest snapshot per symbol).
_FIELDS = (
    "pe", "roe", "roce", "market_cap_cr", "book_value", "dividend_yield",
    "current_price", "debt_to_equity", "sales_growth", "profit_growth",
    "promoter_pct",
)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def quality_score(row: Dict[str, Any]) -> int:
    """A transparent 0-5 quality composite (Piotroski-*spirit*, NOT the real
    9-point F-score — we lack statement data). +1 for each of: strong ROCE,
    strong ROE, growing profit, growing sales, majority promoter holding."""
    s = 0
    roce = _num(row.get("roce"))
    roe = _num(row.get("roe"))
    pg = _num(row.get("profit_growth"))
    sg = _num(row.get("sales_growth"))
    prom = _num(row.get("promoter_pct"))
    if roce is not None and roce >= 15:
        s += 1
    if roe is not None and roe >= 15:
        s += 1
    if pg is not None and pg > 0:
        s += 1
    if sg is not None and sg > 0:
        s += 1
    if prom is not None and prom >= 50:
        s += 1
    return s


@dataclass
class FundamentalPreset:
    key: str
    name: str
    blurb: str
    # Predicate over a fundamentals row → keep?  (all numeric-guarded)
    predicate: Callable[[Dict[str, Any]], bool]
    # Ranking key (row → sortable number); higher = better unless `asc`.
    rank: Callable[[Dict[str, Any]], float]
    asc: bool = False
    # Which fields this preset leans on (for honest "needs data" messaging).
    needs: List[str] = field(default_factory=list)


def _ge(row, key, thr) -> bool:
    v = _num(row.get(key))
    return v is not None and v >= thr


def _between(row, key, lo, hi) -> bool:
    v = _num(row.get(key))
    return v is not None and lo <= v <= hi


# ── The named presets — the fundamental screens Indian investors run most. ──
PRESETS: Dict[str, FundamentalPreset] = {
    p.key: p for p in [
        FundamentalPreset(
            "low-pe-value", "Low PE Value",
            "Profitable companies trading at a low price-to-earnings multiple.",
            lambda r: _between(r, "pe", 0.1, 15) and _ge(r, "roe", 8),
            lambda r: _num(r.get("pe")) or 1e9, asc=True, needs=["pe", "roe"],
        ),
        FundamentalPreset(
            "high-roce-quality", "High ROCE Quality",
            "Efficient capital allocators — return on capital employed above 20%.",
            lambda r: _ge(r, "roce", 20),
            lambda r: _num(r.get("roce")) or 0, needs=["roce"],
        ),
        FundamentalPreset(
            "quality-compounder", "Quality Compounder",
            "Durable franchises: high ROCE and ROE with growing profit.",
            lambda r: _ge(r, "roce", 15) and _ge(r, "roe", 15) and _ge(r, "profit_growth", 10),
            lambda r: quality_score(r) * 100 + (_num(r.get("roce")) or 0),
            needs=["roce", "roe", "profit_growth"],
        ),
        FundamentalPreset(
            "high-growth", "High Growth",
            "Fast-growing top and bottom line — sales and profit growth above 15%.",
            lambda r: _ge(r, "sales_growth", 15) and _ge(r, "profit_growth", 15),
            lambda r: (_num(r.get("profit_growth")) or 0) + (_num(r.get("sales_growth")) or 0),
            needs=["sales_growth", "profit_growth"],
        ),
        FundamentalPreset(
            "dividend-payer", "Dividend Payer",
            "Steady income — dividend yield above 2% with positive profit growth.",
            lambda r: _ge(r, "dividend_yield", 2) and (_num(r.get("profit_growth")) or 0) >= 0,
            lambda r: _num(r.get("dividend_yield")) or 0, needs=["dividend_yield"],
        ),
        FundamentalPreset(
            "promoter-backed", "Promoter-Backed",
            "High promoter conviction — promoter holding above 55%.",
            lambda r: _ge(r, "promoter_pct", 55),
            lambda r: _num(r.get("promoter_pct")) or 0, needs=["promoter_pct"],
        ),
        FundamentalPreset(
            "quality-score", "Top Quality Score",
            "Ranked by our 0-5 Quality Score (ROCE, ROE, growth, promoter holding).",
            lambda r: quality_score(r) >= 4,
            lambda r: quality_score(r) * 100 + (_num(r.get("roce")) or 0),
            needs=["roce", "roe", "profit_growth", "sales_growth", "promoter_pct"],
        ),
        FundamentalPreset(
            "low-debt", "Low Debt",
            "Conservative balance sheets — debt-to-equity below 0.3.",
            lambda r: _between(r, "debt_to_equity", 0.0, 0.3),
            lambda r: -(_num(r.get("debt_to_equity")) or 1e9), needs=["debt_to_equity"],
        ),
    ]
}


def preset_catalog() -> List[Dict[str, str]]:
    return [{"key": p.key, "name": p.name, "blurb": p.blurb} for p in PRESETS.values()]


def _shape(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {"symbol": row.get("symbol")}
    for k in _FIELDS:
        out[k] = _num(row.get(k))
    out["quality_score"] = quality_score(row)
    return out


def _latest_rows(sb, limit_symbols: int = 2000) -> List[Dict[str, Any]]:
    """Latest snapshot per symbol from fundamentals_history. The table is one
    row per (snapshot_date, symbol); we pull the most recent snapshots and keep
    the first (newest) seen per symbol."""
    rows = (
        sb.table("fundamentals_history")
        .select("symbol, snapshot_date, " + ", ".join(_FIELDS))
        .order("snapshot_date", desc=True)
        .limit(limit_symbols * 3)
        .execute()
        .data
    ) or []
    seen: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sym = r.get("symbol")
        if sym and sym not in seen:
            seen[sym] = r
    return list(seen.values())


def run_fundamental_screen(
    sb,
    *,
    preset: Optional[str] = None,
    criteria: Optional[Dict[str, Any]] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """Run a fundamental screen. Either a named ``preset`` OR a ``criteria`` dict
    of ``{field: {"min": x, "max": y}}`` (e.g. {"pe": {"max": 15}, "roce":
    {"min": 20}}). Returns ranked matches shaped for the UI + chat.

    Honest-empty (count=0) rather than fabricated when the underlying columns
    are unpopulated (e.g. the low-debt preset while debt_to_equity is null)."""
    rows = _latest_rows(sb)
    if preset:
        p = PRESETS.get(preset)
        if p is None:
            return {"error": f"unknown preset '{preset}'", "presets": [k for k in PRESETS]}
        matched = [r for r in rows if _safe(p.predicate, r)]
        matched.sort(key=lambda r: _safe_num(p.rank, r), reverse=not p.asc)
        results = [_shape(r) for r in matched[:limit]]
        # Honest signal: if the preset leans on a column that's null everywhere,
        # say so instead of silently returning nothing.
        note = None
        if not results and p.needs:
            populated = {f for f in p.needs if any(_num(r.get(f)) is not None for r in rows)}
            missing = [f for f in p.needs if f not in populated]
            if missing:
                note = f"No data yet for: {', '.join(missing)}"
        return {"preset": p.key, "name": p.name, "count": len(results),
                "results": results, "note": note}

    # Custom criteria path.
    crit = criteria or {}

    def _ok(r: Dict[str, Any]) -> bool:
        for fld, bound in crit.items():
            if fld not in _FIELDS or not isinstance(bound, dict):
                continue
            v = _num(r.get(fld))
            if v is None:
                return False
            if bound.get("min") is not None and v < float(bound["min"]):
                return False
            if bound.get("max") is not None and v > float(bound["max"]):
                return False
        return True

    matched = [r for r in rows if _ok(r)]
    matched.sort(key=lambda r: quality_score(r), reverse=True)
    return {"preset": None, "criteria": crit, "count": len(matched[:limit]),
            "results": [_shape(r) for r in matched[:limit]]}


def _safe(fn, r) -> bool:
    try:
        return bool(fn(r))
    except Exception:
        return False


def _safe_num(fn, r) -> float:
    try:
        n = fn(r)
        return float(n) if n is not None and n == n else -1e9
    except Exception:
        return -1e9

"""Alpha Factory — one unified per-stock Scores block for the dossier.

Aggregates ONLY real, already-computed sources into a single compact shape:

    alpha           — cross-sectional rank from the nightly ``alpha_scores``
                      table, converted to a percentile vs that day's universe
    momentum/trend/
    low_volatility  — cross-sectional factor percentiles from the AI Factor
                      Screener (``factor_rank``, 30m-cached full-universe run;
                      this symbol is looked up in the cached results)
    mood            — latest ``news_sentiment`` row (score in [-1, 1])
    iv_rank         — IV Rank from ``iv_history`` for F&O names only
                      (honest-None otherwise; needs 20+ days of history)

Shape: ``scores(symbol) -> {symbol, scores: [{key, label, value, pct, note}],
composite}``. ``pct`` is a 0-100 percentile ONLY when the source is genuinely
cross-sectional (alpha + the three factors); mood and iv_rank carry ``pct=None``
because they are per-symbol scales, not peer ranks. ``composite`` is the plain
mean of the available cross-sectional pcts and requires >=2 of them — else None.

Deliberately OMITTED: Quality and Smart-Money scores. We have no fundamentals
or institutional-flow data wired per-symbol, so fabricating those numbers would
violate the honest-empty contract. Add them only when a real source lands.

Deterministic, 0 tokens. ``build_scores`` is pure (tested); the ``_read_*``
readers are thin, best-effort, and monkeypatchable in tests.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The three factor-screener keys we surface, in display order.
_FACTOR_KEYS = ("momentum", "trend", "low_volatility")
_FACTOR_LABELS = {
    "momentum": "Momentum",
    "trend": "Trend",
    "low_volatility": "Low Volatility",
}
# factor_rank truncates to `top`; ask for more than the full NSE main board
# (~2,385 EQ) so any symbol with enough bars is present in the cached results.
_FACTOR_TOP = 5000


# ── pure shaping ─────────────────────────────────────────────────────────


def _rank_to_pct(rank: Optional[int], universe: Optional[int]) -> Optional[float]:
    """Rank 1..n -> percentile [0..100] where rank #1 maps to 100.

    None when the rank/universe is missing or the universe is too small to
    rank against (n < 2). Pure."""
    if rank is None or universe is None:
        return None
    if universe < 2 or rank < 1 or rank > universe:
        return None
    return round((1 - (rank - 1) / (universe - 1)) * 100, 2)


def build_scores(
    alpha: Optional[Dict[str, Any]],
    factor_pcts: Dict[str, Optional[float]],
    mood: Optional[Dict[str, Any]],
    iv: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pure assembly of the unified Scores block from pre-read inputs.

    Missing sources are honestly omitted (no placeholder rows). ``composite``
    is the mean of entries that carry a cross-sectional ``pct``; it needs at
    least 2 of them, else None."""
    entries: List[Dict[str, Any]] = []

    if alpha and alpha.get("rank") is not None:
        pct = _rank_to_pct(alpha.get("rank"), alpha.get("universe"))
        note = f"#{alpha['rank']} of {alpha.get('universe')}" if alpha.get("universe") else f"#{alpha['rank']}"
        if alpha.get("trade_date"):
            note += f" · {alpha['trade_date']}"
        entries.append({
            "key": "alpha", "label": "Alpha Rank",
            "value": int(alpha["rank"]), "pct": pct, "note": note,
        })

    for k in _FACTOR_KEYS:
        p = factor_pcts.get(k) if factor_pcts else None
        if p is None:
            continue
        entries.append({
            "key": k, "label": _FACTOR_LABELS[k],
            "value": None, "pct": round(float(p), 2),
            "note": "percentile vs universe",
        })

    if mood and mood.get("score") is not None:
        hc = mood.get("headlines")
        note = f"{hc} headlines" if hc else "news score"
        if mood.get("trade_date"):
            note += f" · {mood['trade_date']}"
        entries.append({
            "key": "mood", "label": "News Mood",
            "value": round(float(mood["score"]), 3), "pct": None, "note": note,
        })

    if iv and iv.get("iv_rank") is not None:
        days = iv.get("days")
        entries.append({
            "key": "iv_rank", "label": "IV Rank",
            "value": round(float(iv["iv_rank"]), 1), "pct": None,
            "note": f"{days}d IV history" if days else "IV history",
        })

    pcts = [e["pct"] for e in entries if e.get("pct") is not None]
    composite = round(sum(pcts) / len(pcts), 2) if len(pcts) >= 2 else None
    return {"scores": entries, "composite": composite}


# ── thin best-effort readers (monkeypatched in tests) ────────────────────


def _read_alpha(symbol: str) -> Optional[Dict[str, Any]]:
    """Latest alpha_scores rank for the symbol + that day's universe size."""
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        rows = (sb.table("alpha_scores").select("qlib_rank,trade_date")
                .eq("symbol", symbol).order("trade_date", desc=True)
                .limit(1).execute().data or [])
        if not rows or rows[0].get("qlib_rank") is None:
            return None
        td = rows[0].get("trade_date")
        cnt = (sb.table("alpha_scores").select("symbol", count="exact")
               .eq("trade_date", td).execute())
        universe = int(cnt.count) if getattr(cnt, "count", None) else None
        return {"rank": int(rows[0]["qlib_rank"]), "universe": universe,
                "trade_date": str(td) if td else None}
    except Exception as e:
        logger.debug("stock_scores alpha read failed for %s: %s", symbol, e)
        return None


def _read_factor_pcts(symbol: str) -> Dict[str, Optional[float]]:
    """This symbol's factor percentiles out of the cached full-universe
    factor_rank run (30m TTL inside factor_screener). Honest-empty when the
    symbol isn't in the results (too few bars / thin universe)."""
    try:
        from .factor_screener import factor_rank
        out = factor_rank(list(_FACTOR_KEYS), universe=None, top=_FACTOR_TOP)
        for r in out.get("results") or []:
            if r.get("symbol") == symbol:
                return dict(r.get("factor_scores") or {})
        return {}
    except Exception as e:
        logger.debug("stock_scores factor read failed for %s: %s", symbol, e)
        return {}


def _read_mood(symbol: str) -> Optional[Dict[str, Any]]:
    """Latest news_sentiment row for the symbol."""
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        rows = (sb.table("news_sentiment")
                .select("mean_score,headline_count,trade_date")
                .eq("symbol", symbol).order("trade_date", desc=True)
                .limit(1).execute().data or [])
        if not rows or rows[0].get("mean_score") is None:
            return None
        return {"score": float(rows[0]["mean_score"]),
                "headlines": int(rows[0].get("headline_count") or 0),
                "trade_date": str(rows[0].get("trade_date") or "") or None}
    except Exception as e:
        logger.debug("stock_scores mood read failed for %s: %s", symbol, e)
        return None


def _read_iv(symbol: str) -> Optional[Dict[str, Any]]:
    """IV Rank for F&O names only — None for non-F&O or thin IV history.

    Read-only: uses the stored series + its latest point; never records a new
    IV row (the option-chain snapshot job owns the write side)."""
    try:
        from ...core.database import get_supabase_admin
        from ...data.reference.nse_reference import FNO_INDEX_NAME
        sb = get_supabase_admin()
        member = (sb.table("index_constituents").select("symbol")
                  .eq("index_name", FNO_INDEX_NAME).eq("symbol", symbol)
                  .limit(1).execute().data or [])
        if not member:
            return None
        from ..fno_scanner.iv_store import _read_series, compute_iv_rank_percentile
        series = _read_series(symbol)
        if not series:
            return None
        res = compute_iv_rank_percentile(series, series[-1])
        if res.get("iv_rank") is None:
            return None
        return res
    except Exception as e:
        logger.debug("stock_scores iv read failed for %s: %s", symbol, e)
        return None


# ── public entrypoint ────────────────────────────────────────────────────


def scores(symbol: str) -> Dict[str, Any]:
    """Unified per-stock Scores block. {symbol, scores: [...], composite}.

    Every source is best-effort; what's missing is omitted, never invented."""
    sym = (symbol or "").strip().upper()
    if sym.endswith(".NS"):
        sym = sym[:-3]
    if not sym:
        return {"symbol": sym, "scores": [], "composite": None}
    out = build_scores(
        _read_alpha(sym),
        _read_factor_pcts(sym),
        _read_mood(sym),
        _read_iv(sym),
    )
    return {"symbol": sym, **out}

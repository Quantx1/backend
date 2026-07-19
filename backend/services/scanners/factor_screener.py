"""AI Factor Screener — compose continuous factors instead of preset scanners.

Lets a user build a screen like "low-volatility momentum" by selecting one or
more CONTINUOUS factors we can honestly compute from daily candles. Each factor
is computed per symbol, converted to a cross-sectional PERCENTILE [0..100], and
the composite is the mean of the selected-factor percentiles. Higher composite =
ranks better on the combination of chosen factors.

Deterministic, 0 tokens — pure cross-sectional math over real OHLC. Honest-empty
when the universe data is thin. `_percentile_ranks` + `compute_factor_ranking`
are pure (tested); the candle read is one window query via direct Postgres,
mirroring `sector_rotation.py` / `breadth.py`.

Factor sign convention: every factor is oriented so that a HIGHER raw value is
the desirable end (e.g. low_volatility = 1/vol, so calmer names score higher).
We only declare factors we can actually derive from candles — no value/quality,
because fundamentals aren't in the candle store.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Each factor: label (UI), description, and how many trailing daily bars it needs.
# `direction` is informational — raw values are ALREADY oriented so higher=better.
AVAILABLE_FACTORS: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "label": "Momentum",
        "description": "50-day total return (higher = stronger trend).",
        "min_bars": 51,
    },
    "low_volatility": {
        "label": "Low Volatility",
        "description": "Inverse of 20-day realized volatility (calmer names score higher).",
        "min_bars": 21,
    },
    "trend": {
        "label": "Trend",
        "description": "Distance above a RISING 50-day moving average (0 if 50DMA isn't rising).",
        "min_bars": 60,
    },
}

# Longest lookback any factor needs, plus headroom — bounds the per-symbol read.
_MAX_LOOKBACK = 60
_CACHE: Dict[str, tuple] = {}
_TTL_S = 1800


# ── pure factor math ────────────────────────────────────────────────────


def momentum_50d(closes: Sequence[float]) -> Optional[float]:
    """50-bar total return in %. None if < 51 bars or base price is zero."""
    if not closes or len(closes) < 51:
        return None
    c0, c1 = closes[-51], closes[-1]
    if not c0:
        return None
    return round((c1 / c0 - 1) * 100, 4)


def low_volatility_20d(closes: Sequence[float]) -> Optional[float]:
    """Inverse of 20-day realized vol (std of daily returns), oriented so a
    CALMER series scores HIGHER. Returns 1 / (vol + epsilon). None if < 21
    bars or the series is flat (no dispersion to rank on)."""
    if not closes or len(closes) < 21:
        return None
    window = closes[-21:]
    rets: List[float] = []
    for a, b in zip(window[:-1], window[1:]):
        if not a:
            return None
        rets.append(b / a - 1.0)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    vol = math.sqrt(var)
    if vol <= 0:
        return None
    return round(1.0 / vol, 6)


def trend_above_50dma(closes: Sequence[float]) -> Optional[float]:
    """% above the 50-day MA, but ONLY when the 50DMA itself is rising
    (today's 50DMA > the 50DMA five bars ago). If the MA is flat/falling the
    factor is 0 — a name above a falling average isn't in an uptrend. None if
    < 60 bars."""
    if not closes or len(closes) < 60:
        return None
    sma_now = sum(closes[-50:]) / 50.0
    sma_prev = sum(closes[-55:-5]) / 50.0
    if not sma_now:
        return None
    if sma_now <= sma_prev:
        return 0.0
    return round((closes[-1] / sma_now - 1) * 100, 4)


_FACTOR_FUNCS = {
    "momentum": momentum_50d,
    "low_volatility": low_volatility_20d,
    "trend": trend_above_50dma,
}


def _percentile_ranks(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Cross-sectional percentile [0..100] for each value vs its peers.

    Rank = (count of strictly-smaller values) / (n - 1) * 100, so the best
    value maps to 100 and the worst to 0. None inputs stay None (the symbol
    is simply missing that factor). Ties share their lower-bound percentile.
    """
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    n = len(present)
    out: List[Optional[float]] = [None] * len(values)
    if n == 0:
        return out
    if n == 1:
        out[present[0][0]] = 100.0
        return out
    sorted_vals = sorted(v for _, v in present)
    for i, v in present:
        # number of peers strictly less than v
        lo = 0
        for sv in sorted_vals:
            if sv < v:
                lo += 1
            else:
                break
        out[i] = round(lo / (n - 1) * 100, 2)
    return out


def compute_factor_ranking(
    factor_matrix: Dict[str, Dict[str, Optional[float]]],
    factors: Sequence[str],
    *,
    top: int = 25,
) -> List[Dict[str, Any]]:
    """Rank symbols by the mean of their selected-factor percentiles.

    `factor_matrix`: {symbol: {factor_name: raw_value_or_None}}.
    Returns [{symbol, composite, factor_scores}] sorted by composite desc,
    truncated to `top`. `factor_scores` holds each requested factor's
    percentile (0..100), not the raw value. A symbol is included only if it
    has at least one requested-factor percentile (honest-empty otherwise).
    """
    sel = [f for f in factors if f in _FACTOR_FUNCS]
    if not sel or not factor_matrix:
        return []
    symbols = list(factor_matrix.keys())
    # percentile each requested factor across the cross-section
    pct_by_factor: Dict[str, List[Optional[float]]] = {}
    for f in sel:
        raw = [factor_matrix[s].get(f) for s in symbols]
        pct_by_factor[f] = _percentile_ranks(raw)

    results: List[Dict[str, Any]] = []
    for idx, sym in enumerate(symbols):
        scores: Dict[str, float] = {}
        for f in sel:
            p = pct_by_factor[f][idx]
            if p is not None:
                scores[f] = p
        if not scores:
            continue
        composite = round(sum(scores.values()) / len(scores), 2)
        results.append({"symbol": sym, "composite": composite, "factor_scores": scores})

    results.sort(key=lambda r: r["composite"], reverse=True)
    return results[: max(1, top)]


# ── candle access (one window query, direct Postgres) ───────────────────


def _read_closes(universe: Optional[str]) -> Dict[str, List[float]]:
    """Per-symbol trailing closes (oldest->newest) from `candles`.

    One window query keeps the last `_MAX_LOOKBACK` daily bars per symbol.
    Optional `universe` narrows to an index's constituents (via
    `index_constituents`); unknown/empty universe = the full candle store.
    """
    from ...data.ohlc_store import pg_connect
    members: Optional[List[str]] = None
    if universe:
        members = _universe_symbols(universe)
        if not members:
            return {}
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            params: List[Any] = [_MAX_LOOKBACK]
            sym_filter = ""
            if members:
                sym_filter = "AND stock_symbol = ANY(%s)"
                params.append(members)
            # The only interpolated fragment is sym_filter, a fixed internal
            # constant ("" or "AND stock_symbol = ANY(%s)"); every value is bound
            # via %s. Single-lined so the justified nosec lands on the flagged node.
            cur.execute(f"WITH ranked AS (SELECT stock_symbol, close, timestamp, row_number() OVER (PARTITION BY stock_symbol ORDER BY timestamp DESC) AS rn FROM candles WHERE interval='1d' {sym_filter}) SELECT stock_symbol, close FROM ranked WHERE rn <= %s ORDER BY stock_symbol, timestamp ASC", params)  # nosec B608
            out: Dict[str, List[float]] = {}
            for sym, close in cur.fetchall():
                if close is None:
                    continue
                out.setdefault(sym, []).append(float(close))
            return out
    finally:
        conn.close()


def _universe_symbols(universe: str) -> List[str]:
    """Resolve an index name -> its constituent symbols. Honest-empty on miss."""
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        name = universe.strip().upper()
        rows = (sb.table("index_constituents").select("symbol")
                .eq("index_name", name).limit(1000).execute().data or [])
        return [r["symbol"] for r in rows if r.get("symbol")]
    except Exception as e:
        logger.debug("factor universe resolve failed for %s: %s", universe, e)
        return []


def _build_matrix(
    closes_by_symbol: Dict[str, List[float]], factors: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute each requested factor's raw value per symbol from its closes."""
    sel = [f for f in factors if f in _FACTOR_FUNCS]
    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    for sym, closes in closes_by_symbol.items():
        row: Dict[str, Optional[float]] = {}
        for f in sel:
            row[f] = _FACTOR_FUNCS[f](closes)
        matrix[sym] = row
    return matrix


def factor_rank(
    factors: List[str], universe: Optional[str] = None, top: int = 25,
) -> Dict[str, Any]:
    """Compose continuous factors into one cross-sectional ranking.

    {factors, available_factors, results: [{symbol, composite, factor_scores}],
    universe_size}. `results` sorted by composite desc, top N. Only honestly
    computable factors are accepted; unknown ones are dropped. Honest-empty
    (results=[]) when the universe data is thin. Cached 30m per (factors,
    universe) key — deterministic, so a re-run within the window is free."""
    sel = [f for f in factors if f in AVAILABLE_FACTORS]
    available = [
        {"key": k, "label": v["label"], "description": v["description"]}
        for k, v in AVAILABLE_FACTORS.items()
    ]
    base = {
        "factors": sel,
        "available_factors": available,
        "results": [],
        "universe_size": 0,
    }
    if not sel:
        return base

    cache_key = f"{','.join(sorted(sel))}|{(universe or '').upper()}|{top}"
    hit = _CACHE.get(cache_key)
    if hit and (time.monotonic() - hit[0]) < _TTL_S:
        return hit[1]

    try:
        closes_by_symbol = _read_closes(universe)
    except Exception as e:
        logger.debug("factor_rank candle read failed: %s", e)
        return base
    if not closes_by_symbol:
        return base

    matrix = _build_matrix(closes_by_symbol, sel)
    results = compute_factor_ranking(matrix, sel, top=top)
    out = {
        "factors": sel,
        "available_factors": available,
        "results": results,
        "universe_size": len(closes_by_symbol),
    }
    if results:
        _CACHE[cache_key] = (time.monotonic(), out)
    return out

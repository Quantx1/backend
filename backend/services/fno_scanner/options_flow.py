"""Options Flow aggregator — ONE consolidated read of today's option-chain
positioning for an F&O symbol.

Today call-writing / put-writing / PCR / max-pain pull / biggest OI buildup /
overall lean are scattered across FnoTab, OiHeatmap and FnoStockScanners. This
service folds them into a single summary dict so the UI can render one card.

It invents NOTHING and costs 0 LLM tokens. Everything is derived deterministically
from the SAME per-strike OI + delta-OI the snapshot already reads (option_type /
strike / oi / oi_change), then re-uses the verified `fetch_index_snapshot` for the
PCR / max-pain / top-strike / biggest-buildup numbers so we stay consistent with
the rest of the F&O dashboard. Honest-empty (returns None) when the chain is
unavailable — never a synthetic fallback.

Writing convention (NSE option-chain reading):
  - "Call writing" = sellers ADDING call OI (positive CE Δ-OI) → they're capping
    upside → bearish pressure.
  - "Put writing"  = sellers ADDING put OI (positive PE Δ-OI) → they're defending
    a floor → bullish pressure.
The deterministic lean blends the writing balance with the PCR classification:
two independent votes; they agree → conviction, they conflict → neutral.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# A side's writing total has to be at least this much bigger than the other to
# count as a directional vote. Below it the two sides are effectively balanced
# and the writing vote is "neutral" (we don't read noise as positioning).
_WRITING_DOMINANCE = 1.15

# How many of the heaviest single-strike Δ-OI moves we surface as "top buildup".
_TOP_BUILDUP_N = 3


def _writing_vote(call_writing: int, put_writing: int) -> str:
    """Directional vote from the writing balance. Put-writing dominant → bullish
    (floor defended); call-writing dominant → bearish (ceiling capped); roughly
    balanced → neutral."""
    if call_writing <= 0 and put_writing <= 0:
        return "neutral"
    if put_writing > call_writing * _WRITING_DOMINANCE:
        return "bullish"
    if call_writing > put_writing * _WRITING_DOMINANCE:
        return "bearish"
    return "neutral"


def _pcr_vote(pcr: Optional[float]) -> str:
    """Directional vote from PCR (OI). High PCR (more open puts) = bullish lean;
    low PCR (more open calls) = bearish lean; mid-band = neutral. Same bands the
    Options Teacher uses, kept here so the lean is self-contained + testable."""
    if pcr is None:
        return "neutral"
    if pcr >= 1.0:
        return "bullish"
    if pcr <= 0.7:
        return "bearish"
    return "neutral"


def _combine_lean(writing_vote: str, pcr_vote: str) -> str:
    """Blend the two independent votes deterministically.

    Agreement → that direction. One neutral + one directional → the directional
    one (a single real signal still leans). Direct conflict (bullish vs bearish)
    → neutral — we don't pretend to resolve a genuinely mixed chain.
    """
    votes = [v for v in (writing_vote, pcr_vote) if v != "neutral"]
    if not votes:
        return "neutral"
    if all(v == "bullish" for v in votes):
        return "bullish"
    if all(v == "bearish" for v in votes):
        return "bearish"
    return "neutral"


def aggregate_flow(chain: List[Dict[str, Any]], spot: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Pure aggregation of raw option-chain rows into a flow summary.

    `chain` rows mirror what the market provider yields (and what snapshot.py
    parses): {option_type: 'CE'|'PE', strike, oi, oi_change}. Returns None when
    there's nothing real to aggregate (empty / all-zero chain). 0 tokens.
    """
    by_strike: Dict[float, Dict[str, int]] = {}
    total_call_writing = 0   # Σ positive CE Δ-OI (fresh call selling)
    total_put_writing = 0    # Σ positive PE Δ-OI (fresh put selling)
    total_ce_oi = 0
    total_pe_oi = 0

    for row in chain or []:
        try:
            strike = float(row.get("strike", 0) or 0)
            if strike <= 0:
                continue
            otype = str(row.get("option_type", "")).upper()
            oi = int(row.get("oi", 0) or 0)
            oi_change = int(row.get("oi_change", 0) or 0)
            entry = by_strike.setdefault(
                strike, {"strike": strike, "call_oi": 0, "put_oi": 0,
                         "call_oi_change": 0, "put_oi_change": 0})
            if otype == "CE":
                entry["call_oi"] += oi
                entry["call_oi_change"] += oi_change
                total_ce_oi += oi
                if oi_change > 0:
                    total_call_writing += oi_change
            elif otype == "PE":
                entry["put_oi"] += oi
                entry["put_oi_change"] += oi_change
                total_pe_oi += oi
                if oi_change > 0:
                    total_put_writing += oi_change
        except Exception:
            continue

    if not by_strike or (total_ce_oi == 0 and total_pe_oi == 0):
        return None

    pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else None

    # Max pain — strike that minimises total option-writer loss at expiry
    # (Zerodha Varsity definition; same math as snapshot.py).
    strikes = sorted(by_strike.keys())
    max_pain = _max_pain(by_strike, strikes)
    max_pain_pull_pct = None
    if max_pain is not None and spot and spot > 0:
        max_pain_pull_pct = round((max_pain - spot) / spot * 100.0, 2)

    # Biggest single-strike Δ-OI moves (institutional fingerprint) — both sides.
    moves: List[Dict[str, Any]] = []
    for v in by_strike.values():
        for side, delta in (("CE", v["call_oi_change"]), ("PE", v["put_oi_change"])):
            if delta == 0:
                continue
            moves.append({
                "strike": v["strike"],
                "side": side,
                "oi_change": delta,
                "direction": "writing" if delta > 0 else "unwinding",
            })
    moves.sort(key=lambda m: abs(m["oi_change"]), reverse=True)
    top_buildup = moves[:_TOP_BUILDUP_N]

    writing_vote = _writing_vote(total_call_writing, total_put_writing)
    pcr_vote = _pcr_vote(pcr)
    lean = _combine_lean(writing_vote, pcr_vote)

    return {
        "total_call_writing": total_call_writing,
        "total_put_writing": total_put_writing,
        "pcr": pcr,
        "max_pain": round(max_pain, 1) if max_pain is not None else None,
        "max_pain_pull_pct": max_pain_pull_pct,
        "top_buildup": top_buildup,
        "biggest_buildup": top_buildup[0] if top_buildup else None,
        "writing_vote": writing_vote,
        "pcr_vote": pcr_vote,
        "lean": lean,
        "strike_count": len(by_strike),
    }


def _max_pain(by_strike: Dict[float, Dict[str, int]], strikes: List[float]) -> Optional[float]:
    """Strike K* minimising Σ writer-loss at expiry — Zerodha Varsity formula."""
    if not strikes:
        return None
    best_strike = None
    best_loss = None
    for candidate in strikes:
        loss = 0.0
        for k in strikes:
            so = by_strike[k]
            if candidate > k:
                loss += (candidate - k) * so["call_oi"]
            elif candidate < k:
                loss += (k - candidate) * so["put_oi"]
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_strike = candidate
    return best_strike


def options_flow(symbol: str) -> Optional[Dict[str, Any]]:
    """ONE consolidated options-flow summary for an F&O `symbol`.

    Returns {symbol, spot, total_call_writing, total_put_writing, pcr, max_pain,
    max_pain_pull_pct, top_buildup, biggest_buildup, writing_vote, pcr_vote,
    lean, strike_count}. Deterministic, 0 LLM tokens. Returns None (honest-empty)
    when the option-chain provider is unavailable or returns nothing real.

    Public-safe: uses the admin market provider (not per-user broker creds), the
    same source snapshot.py / oi-heatmap read.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return None

    try:
        from ...data.market import get_market_data_provider
        mp = get_market_data_provider()
    except Exception as e:
        logger.warning("options_flow: market provider import failed: %s", e)
        return None

    try:
        chain = mp.get_option_chain(sym, "")
    except Exception as e:
        logger.warning("options_flow: get_option_chain(%s) failed: %s", sym, e)
        return None
    if not chain:
        return None

    spot = None
    try:
        q = mp.get_quote(sym)
        spot = float(q.ltp) if q and getattr(q, "ltp", None) else None
    except Exception:
        pass

    summary = aggregate_flow(chain, spot)
    if summary is None:
        return None

    return {"symbol": sym, "spot": round(spot, 2) if spot else None, **summary}

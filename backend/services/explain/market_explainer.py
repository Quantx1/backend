"""AI Market Explainer — index-level plain-English market summary.

The whole-market analogue of `why_moving`: assemble REAL facts deterministically
by REUSING existing services (NIFTY % change from the market provider, true
advance/decline breadth, sector rotation leaders/laggards, current regime), build
a `drivers` list that is ALWAYS returned (0 tokens), then OPTIONALLY narrate over
the facts with the free grounded reasoner (cached per day) only when `use_llm`.
Honest-empty when no real facts can be assembled.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SEBI-safe desk voice for the public market-explainer narrative. The generic
# grounded reasoner prompt guards against hallucinated numbers and emoji but
# does NOT forbid promised returns, buy/sell directives, or real model names —
# so this public, everyone-can-see summary passes its own guarded system prompt.
# No assured returns, no personalised buy/sell advice, no emoji, engine public
# names only (Alpha / Mood / Regime), never real model names.
_EXPLAIN_SYSTEM = (
    "You are an Indian-equities desk analyst writing 'What's happening in the "
    "market' for a retail trader. Reason ONLY over the REAL facts provided as "
    "JSON and NEVER invent prices, levels, or numbers; when you cite a figure "
    "use the EXACT value from the facts, and skip any theme the facts don't "
    "cover. Structure the read as flowing prose (no markdown headers, no "
    "lists): first the tape and market internals (breadth, highs/lows), then "
    "institutional flows and FII index-futures positioning and what that "
    "stance implies, then volatility (VIX vs realized), then what CHANGED "
    "versus yesterday and what is worth watching next session. Describe the "
    "directional read in plain words (constructive / cautious / mixed). Do "
    "NOT promise or imply any return, profit, or assured outcome, and do NOT "
    "tell the reader to buy or sell any specific security — describe the "
    "picture and the risks only. NO emoji or decorative symbols. Refer to "
    "internal models only as Alpha, Mood, or Regime; never by any other name. "
    "6-8 sentences of plain prose."
)


def _assemble_facts() -> Dict[str, Any]:
    """Gather the real, current market-wide facts. Best-effort per factor."""
    facts: Dict[str, Any] = {}

    # NIFTY % change — same market-provider access why_moving uses.
    try:
        from ...data.market import get_market_data_provider
        mp = get_market_data_provider()
        q = mp.get_quote("NIFTY") or {}
        chg = getattr(q, "change_percent", None)
        if chg is not None:
            facts["nifty"] = {"ltp": getattr(q, "ltp", None), "change_pct": round(chg, 2)}
    except Exception as e:
        logger.debug("market_explainer nifty facts failed: %s", e)

    # True advance/decline breadth today (services/breadth.py).
    try:
        from ..scanners.breadth import breadth
        b = breadth()
        today = (b or {}).get("today")
        if today and (today.get("adv") is not None or today.get("dec") is not None):
            adv, dec = int(today.get("adv") or 0), int(today.get("dec") or 0)
            tot = adv + dec
            facts["breadth"] = {
                "adv": adv, "dec": dec, "ratio": (b or {}).get("ratio"),
                "adv_pct": round(adv / tot * 100) if tot else None,
            }
    except Exception as e:
        logger.debug("market_explainer breadth facts failed: %s", e)

    # Sector leaders / laggards (services/sector_rotation.py RRG quadrants).
    try:
        from ..scanners.sector_rotation import sector_rotation
        rows = sector_rotation() or []
        if rows:
            leading = [r["sector"] for r in rows if r.get("quadrant") == "leading"][:3]
            lagging = [r["sector"] for r in rows if r.get("quadrant") == "lagging"][:3]
            # Fall back to RS-long ordering if no clean quadrant labels.
            if not leading and not lagging:
                leading = [r["sector"] for r in rows[:2]]
                lagging = [r["sector"] for r in rows[-2:]]
            facts["sectors"] = {"leading": leading, "lagging": lagging,
                                "top": rows[:6]}
    except Exception as e:
        logger.debug("market_explainer sector facts failed: %s", e)

    # Current market regime (same regime_history read why_moving uses).
    try:
        from ...core.database import get_supabase_admin
        sb = get_supabase_admin()
        rrow = (sb.table("regime_history").select("regime,vix")
                .order("detected_at", desc=True).limit(1).execute().data or [])
        if rrow and rrow[0].get("regime"):
            facts["regime"] = {"market": rrow[0].get("regime"), "vix": rrow[0].get("vix")}
    except Exception as e:
        logger.debug("market_explainer regime facts failed: %s", e)

    # Market Pulse — internals, flow streaks, FII index-futures positioning,
    # vol read and the vs-yesterday diff (all EOD-derived, 10-min cached).
    try:
        from ..scanners.market_pulse import market_pulse
        p = market_pulse() or {}
        pb = p.get("breadth") or {}
        if pb.get("score") is not None:
            facts["internals"] = {
                "breadth_score": pb.get("score"), "band": pb.get("band"),
                "pct_above_50dma": pb.get("pct_above_50dma"),
                "pct_above_200dma": pb.get("pct_above_200dma"),
                "new_52w_highs": pb.get("new_highs"), "new_52w_lows": pb.get("new_lows"),
            }
        fl = p.get("flows") or {}
        if fl.get("fii") or fl.get("dii"):
            facts["flow_streaks"] = {k: fl.get(k) for k in ("fii", "dii", "last_date")}
        pos = p.get("positioning")
        if pos:
            facts["fii_index_futures"] = {
                "net_contracts": pos.get("net"), "long_share_pct": pos.get("long_share_pct"),
                "net_delta_vs_prev": pos.get("net_delta"), "as_of": pos.get("date"),
            }
        v = p.get("vol") or {}
        if v.get("vix") is not None:
            facts["volatility"] = {"india_vix": v.get("vix"),
                                   "nifty_hv20": (v.get("hv") or {}).get("20"),
                                   "read": v.get("read")}
        if p.get("diff"):
            facts["changed_vs_yesterday"] = [
                f"{d.get('label')} ({d.get('detail')})" for d in p["diff"]
            ]
    except Exception as e:
        logger.debug("market_explainer pulse facts failed: %s", e)

    # Yesterday's settled EOD movers (candle store — SEBI-safe).
    try:
        from ..briefing.market_briefing import _eod_movers
        movers = _eod_movers(3)
        if movers:
            facts["eod_movers"] = movers
    except Exception as e:
        logger.debug("market_explainer movers facts failed: %s", e)

    return facts


def build_drivers(facts: Dict[str, Any]) -> List[str]:
    """Deterministic plain bullet drivers — always available, 0 tokens. Pure."""
    out: List[str] = []
    n = facts.get("nifty") or {}
    if n.get("change_pct") is not None:
        out.append(f"NIFTY {'+' if n['change_pct'] >= 0 else ''}{n['change_pct']}% today.")
    b = facts.get("breadth") or {}
    if b.get("adv") is not None and b.get("dec") is not None:
        tone = "positive" if b["adv"] >= b["dec"] else "negative"
        pct = f" ({b['adv_pct']}% advancing)" if b.get("adv_pct") is not None else ""
        out.append(f"Breadth {tone}: {b['adv']} adv / {b['dec']} dec{pct}.")
    s = facts.get("sectors") or {}
    if s.get("leading"):
        out.append(f"Leading: {', '.join(s['leading'])}.")
    if s.get("lagging"):
        out.append(f"Lagging: {', '.join(s['lagging'])}.")
    rg = facts.get("regime") or {}
    if rg.get("market"):
        tail = f" (VIX {rg['vix']})" if rg.get("vix") is not None else ""
        out.append(f"Regime: {str(rg['market']).capitalize()}{tail}.")
    itn = facts.get("internals") or {}
    if itn.get("breadth_score") is not None:
        out.append(f"Internals: breadth score {itn['breadth_score']}/100 ({itn.get('band')}); "
                   f"{itn.get('new_52w_highs', 0)} new 52w highs vs {itn.get('new_52w_lows', 0)} lows.")
    fs = facts.get("flow_streaks") or {}
    fii, dii = fs.get("fii"), fs.get("dii")
    if fii and dii:
        out.append(f"Flows: FII {fii['side']} {fii['days']} sessions ({fii['cum_cr']:+,.0f} Cr) "
                   f"vs DII {dii['side']} {dii['days']} ({dii['cum_cr']:+,.0f} Cr).")
    pos = facts.get("fii_index_futures") or {}
    if pos.get("net_contracts") is not None:
        side = "long" if pos["net_contracts"] >= 0 else "short"
        out.append(f"FII index futures: net {side} {abs(pos['net_contracts']):,} contracts "
                   f"({pos.get('long_share_pct')}% long share).")
    vv = facts.get("volatility") or {}
    if vv.get("read"):
        out.append(f"Vol: VIX {vv.get('india_vix')} vs HV20 {vv.get('nifty_hv20')} — {vv['read']}.")
    return out


def explain_market(*, use_llm: bool = False) -> Dict[str, Any]:
    """{facts, drivers, narrative}. Drivers deterministic + always returned;
    narrative is the grounded reasoner, cached per day, only when use_llm.
    Honest-empty (no drivers) when no real facts can be assembled."""
    facts = _assemble_facts()
    drivers = build_drivers(facts)
    narrative: Optional[str] = None
    if use_llm and drivers:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            facts,
            "Write 'What's happening in the market' for an Indian equities "
            "trader: the tape and internals, flows and FII positioning, "
            "volatility, what changed vs yesterday, and what to watch next "
            "session. 6-8 plain sentences.",
            # deep-reasoning tier (market_brief → LLM_DEEP_MODEL); day-cached,
            # so the strongest model costs one shared call per day.
            cache_key=f"marketexplain:v3:{date.today().isoformat()}",
            role="market_brief", system=_EXPLAIN_SYSTEM)
    return {"facts": facts, "drivers": drivers, "narrative": narrative}

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
            "Summarize today's Indian equity market in 3-4 plain sentences a "
            "beginner can follow.",
            cache_key=f"marketexplain:{date.today().isoformat()}",
            role="responder")
    return {"facts": facts, "drivers": drivers, "narrative": narrative}

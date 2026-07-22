"""Forecast Read — the AI layer for the stock page's Forecast tab (2026-07-21).

Assembles the tab's own deterministic evidence — the stock's empirical
setup base rates (with sample sizes), the Fibonacci/pivot structure from
the technical panel, 52-week anchors, ATR, and any confirmed earnings
event — and has the grounded reasoner write a probability-honest framing:
which base rate is most relevant now, what structure says about the room
to move, and the event risk. Never a price target, never buy/sell.
Cached per symbol per day.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_FORECAST_SYSTEM = (
    "You are a quant analyst writing the 'Forecast' read for one NSE stock "
    "over a days-to-weeks swing horizon. Reason ONLY over the facts JSON. "
    "You have: empirical base rates of this stock's own setups (probability "
    "of a +target% move within the horizon, WITH sample sizes), the current "
    "price structure (Fibonacci retracement, floor pivots, 52-week range, "
    "ATR), and any confirmed earnings event. Write 3-5 sentences: which base "
    "rate applies to the CURRENT state and how much weight its sample size "
    "deserves; what the structure implies about room to move and where the "
    "data says the move would stall or fail; and the event risk if one "
    "exists. Quote exact numbers. NEVER give a price target or tell the "
    "reader to buy or sell — frame everything as historical frequencies and "
    "structural observations. No emoji, no markdown."
)


def forecast_read(symbol: str, *, use_llm: bool = False, user_id: Optional[str] = None) -> Dict[str, Any]:
    """{facts, narrative} — facts always (deterministic), narrative when
    use_llm (grounded, day-cached)."""
    sym = symbol.strip().upper()
    facts: Dict[str, Any] = {"symbol": sym, "as_of": date.today().isoformat(),
                             "horizon": "swing (days to weeks)"}

    try:
        from ..scanners.probability_engine import setup_probabilities
        pb = setup_probabilities(sym)
        setups = [s for s in (pb.get("setups") or []) if s.get("prob_pct") is not None]
        if setups:
            facts["setup_base_rates"] = {
                "horizon_days": pb.get("horizon"), "target_pct": pb.get("target"),
                "setups": setups,
            }
    except Exception as e:
        logger.debug("forecast_read probabilities failed %s: %s", sym, e)

    try:
        from ..market.technical_panel import technical_panel
        tp = technical_panel(sym)
        if tp.get("available"):
            facts["structure"] = {
                "price": tp.get("price"),
                "fibonacci": tp.get("fibonacci"),
                "pivots": tp.get("pivots"),
                "week52": tp.get("week52"),
                "atr_pct_per_day": (tp.get("atr") or {}).get("pct"),
                "nearest_support": (tp.get("supports") or [{}])[0] or None,
                "nearest_resistance": (tp.get("resistances") or [{}])[0] or None,
            }
    except Exception as e:
        logger.debug("forecast_read structure failed %s: %s", sym, e)

    try:
        from ..news.earnings_preview import preview
        ep = preview(sym, use_llm=False)
        ef = (ep or {}).get("facts") or {}
        if (ef.get("earnings") or {}).get("announce_date"):
            facts["earnings_event"] = {
                **ef["earnings"],
                **({"volatility": ef["volatility"]} if ef.get("volatility") else {}),
            }
    except Exception as e:
        logger.debug("forecast_read earnings failed %s: %s", sym, e)

    narrative: Optional[str] = None
    if use_llm and (facts.get("setup_base_rates") or facts.get("structure")):
        try:
            from ...ai.agents.grounded import grounded_reason
            narrative = grounded_reason(
                facts,
                f"Write the probability-honest Forecast read for {sym}: the most "
                "relevant base rate and its sample-size weight, what the price "
                "structure implies, and the event risk.",
                cache_key=f"forecastread:v1:{sym}:{date.today().isoformat()}",
                system=_FORECAST_SYSTEM, user_id=user_id)
        except Exception as e:
            logger.debug("forecast_read narrative failed %s: %s", sym, e)

    return {"symbol": sym, "facts": facts, "narrative": narrative}

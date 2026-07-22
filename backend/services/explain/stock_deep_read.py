"""AI Trade Desk — per-symbol deep-reasoning synthesis (2026-07-21).

The stock-page hero: assemble EVERY deterministic read the platform already
computes for one symbol (fused verdict + factor leans, day-move facts,
volume/delivery intelligence, CVD proxy, relative strength, empirical setup
base rates, cached fundamentals) into one facts JSON, then have the
deep-reasoning tier (role="market_brief" → LLM_DEEP_MODEL) write a
PM-grade read: setup, evidence weighing with contradictions called out,
scenario map, biggest risk, what to watch.

Cost model: the narrative is generated ONLY on explicit request
(`generate=True`) and cached per symbol per day, so page loads cost zero
LLM tokens and repeat visitors share one call. Facts + deterministic
drivers are always available for free.

SEBI posture: synthesis over EOD published data + our own model outputs;
prose is analysis with scenarios/risks — never assured returns, never a
buy/sell instruction. Engines referenced by public names only.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DESK_SYSTEM = (
    "You are the senior desk PM of an Indian-equities swing desk, writing the "
    "daily 'AI Trade Desk' deep read for ONE NSE stock over a days-to-weeks "
    "horizon. Reason ONLY over the REAL facts provided as JSON — never invent "
    "prices, levels, news, or numbers, and when you cite a figure use the "
    "EXACT value from the facts. Weigh the evidence like a portfolio manager: "
    "say which factors agree, and explicitly name the strongest CONTRADICTION "
    "in the data (there almost always is one). Structure the read as short "
    "labelled paragraphs, each starting with exactly one of these labels on "
    "its own: 'Setup:', 'Evidence:', 'Scenarios:', 'Risk:', 'Watch:'. "
    "Setup: one-sentence read plus a conviction word (constructive / mixed / "
    "cautious / avoid-for-now) grounded in the composite score. Evidence: the "
    "2-3 strongest agreeing facts with their exact numbers, then the biggest "
    "contradiction. Scenarios: the bull path and the bear path, each tied to "
    "concrete facts (relative-strength windows, base rates WITH their sample "
    "sizes, volume/delivery behaviour) — describe what confirmation or "
    "invalidation would look like in the data, not price targets. Risk: the "
    "single biggest risk (regime, event, crowding, thin data). Watch: what to "
    "check next session. Skip any theme the facts don't cover. Do NOT promise "
    "or imply any return or assured outcome, and do NOT instruct the reader "
    "to buy or sell — describe evidence, scenarios, and risks only. NO emoji. "
    "Refer to internal models only as Alpha, Mood, or Regime. 10-14 sentences "
    "total across the five paragraphs."
)


def _cache_key(symbol: str) -> str:
    return f"deepread:v1:{symbol}:{date.today().isoformat()}"


def _assemble_facts(symbol: str) -> Dict[str, Any]:
    """Gather every deterministic per-symbol read. Best-effort per factor —
    a failed source is omitted, never fabricated."""
    sym = symbol.strip().upper()
    facts: Dict[str, Any] = {
        "symbol": sym,
        "as_of": date.today().isoformat(),
        "data_basis": "EOD published data + internal model outputs",
        "horizon": "swing (days to weeks)",
    }

    # Fused verdict — composite score + per-factor leans (Alpha/trend/mood/regime…).
    try:
        from ..scanners.fusion_verdict import build_verdict
        v = build_verdict(sym, use_llm=False)
        facts["fused_verdict"] = {
            k: v.get(k) for k in ("verdict", "composite", "direction", "gated", "factors", "note")
            if v.get(k) is not None
        }
    except Exception as e:
        logger.debug("deep_read verdict failed for %s: %s", sym, e)

    # Today's move facts — price, volume vs avg, RS, regime context.
    try:
        from .why_moving import assemble_facts as _move_facts
        facts["day_move"] = _move_facts(sym)
    except Exception as e:
        logger.debug("deep_read day_move failed for %s: %s", sym, e)

    # Volume + delivery intelligence (NSE delivery % — accumulation vs churn).
    try:
        from ..market.volume_intelligence import volume_intel
        vi = volume_intel(sym, use_llm=False)
        facts["volume_delivery"] = {
            k: vi.get(k) for k in ("x_avg", "vol_percentile", "delivery_today", "delivery_trend", "signal")
            if vi.get(k) is not None
        }
    except Exception as e:
        logger.debug("deep_read volume_intel failed for %s: %s", sym, e)

    # Cumulative volume delta proxy (bar-level, honest label).
    try:
        from ..market.footprint import footprint
        fp = footprint(sym, days=60)
        latest = fp.get("latest") or {}
        if latest:
            facts["cvd_proxy"] = {
                "trend": fp.get("trend"),
                "today_buy_pct": latest.get("buy_pct"),
                "note": "bar-level proxy (close-location x volume), not tick data",
            }
    except Exception as e:
        logger.debug("deep_read footprint failed for %s: %s", sym, e)

    # Relative strength vs NIFTY across 20/50/120 sessions.
    try:
        from ..scanners.relative_strength import symbol_rs
        rs = symbol_rs(sym)
        if any(rs.get(k) is not None for k in ("rs_20d", "rs_50d", "rs_120d")):
            facts["relative_strength_vs_nifty_pct"] = {
                k: rs.get(k) for k in ("rs_20d", "rs_50d", "rs_120d", "outperforming")
            }
    except Exception as e:
        logger.debug("deep_read rs failed for %s: %s", sym, e)

    # Empirical setup base rates — how often THIS stock's setups followed through.
    try:
        from ..scanners.probability_engine import setup_probabilities
        pb = setup_probabilities(sym)
        setups = [s for s in (pb.get("setups") or []) if s.get("prob_pct") is not None]
        if setups:
            facts["setup_base_rates"] = {
                "horizon_days": pb.get("horizon"),
                "target_pct": pb.get("target"),
                "setups": setups,
            }
    except Exception as e:
        logger.debug("deep_read probabilities failed for %s: %s", sym, e)

    # Cached fundamentals snapshot (screener.in-sourced, EOD).
    try:
        from ...core.database import get_supabase_admin
        rows = (
            get_supabase_admin().table("fundamentals_history")
            .select("snapshot_date,pe,roe,roce,market_cap_cr,sales_growth,profit_growth,promoter_pct")
            .eq("symbol", sym).order("snapshot_date", desc=True).limit(1).execute()
        ).data or []
        if rows:
            r = rows[0]
            facts["fundamentals"] = {k: r.get(k) for k in (
                "snapshot_date", "pe", "roe", "roce", "market_cap_cr",
                "sales_growth", "profit_growth", "promoter_pct",
            ) if r.get(k) is not None}
    except Exception as e:
        logger.debug("deep_read fundamentals failed for %s: %s", sym, e)

    return facts


def build_drivers(facts: Dict[str, Any]) -> List[str]:
    """Deterministic evidence bullets — always returned, zero tokens."""
    out: List[str] = []
    fv = facts.get("fused_verdict") or {}
    if fv.get("composite") is not None:
        out.append(f"Fused verdict {fv.get('verdict')} · composite {fv.get('composite')}/100 ({fv.get('direction')})")
    rs = facts.get("relative_strength_vs_nifty_pct") or {}
    if rs.get("rs_50d") is not None:
        tag = "outperforming" if rs.get("outperforming") else "lagging"
        out.append(f"RS vs NIFTY: 20d {rs.get('rs_20d')}% · 50d {rs.get('rs_50d')}% — {tag}")
    vd = facts.get("volume_delivery") or {}
    if vd.get("x_avg") is not None:
        line = f"Volume {vd.get('x_avg')}x avg"
        if vd.get("delivery_today") is not None:
            line += f" · delivery {vd['delivery_today']}%"
        out.append(f"{line} ({vd.get('signal')})")
    cvd = facts.get("cvd_proxy") or {}
    if cvd.get("trend"):
        out.append(f"CVD proxy {cvd['trend']} · today {cvd.get('today_buy_pct')}% buy-side")
    sb = facts.get("setup_base_rates") or {}
    best = max((sb.get("setups") or []), key=lambda s: s.get("prob_pct") or 0, default=None)
    if best:
        out.append(
            f"Best base rate: {best.get('name', best.get('setup', 'setup'))} "
            f"{best.get('prob_pct')}% over {best.get('occurrences')} occurrences"
        )
    fn = facts.get("fundamentals") or {}
    if fn.get("pe") is not None:
        out.append(f"Fundamentals: P/E {fn.get('pe')} · ROE {fn.get('roe')}% · ROCE {fn.get('roce')}%")
    return out


def deep_read(symbol: str, *, generate: bool = False, user_id: Optional[str] = None) -> Dict[str, Any]:
    """{facts, drivers, narrative, generated}. Narrative only when `generate`
    (or already cached today); facts + drivers always, deterministically."""
    sym = symbol.strip().upper()
    key = _cache_key(sym)

    # A cached narrative is free to serve regardless of the generate flag.
    from ...ai.agents.response_cache import cache_get
    cached = cache_get(key)
    cached_answer = (cached or {}).get("answer")

    facts = _assemble_facts(sym)
    drivers = build_drivers(facts)

    narrative: Optional[str] = cached_answer
    if narrative is None and generate and drivers:
        from ...ai.agents.grounded import grounded_reason
        narrative = grounded_reason(
            facts,
            f"Write the AI Trade Desk deep read for {sym}: setup and "
            "conviction, evidence with the biggest contradiction, bull and "
            "bear scenarios grounded in the exact numbers, the biggest risk, "
            "and what to watch next session.",
            cache_key=key,
            role="market_brief", system=_DESK_SYSTEM, user_id=user_id)

    return {
        "symbol": sym,
        "facts": facts,
        "drivers": drivers,
        "narrative": narrative,
        "generated": narrative is not None,
        "from_cache": cached_answer is not None,
    }

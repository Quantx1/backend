"""
Tool registry — callables that agents can invoke to fetch real data from
the app database / market-data providers.

Registering a tool:

    @tool(name="get_portfolio", description="Returns the user's open paper + live positions")
    async def _get_portfolio(user_id: str) -> Dict[str, Any]:
        ...

Every tool:
- is async
- takes JSON-serializable kwargs
- returns a JSON-serializable dict
- never raises — catch + return ``{"error": "..."}``

The Copilot planner Agent reads ``tool_registry.schema()`` to know what's
available and builds an LLM prompt (sent through the OpenRouter gateway)
instructing the model to emit ``{"tool": "...", "args": {...}}`` objects.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .state import AgentState, ToolCall

logger = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[Dict[str, Any]]]


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, str]  # param name → one-line description
    fn: ToolFn


@dataclass
class ToolRegistry:
    tools: Dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

    def schema(self) -> List[Dict[str, Any]]:
        """Return an LLM-prompt-friendly JSON schema of available tools."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "params": spec.params,
            }
            for spec in self.tools.values()
        ]

    async def call(
        self,
        state: AgentState,
        name: str,
        **args: Any,
    ) -> Dict[str, Any]:
        spec = self.get(name)
        started = datetime.utcnow()
        t0 = time.monotonic()
        if spec is None:
            call = ToolCall(
                name=name, args=args, result=None,
                started_at=started, duration_ms=0,
                error=f"unknown tool: {name}",
            )
            state.tool_trace.append(call)
            return {"error": f"unknown tool: {name}"}

        try:
            result = await spec.fn(**args)
            call = ToolCall(
                name=name, args=args, result=result,
                started_at=started,
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
            state.tool_trace.append(call)
            return result
        except Exception as e:
            logger.warning("Tool %s failed: %s", name, e)
            call = ToolCall(
                name=name, args=args, result=None,
                started_at=started,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error=str(e),
            )
            state.tool_trace.append(call)
            return {"error": str(e)}


# Module-level singleton.
tool_registry = ToolRegistry()


def tool(*, name: str, description: str, params: Optional[Dict[str, str]] = None):
    """Decorator — register an async function as a callable tool."""
    def wrap(fn: ToolFn) -> ToolFn:
        tool_registry.register(
            ToolSpec(
                name=name,
                description=description,
                params=params or {},
                fn=fn,
            )
        )

        @functools.wraps(fn)
        async def _wrapped(*a, **kw):
            return await fn(*a, **kw)

        return _wrapped

    return wrap


# ============================================================================
# BUILT-IN TOOLS — concrete data fetchers the 3 graphs need on day 1.
# ============================================================================
# These read from Supabase using the admin client so the agent sidesteps
# per-user RLS auth (the caller has already been authenticated at the API
# boundary — we pass user_id into each call).
# ============================================================================


def _client():
    from ...core.database import get_supabase_admin
    return get_supabase_admin()


def _num(v: Any) -> Optional[float]:
    """Coerce to float, dropping None / NaN / junk. (Local copy — importing
    the copilot.py sibling would be circular; it imports tool_registry here.)"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


@tool(
    name="get_portfolio",
    description="Return the user's open paper + live positions with current LTP, entry, PnL.",
    params={"user_id": "Supabase auth.users.id of the user"},
)
async def _get_portfolio(user_id: str) -> Dict[str, Any]:
    client = _client()
    live = client.table("positions").select(
        "symbol, quantity, entry_price, current_price, pnl, pnl_percent, product"
    ).eq("user_id", user_id).eq("status", "open").execute()
    paper = client.table("paper_positions").select(
        "symbol, qty, entry_price, entry_date, status, stop_loss, target"
    ).eq("user_id", user_id).eq("status", "open").execute()
    return {
        "live_positions": live.data or [],
        "paper_positions": paper.data or [],
    }


@tool(
    name="get_watchlist",
    description="Return the user's watched symbols.",
    params={"user_id": "Supabase auth.users.id of the user"},
)
async def _get_watchlist(user_id: str) -> Dict[str, Any]:
    client = _client()
    rows = client.table("watchlist").select("symbol, added_at").eq(
        "user_id", user_id
    ).execute()
    return {"watchlist": rows.data or []}


@tool(
    name="get_signal",
    description="Return full signal details (price levels, model scores, regime at signal) for one signal id.",
    params={"signal_id": "UUID of the signal"},
)
async def _get_signal(signal_id: str) -> Dict[str, Any]:
    client = _client()
    rows = client.table("signals").select("*").eq("id", signal_id).limit(1).execute()
    data = (rows.data or [None])[0]
    if data is None:
        return {"error": f"signal {signal_id} not found"}
    return {"signal": data}


@tool(name="get_todays_signals",
      description="Return today's active signals across all users (max 20). Useful for Copilot 'what should I look at today' queries.",
      params={"max_n": "Maximum signals to return (default 20)"},
      )
async def _get_todays_signals(max_n: int = 20) -> Dict[str, Any]:
    client = _client()
    from datetime import date

    today = date.today().isoformat()
    rows = (
        client.table("signals")
        .select("symbol, direction, confidence, entry_price, target_1, stop_loss, regime_at_signal, strategy_names")
        .eq("date", today)
        .eq("status", "active")
        .order("confidence", desc=True)
        .limit(int(max_n))
        .execute()
    )
    return {"signals": rows.data or []}


@tool(
    name="get_stock_snapshot",
    description="Return recent OHLCV + indicator snapshot for a symbol (last 60 trading days).",
    params={"symbol": "NSE ticker, e.g. TCS or RELIANCE"},
)
async def _get_stock_snapshot(symbol: str) -> Dict[str, Any]:
    try:
        from ...data.market import get_market_data_provider
        provider = get_market_data_provider()
        df = provider.get_historical(symbol.upper(), period="3mo", interval="1d")
        if df is None or len(df) == 0:
            return {"error": f"no data for {symbol}"}
        df = df.tail(60).copy()
        df.columns = [c.lower() for c in df.columns]
        last = df.iloc[-1]
        first = df.iloc[0]
        pct_3m = ((last["close"] - first["close"]) / first["close"] * 100) if first["close"] else 0
        # Closing-price series (≤60 pts) so the copilot can render a real price
        # sparkline artifact in chat — actual data, not a synthetic curve.
        series = [round(float(c), 2) for c in df["close"].tolist() if c == c]
        return {
            "symbol": symbol.upper(),
            "last_close": float(last["close"]),
            "last_volume": float(last.get("volume", 0) or 0),
            "high_3m": float(df["high"].max()),
            "low_3m": float(df["low"].min()),
            "pct_change_3m": round(float(pct_3m), 2),
            "bars": len(df),
            "series": series,
        }
    except Exception as e:
        return {"error": str(e)}


@tool(
    name="explain_move",
    description=(
        "Explain WHY a stock is moving today — fuses price action, volume vs "
        "20-day average, futures OI build-up, relative strength vs NIFTY, and "
        "market regime into grounded drivers. Use for 'why is X moving/up/down "
        "today', 'what's driving X'. Returns { drivers[], facts }."
    ),
    params={"symbol": "NSE ticker, e.g. RELIANCE or TCS"},
)
async def _explain_move(symbol: str) -> Dict[str, Any]:
    try:
        import asyncio
        from ...services.explain.why_moving import explain_move
        # use_llm=False: the copilot's own Responder narrates from these drivers,
        # so we don't double-spend on the model.
        res = await asyncio.to_thread(explain_move, symbol, use_llm=False)
        return {"symbol": res["symbol"], "drivers": res["drivers"], "facts": res["facts"]}
    except Exception as e:
        return {"error": str(e)}


@tool(
    name="get_current_regime",
    description="Return the current market regime (bull/sideways/bear) + probabilities + nifty_close + vix.",
    params={},
)
async def _get_current_regime() -> Dict[str, Any]:
    client = _client()
    rows = (
        client.table("regime_history")
        .select("regime, prob_bull, prob_sideways, prob_bear, vix, nifty_close, detected_at")
        .order("detected_at", desc=True)
        .limit(1)
        .execute()
    )
    row = (rows.data or [None])[0]
    if row is None:
        return {"regime": "bull", "prob_bull": 1.0, "source": "fallback"}
    return row


@tool(
    name="suggest_options_strategy",
    description=(
        "Recommend one multi-leg options structure (Bull Call Spread, "
        "Iron Condor, Long Straddle, etc.) given the user's market view. "
        "Use when the user asks for an 'options play', wants to 'hedge "
        "my book', asks for an 'income strategy', or specifically asks for "
        "an options recommendation. Always pass include_portfolio=true when "
        "the user mentions hedging, current positions, or 'my book'. "
        "Output: { template, lots_suggestion, reasoning, expected_outcome, "
        "risk_summary, proposal: { legs[], net_premium, max_profit, max_loss } }"
    ),
    params={
        "user_id": "Supabase auth.users.id of the user",
        "view": "User's market view (e.g. 'bullish nifty next week', 'range-bound')",
        "symbol": "NIFTY | BANKNIFTY | FINNIFTY (default NIFTY)",
        "include_portfolio": "true to feed the user's open positions into the prompt for hedge sizing",
    },
)
async def _suggest_options_strategy(
    user_id: str,
    view: str,
    symbol: str = "NIFTY",
    include_portfolio: bool = False,
) -> Dict[str, Any]:
    """Copilot wrapper around the F&O AI advisor.

    Shape mirrors the /api/fo-strategies/ai-suggest response so the
    copilot orchestrator can ingest + summarise it consistently.
    """
    # Lazy import — avoids a startup-time circular through fo_strategies_routes
    from ...api.fo_strategies_routes import (
        AISuggestRequest, ai_suggest_strategy,
    )
    from ...core.tiers import UserTier, Tier

    # Build a stub UserTier — the advisor only reads .user_id for context,
    # and the route's tier gate is bypassed because we're calling the
    # function directly (the copilot orchestrator already gates on tier
    # at /api/ai/copilot/chat).
    stub_user = UserTier(user_id=user_id, tier=Tier.ELITE, is_admin=False)
    body = AISuggestRequest(
        prompt=view, symbol=symbol, include_portfolio=bool(include_portfolio),
    )
    try:
        return await ai_suggest_strategy(body=body, user=stub_user)
    except Exception as e:
        logger.debug("suggest_options_strategy failed: %s", e)
        return {"error": str(e)[:200]}


# ============================================================================
# PHASE 2 — CHAT SUPERPOWERS (2026-07-11)
# Read-only tools wrapping endpoints/services the app already ships, so the
# copilot can answer the full uTrade-Intelligence capability list end-to-end:
# fundamentals, technicals, news sentiment, F&O chain (PCR/max-pain/OI S-R),
# sector rotation, FII/DII flow, NL screening (with result rows), and the
# user's own strategies + deploy status. Each degrades to {error/available}
# honestly — the Responder narrates qualitatively when data is absent.
# ============================================================================


@tool(
    name="get_fundamentals",
    description=(
        "Return a stock's fundamentals — PE, ROE, ROCE, market cap (₹ cr), book "
        "value, dividend yield, sales/profit growth, promoter holding %. Use for "
        "'what is X's ROE', 'is X cheap', 'fundamentals of X', or comparing two "
        "stocks fundamentally (call once per symbol). Returns { fundamentals } or "
        "{ available:false } when the symbol isn't covered."
    ),
    params={"symbol": "NSE ticker, e.g. RELIANCE, TCS, INFY"},
)
async def _get_fundamentals(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"available": False, "symbol": symbol}
    keys = ("pe", "roe", "roce", "market_cap_cr", "book_value", "dividend_yield",
            "current_price", "sales_growth", "profit_growth", "promoter_pct")
    try:
        client = _client()
        rows = (
            client.table("fundamentals_history")
            .select("snapshot_date, " + ", ".join(keys))
            .eq("symbol", sym)
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
            .data
        ) or []
        if not rows:
            return {"available": False, "symbol": sym}
        row = rows[0]
        fundamentals = {k: _num(row.get(k)) for k in keys}
        return {
            "available": True,
            "symbol": sym,
            "as_of": row.get("snapshot_date"),
            "fundamentals": fundamentals,
        }
    except Exception as e:
        return {"error": str(e)[:200], "symbol": sym}


@tool(
    name="get_technicals",
    description=(
        "Return a stock's technical-indicator snapshot — RSI, MACD (+signal), "
        "SMA 20/50/200, EMA 21, ADX, ATR, Bollinger bands, volume ratio, and a "
        "plain-English trend label. Use for 'is X overbought', 'technicals on X', "
        "'RSI of X', 'trend on X'."
    ),
    params={"symbol": "NSE ticker, e.g. RELIANCE, TCS"},
)
async def _get_technicals(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "no symbol"}
    try:
        from ...api.screener_routes import get_stock_technicals
        res = await get_stock_technicals(symbol=sym)
        if not res.get("success"):
            return {"error": res.get("error") or "no technicals", "symbol": sym}
        return res
    except Exception as e:
        return {"error": str(e)[:200], "symbol": sym}


@tool(
    name="get_news_sentiment",
    description=(
        "Return recent news sentiment for a stock — a materiality-weighted mood "
        "score + label, de-duplicated top stories with event type / impact / "
        "sentiment, and the single most-material story. Deterministic (0 LLM "
        "tokens); the assistant narrates. Use for 'any news on X', 'why is X in "
        "the news', 'sentiment on X'. Returns { available, mood_score, label, "
        "top_story, stories[] }."
    ),
    params={"symbol": "NSE ticker, e.g. RELIANCE, TCS"},
)
async def _get_news_sentiment(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"available": False, "symbol": symbol}
    try:
        from ...services.news.news_intelligence import analyze
        res = await analyze(sym, use_llm=False, use_narrative=False)
        stories = [
            {"title": s.get("title"), "event": s.get("event_label"),
             "impact": s.get("impact"), "sentiment": s.get("label")}
            for s in (res.get("stories") or [])[:5]
        ]
        top = res.get("top_story") or {}
        return {
            "available": bool(res.get("available")),
            "symbol": sym,
            "mood_score": _num(res.get("mood_score")),
            "label": res.get("label"),
            "story_count": res.get("story_count", 0),
            "top_story": top.get("title") if isinstance(top, dict) else None,
            "event_breakdown": res.get("event_breakdown") or [],
            "stories": stories,
        }
    except Exception as e:
        return {"error": str(e)[:200], "symbol": sym}


@tool(
    name="get_fno_snapshot",
    description=(
        "Return the F&O option-chain read for an index or F&O stock — PCR (OI) "
        "with a bias tag, max pain + distance from spot, the top call-OI strikes "
        "(resistance) and put-OI strikes (support), IV, and a plain-English teach "
        "summary. Use for 'NIFTY max pain', 'BANKNIFTY PCR', 'where's the OI "
        "support', 'option chain read for X'. Valid: NIFTY, BANKNIFTY, FINNIFTY, "
        "MIDCPNIFTY, or an F&O stock (not SENSEX)."
    ),
    params={"symbol": "NIFTY | BANKNIFTY | FINNIFTY | MIDCPNIFTY | F&O stock ticker"},
)
async def _get_fno_snapshot(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "no symbol"}
    try:
        import asyncio
        from ...services.fno_scanner.snapshot import fetch_index_snapshot, teach_snapshot
        snap = await asyncio.to_thread(fetch_index_snapshot, sym)
        if snap is None:
            return {"error": "option chain unavailable for this symbol", "symbol": sym}
        d = snap.to_dict()
        d["teach"] = teach_snapshot(d)
        return d
    except Exception as e:
        return {"error": str(e)[:200], "symbol": sym}


@tool(
    name="get_sector_performance",
    description=(
        "Return sector rotation across the NSE — each sector's short (~5d) and "
        "long (~20d) relative strength vs the market and its RRG quadrant "
        "(leading / weakening / lagging / improving). Use for 'which sectors are "
        "strong', 'sector rotation', 'where is money flowing', 'best/worst "
        "sectors'. Returns { sectors:[{sector, quadrant, rs_short, rs_long}] }."
    ),
    params={},
)
async def _get_sector_performance() -> Dict[str, Any]:
    try:
        import asyncio
        from ...services.scanners.sector_rotation import sector_rotation
        rows = await asyncio.to_thread(sector_rotation)
        return {"sectors": rows or []}
    except Exception as e:
        return {"error": str(e)[:200]}


@tool(
    name="get_fii_dii_flow",
    description=(
        "Return the latest cash-market FII vs DII net flow in ₹ crore (net = buy "
        "− sell; FII net < 0 = foreigners sold, DII net > 0 = domestics bought). "
        "Nightly EOD data, not live. Use for 'what did FII do today', 'FII DII "
        "flow', 'is smart money buying'."
    ),
    params={},
)
async def _get_fii_dii_flow() -> Dict[str, Any]:
    try:
        client = _client()
        latest = (
            client.table("fii_dii_flow_eod").select("date")
            .eq("segment", "CASH").order("date", desc=True).limit(1)
            .execute().data
        ) or []
        if not latest:
            return {"available": False}
        as_of = latest[0]["date"]
        rows = (
            client.table("fii_dii_flow_eod")
            .select("fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net")
            .eq("segment", "CASH").eq("date", as_of).limit(1)
            .execute().data
        ) or []
        if not rows:
            return {"available": False}
        r = rows[0]
        return {
            "available": True,
            "as_of": as_of,
            "segment": "CASH",
            "fii_net": _num(r.get("fii_net")),
            "dii_net": _num(r.get("dii_net")),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


@tool(
    name="run_screen",
    description=(
        "Run a natural-language stock screen — maps free text to real scanners "
        "(rules first, LLM for nuance) and returns the matching NSE stocks with "
        "LTP, % change, RSI and how many setups each fired. Use when the user "
        "describes a setup to find: 'oversold largecaps in an uptrend', "
        "'breakouts on heavy volume', 'high momentum names making new highs'. "
        "Returns { recognized, scanners_used, count, results[] }."
    ),
    params={
        "query": "Plain-English screen description",
        "limit": "Max results to return (default 15)",
    },
)
async def _run_screen(query: str, limit: int = 15) -> Dict[str, Any]:
    q = (query or "").strip()
    if len(q) < 2:
        return {"error": "empty query"}
    try:
        import asyncio
        from ...services.screener_v2.nl_screen import resolve_screen_query, scanner_label
        from ...services.screener_v2 import confluence_scan
        from ...data.screener.engine import NSE_STOCK_INFO, get_live_screener
        from ...api.screener_routes import _computed_data_or_503

        resolved = resolve_screen_query(q)
        ids = resolved["scanner_ids"]
        if not ids:
            return {"recognized": False, "query": q, "count": 0, "results": [],
                    "scanners_used": []}
        screener = get_live_screener()
        try:
            summary_df, _ = await _computed_data_or_503(screener)
        except Exception:
            summary_df = None
        if summary_df is None or getattr(summary_df, "empty", True):
            return {"error": "screener data not ready", "recognized": True, "query": q,
                    "scanners_used": [{"id": i, "name": scanner_label(i)} for i in ids]}
        matches = await asyncio.to_thread(
            confluence_scan, summary_df, scanner_ids=ids, stock_info=NSE_STOCK_INFO,
            min_hits=1, limit=int(limit),
        )
        results = [
            {
                "symbol": m.symbol, "name": m.name, "sector": m.sector,
                "last_price": round(float(m.last_price), 2),
                "change_pct": round(float(m.change_pct), 2),
                "rsi": round(float(m.rsi), 1),
                "hit_count": m.hit_count,
            }
            for m in matches
        ]
        return {
            "recognized": True, "query": q, "source": resolved["source"],
            "scanners_used": [{"id": i, "name": scanner_label(i)} for i in ids],
            "count": len(results), "results": results,
        }
    except Exception as e:
        return {"error": str(e)[:200], "query": q}


@tool(
    name="get_ipo_calendar",
    description=(
        "Return the Indian IPO calendar — currently-OPEN issues (with their live "
        "subscription multiple, price band, dates) and UPCOMING issues. Use for "
        "'which IPOs are open now', 'upcoming IPOs', 'IPO calendar', 'is X IPO "
        "subscribed'. Source: NSE primary-market feed. Note: does NOT include GMP "
        "(grey-market premium) — that is unofficial data we do not provide."
    ),
    params={},
)
async def _get_ipo_calendar() -> Dict[str, Any]:
    try:
        import asyncio
        from ...services.ipo.ipo_calendar import fetch_ipo_calendar
        data = await asyncio.to_thread(fetch_ipo_calendar)
        # Trim to what the responder + artifact need.
        def _slim(x):
            return {
                "symbol": x.get("symbol"), "company": x.get("company"),
                "price_band": x.get("price_band"), "open_date": x.get("open_date"),
                "close_date": x.get("close_date"), "status": x.get("status"),
                "subscription_x": x.get("subscription_x"),
            }
        return {
            "available": data.get("available", False),
            "open": [_slim(x) for x in (data.get("open") or [])],
            "upcoming": [_slim(x) for x in (data.get("upcoming") or [])],
            "note": data.get("note"),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


@tool(
    name="run_fundamental_screen",
    description=(
        "Screen stocks by FUNDAMENTALS (not price/technicals) — PE, ROE, ROCE, "
        "dividend yield, sales/profit growth, promoter holding, and a 0-5 Quality "
        "Score. Use for 'find low PE value stocks', 'high ROCE quality companies', "
        "'debt-free stocks', 'dividend payers', 'quality compounders', 'high "
        "growth stocks', 'high promoter holding'. Pass a preset key. Returns "
        "{ name, count, results:[{symbol, pe, roe, roce, ..., quality_score}], note }."
    ),
    params={
        "preset": (
            "One of: low-pe-value | high-roce-quality | quality-compounder | "
            "high-growth | dividend-payer | promoter-backed | quality-score | low-debt"
        ),
        "limit": "Max results (default 15)",
    },
)
async def _run_fundamental_screen(preset: str, limit: int = 15) -> Dict[str, Any]:
    try:
        import asyncio
        from ...services.screener_v2.fundamental_screen import run_fundamental_screen
        res = await asyncio.to_thread(
            run_fundamental_screen, _client(), preset=str(preset), limit=int(limit),
        )
        return res
    except Exception as e:
        return {"error": str(e)[:200], "preset": preset}


_OP_WORDS = {
    "<": "drops below", ">": "rises above", "<=": "is at or below",
    ">=": "is at or above", "==": "equals", "crosses_above": "crosses above",
    "crosses_below": "crosses below", "between": "is between", "outside": "is outside",
}


def _summarize_condition(c: Optional[Dict[str, Any]]) -> str:
    """Compact plain-English of a DSL Condition (mirrors the frontend
    dsl-plain renderer) for the strategy-card artifact."""
    if not isinstance(c, dict):
        return "—"
    kind = c.get("kind")
    if kind in ("composite_and", "composite_or"):
        joiner = " and " if kind == "composite_and" else " or "
        parts = [_summarize_condition(ch) for ch in (c.get("children") or [])]
        return joiner.join(p for p in parts if p and p != "—")
    ind = c.get("indicator") or (c.get("engine") or "").title()
    op = _OP_WORDS.get(str(c.get("op")), str(c.get("op") or ""))
    val = c.get("value")
    if isinstance(val, list):
        val = "–".join(str(v) for v in val)
    return " ".join(str(x) for x in (ind, op, val) if x not in (None, "")).strip() or "—"


@tool(
    name="compile_strategy",
    description=(
        "Compile a plain-English trading strategy description into real, "
        "editable rules. Use when the user asks to BUILD / WRITE / CREATE a "
        "strategy, e.g. '5-min momentum strategy with RSI>50 and MACD, 1% stop "
        "2% target', 'EMA crossover swing on Reliance'. Returns the strategy "
        "name, entry/exit rules in plain English, risk (stop/target/square-off), "
        "and the DSL. It does NOT backtest or deploy — those happen in Studio "
        "behind the walk-forward Sharpe gate (there's a deep link for that). "
        "May return { needs_clarification, question } if the description is too "
        "vague."
    ),
    params={"prompt": "The user's plain-English strategy description"},
)
async def _compile_strategy(prompt: str) -> Dict[str, Any]:
    p = (prompt or "").strip()
    if len(p) < 3:
        return {"error": "describe the strategy in a sentence"}
    try:
        import asyncio
        from ...ai.strategy.studio import (
            ClarificationNeeded, compile_strategy,
            is_studio_available, precheck_clarification,
        )
        if not is_studio_available():
            return {"error": "strategy compiler unavailable"}
        pre = precheck_clarification(p)
        if pre is not None:
            return {"needs_clarification": True, "question": pre.question,
                    "missing": pre.missing}
        result = await asyncio.to_thread(compile_strategy, p)
        if isinstance(result, ClarificationNeeded):
            return {"needs_clarification": True, "question": result.question,
                    "missing": result.missing}
        dsl = result.model_dump(mode="json")
        return {
            "name": dsl.get("name"),
            "segment": dsl.get("instrument_segment"),
            "timeframe": dsl.get("timeframe"),
            "symbol": dsl.get("symbol"),
            "universe": dsl.get("universe"),
            "entry_rule": _summarize_condition(dsl.get("entry")),
            "exit_rule": _summarize_condition(dsl.get("exit")),
            "stop_loss_pct": dsl.get("stop_loss_pct"),
            "take_profit_pct": dsl.get("take_profit_pct"),
            "trailing_stop_pct": dsl.get("trailing_stop_pct"),
            "square_off_time": dsl.get("square_off_time"),
            "dsl": dsl,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


@tool(
    name="get_my_strategies",
    description=(
        "Return the user's own trading strategies with their lifecycle status "
        "(draft / backtest / paper / live / paused), backtest Sharpe where "
        "available, and whether each is deployed. Use for 'my strategies', 'is my "
        "X strategy live', 'how are my strategies doing', 'what have I built'."
    ),
    params={"user_id": "Supabase auth.users.id of the user"},
)
async def _get_my_strategies(user_id: str) -> Dict[str, Any]:
    try:
        from ...ai.strategy import registry as strat_registry
        rows = strat_registry.list_strategies(_client(), user_id=user_id, limit=50)
        strategies = []
        for r in rows:
            bt = r.get("last_backtest") or {}
            sharpe = None
            for k in ("sharpe", "sharpe_ratio", "oos_sharpe"):
                if bt.get(k) is not None:
                    sharpe = _num(bt.get(k))
                    break
            strategies.append({
                "name": r.get("name"),
                "status": r.get("status"),
                "deployed": bool(r.get("deployed_at")),
                "sharpe": sharpe,
                "last_run_at": r.get("last_run_at"),
            })
        live = sum(1 for s in strategies if s["status"] in ("live", "paper"))
        return {"count": len(strategies), "live_or_paper": live, "strategies": strategies}
    except Exception as e:
        return {"error": str(e)[:200]}

"""AI Daily Market Briefing — pre-market read + post-market wrap.

The synthesized, everyone-can-see companion to the (broker-gated) live board.
Assembled ENTIRELY from SAFE data:
  - global overnight cues (foreign / non-NSE feeds)               [1a]
  - FII/DII EOD provisional daily net (public market statistics)  [1b]
  - derived analytics (Regime, breadth, sector rotation, Mood)    our IP
  - EOD / previous-session index levels + India VIX               EOD published
  - deterministic events (index expiry)                           calendar

Every section is deterministic (0 tokens, always present, honest-empty when a
factor is genuinely missing). One OPTIONAL cached LLM narrative per
(trading-date, session) narrates over the facts — first visitor triggers it,
everyone else shares the cached result (cheap, budget-guarded). Reuses the same
grounded-reasoner + day-cache mechanism `market_explainer(use_llm=True)` uses.

SEBI note: EOD-published statistics (previous-session index levels, FII/DII
daily net) are treated here as PUBLIC market statistics, safe to show to
everyone when clearly labelled `EOD · provisional`. A SEBI-registered
professional should confirm this EOD-published-statistics line before paid /
public launch. This service NEVER exposes real-time intraday NSE quotes, depth
or live OI — those stay Path-A gated on the live board.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

_DISCLAIMER = "For educational purposes — not investment advice."
_EOD_LABEL = "EOD · provisional"

# Grounded-reasoner system prompt — SEBI-safe desk voice. No emoji, no assured
# returns, engine public names only (Alpha / Mood / Regime), never real model
# names.
_BRIEFING_SYSTEM = (
    "You are the head of a professional Indian-equities quant desk writing the "
    "daily market briefing retail traders use to orient their decisions. "
    "Reason ONLY over the REAL facts provided as JSON and NEVER invent prices, "
    "levels, or numbers that are not present; when you cite a figure use the "
    "EXACT value from the facts, and silently skip any theme the facts do not "
    "cover. Think like a portfolio manager: weigh the evidence on BOTH sides, "
    "say which factors dominate and why, and be explicit about conviction "
    "(clear / mixed / low-visibility). Structure the note as flowing prose in "
    "this order: (1) the one-line read of where the market stands, (2) the "
    "two or three decisive facts driving it, (3) the scenario map — what "
    "confirms the current read and what would flip it, framed as observable "
    "conditions from the facts (breadth, flows, levels already given), "
    "(4) the biggest risk to the read, (5) exactly what to watch next "
    "session. Voice: sharp, neutral, institutional. NO emoji or decorative "
    "symbols. Do NOT promise or imply any return, profit, or assured outcome, "
    "and do NOT tell the reader to buy or sell any specific security — "
    "describe conditions and scenarios only; the decision stays with the "
    "reader. Refer to internal models only as Alpha, Mood, or Regime; never "
    "by any other name. Plain prose, no markdown headers, 6-9 tight sentences."
)


# ── session + trading-date helpers ────────────────────────────────────────
def current_session(now_ist: Optional[datetime] = None) -> str:
    """Resolve the natural briefing session for the moment (IST).

    - 'premarket'  before 09:15, and on weekends / holidays (shows the read
                   for the next / most-recent session)
    - 'intraday'   09:15–15:30 on a trading day
    - 'postmarket' after 15:30 on a trading day
    """
    now = now_ist or datetime.now(IST)
    try:
        from ...data.market import get_market_data_provider
        is_trading = get_market_data_provider().is_trading_day(now.date())
    except Exception:
        is_trading = now.weekday() < 5
    if not is_trading:
        return "premarket"
    t = now.time()
    if t < datetime.strptime("09:15", "%H:%M").time():
        return "premarket"
    if t <= datetime.strptime("15:30", "%H:%M").time():
        return "intraday"
    return "postmarket"


def _trading_date_for(session: str, now_ist: Optional[datetime] = None) -> str:
    """ISO trading date the briefing refers to. Premarket/intraday → today if a
    trading day else the next trading day. Postmarket → today if a trading day
    else the most-recent trading day."""
    now = now_ist or datetime.now(IST)
    today = now.date()
    try:
        from ...data.market import get_market_data_provider
        provider = get_market_data_provider()
        is_trading = provider.is_trading_day(today)
    except Exception:
        provider = None
        is_trading = today.weekday() < 5

    def _is_td(d: date) -> bool:
        try:
            return provider.is_trading_day(d) if provider else d.weekday() < 5
        except Exception:
            return d.weekday() < 5

    if session == "postmarket":
        d = today
        for _ in range(10):
            if _is_td(d):
                return d.isoformat()
            d -= timedelta(days=1)
        return today.isoformat()
    # premarket / intraday
    if is_trading:
        return today.isoformat()
    d = today + timedelta(days=1)
    for _ in range(10):
        if _is_td(d):
            return d.isoformat()
        d += timedelta(days=1)
    return today.isoformat()


def _next_weekday(d: date, weekday: int) -> date:
    """Next date on/after ``d`` falling on ``weekday`` (Mon=0 … Sun=6)."""
    return d + timedelta(days=(weekday - d.weekday()) % 7)


def _last_thursday(year: int, month: int) -> date:
    """Last Thursday of a month."""
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() != 3:  # Thursday
        d -= timedelta(days=1)
    return d


def _expiry() -> Dict[str, Optional[str]]:
    """Deterministic NSE index-expiry dates (Nifty weekly = Thursday; monthly =
    last Thursday). Best-effort holiday shift to the prior trading day."""
    today = datetime.now(IST).date()
    weekly = _next_weekday(today, 3)
    monthly = _last_thursday(today.year, today.month)
    if monthly < today:
        nm = today.month % 12 + 1
        ny = today.year + (1 if today.month == 12 else 0)
        monthly = _last_thursday(ny, nm)
    try:
        from ...data.market import get_market_data_provider
        provider = get_market_data_provider()
        for _ in range(3):
            if provider.is_trading_day(weekly):
                break
            weekly -= timedelta(days=1)
        for _ in range(3):
            if provider.is_trading_day(monthly):
                break
            monthly -= timedelta(days=1)
    except Exception:
        pass
    return {"weekly": weekly.isoformat(), "monthly": monthly.isoformat()}


# ── FII/DII EOD (public statistics) ───────────────────────────────────────
def _eod_movers(limit: int = 4) -> List[Dict[str, Any]]:
    """Top EOD close-to-close gainers + losers for the latest settled session,
    straight from the candle store — SEBI-safe (settled data, no live quotes).
    Returns [{symbol, change_pct, driver:None}] matching the frontend movers
    contract; honest-empty on any failure."""
    try:
        from ...data.ohlc_store import pg_connect
        conn = pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH ranked AS (
                      SELECT stock_symbol, timestamp::date AS dt, close,
                             row_number() OVER (PARTITION BY stock_symbol
                                                ORDER BY timestamp DESC) AS rn
                      FROM candles
                      WHERE interval = '1d'
                        AND timestamp >= now() - interval '12 days'
                        AND stock_symbol NOT IN
                            ('NIFTY','BANKNIFTY','SENSEX','FINNIFTY',
                             'MIDCPNIFTY','INDIAVIX','VIX')
                    ),
                    pair AS (
                      SELECT stock_symbol,
                             max(CASE WHEN rn = 1 THEN close END) AS last,
                             max(CASE WHEN rn = 1 THEN dt END)    AS dt,
                             max(CASE WHEN rn = 2 THEN close END) AS prev
                      FROM ranked GROUP BY stock_symbol
                    )
                    SELECT stock_symbol, last, prev FROM pair
                    WHERE last IS NOT NULL AND prev IS NOT NULL AND prev > 0
                      AND dt = (SELECT max(dt) FROM pair)
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        moves = sorted(
            # psycopg2 returns NUMERIC as Decimal — cast before the float math
            ((sym, round((float(last) / float(prev) - 1.0) * 100.0, 2))
             for sym, last, prev in rows),
            key=lambda x: x[1],
        )
        if not moves:
            return []
        losers = [m for m in moves[:limit] if m[1] < 0]
        gainers = [m for m in moves[-limit:][::-1] if m[1] > 0]
        return [
            {"symbol": sym, "change_pct": pct, "driver": None}
            for sym, pct in gainers + losers
        ]
    except Exception as e:  # noqa: BLE001
        logger.debug("eod movers failed: %s", e)
        return []


def _gameplan(flows: Dict[str, Any], events: Dict[str, Any],
              india_facts: Dict[str, Any], movers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tomorrow's gameplan — the desk checklist a trader reads at 19:00 and
    again at 08:45. Deterministic bullets from data already computed this
    request (+ the cached Market Pulse), so it costs zero tokens and can never
    hallucinate. Facts only — the trading decision stays with the user."""
    bullets: List[str] = []
    try:
        from ..scanners.market_pulse import market_pulse
        pulse = market_pulse()
    except Exception:
        pulse = {}

    regime = (india_facts or {}).get("regime")
    if isinstance(regime, dict):  # _india_facts carries {market, vix}
        regime = regime.get("market")
    if regime:
        bullets.append(f"Regime: {str(regime).capitalize()} — position sizing follows the regime gate.")

    b = (pulse or {}).get("breadth") or {}
    if b.get("score") is not None:
        diff = next((d for d in (pulse.get("diff") or []) if d.get("metric") == "breadth_score"), None)
        trend = f" ({diff['detail']})" if diff else ""
        bullets.append(f"Breadth {b['score']}/100 ({b.get('band')}){trend}; "
                       f"{b.get('new_highs', 0)} new 52w highs vs {b.get('new_lows', 0)} lows.")

    fl = (pulse or {}).get("flows") or {}
    fii, dii = fl.get("fii"), fl.get("dii")
    if fii and dii:
        bullets.append(
            f"Flows: FII {fii['side']} {fii['days']} straight session{'s' if fii['days'] != 1 else ''} "
            f"({fii['cum_cr']:+,.0f} Cr) vs DII {dii['side']} {dii['days']} ({dii['cum_cr']:+,.0f} Cr) — a flip in either streak is the tell."
        )

    vol = (pulse or {}).get("vol") or {}
    if vol.get("vix") is not None and vol.get("read"):
        bullets.append(f"Vol: India VIX {vol['vix']} vs NIFTY HV20 {(vol.get('hv') or {}).get('20', '—')} — {vol['read']}.")

    expiry = (events or {}).get("expiry") or {}
    exp_bits = [f"{k} expiry {v}" for k, v in (("weekly", expiry.get("weekly")), ("monthly", expiry.get("monthly"))) if v]
    ev_items = (events or {}).get("items") or []
    if exp_bits or ev_items:
        names = ", ".join(str(e.get("symbol") or e.get("name") or "") for e in ev_items[:4]).strip(", ")
        bullets.append("Calendar: " + " · ".join(exp_bits + ([f"earnings: {names}"] if names else [])) + ".")

    if movers:
        gain = [m for m in movers if (m.get("change_pct") or 0) > 0][:3]
        lose = [m for m in movers if (m.get("change_pct") or 0) < 0][:3]
        fmt = lambda ms: " · ".join(f"{m['symbol']} {m['change_pct']:+.1f}%" for m in ms)  # noqa: E731
        watch = " / ".join(x for x in (fmt(gain), fmt(lose)) if x)
        if watch:
            bullets.append(f"On watch from today's tape (EOD): {watch} — check for follow-through vs fade at open.")

    bullets.append("Pre-open (9:00–9:08): confirm the gap against overnight global cues before acting on any plan.")
    return {"bullets": bullets, "note": "Derived + EOD-published facts · not investment advice"}


def fii_dii_eod(sessions: int = 5) -> Dict[str, Any]:
    """EOD FII/DII daily net (cash) + a short trailing trend, from the locked
    `ml.data.fii_dii_history` store. Values in ₹ Cr. F&O net is not carried by
    that store → ``fno_net`` is None. Honest-empty when no data is available.

    Public EOD statistics — labelled `provisional`. NOT an intraday feed.
    """
    empty: Dict[str, Any] = {
        "date": None,
        "provisional": True,
        "fii": {"cash_net": None, "fno_net": None},
        "dii": {"cash_net": None},
        "trend": [],
        "source": "NSE (EOD provisional)",
    }
    try:
        from ml.data.fii_dii_history import fii_dii_series
        end = date.today()
        start = end - timedelta(days=25)
        df = fii_dii_series(start.isoformat(), end.isoformat())
    except Exception as e:  # noqa: BLE001
        logger.debug("fii_dii_eod read failed: %s", e)
        return empty
    if df is None or getattr(df, "empty", True):
        return empty
    try:
        df = df.sort_index()
        tail = df.tail(max(1, sessions))
        trend = [
            {
                "date": idx.date().isoformat(),
                "fii_cash": round(float(row.get("fii_net", 0.0) or 0.0), 1),
                "dii_cash": round(float(row.get("dii_net", 0.0) or 0.0), 1),
            }
            for idx, row in tail.iterrows()
        ]
        last_idx = df.index[-1]
        last = df.iloc[-1]
        return {
            "date": last_idx.date().isoformat(),
            "provisional": True,
            "fii": {"cash_net": round(float(last.get("fii_net", 0.0) or 0.0), 1), "fno_net": None},
            "dii": {"cash_net": round(float(last.get("dii_net", 0.0) or 0.0), 1)},
            "trend": trend,
            "source": "NSE (EOD provisional)",
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("fii_dii_eod shaping failed: %s", e)
        return empty


# ── global cues (reuse the /api/market/global provider) ───────────────────
def _global() -> Dict[str, Any]:
    """Global cues + a GIFT-Nifty gap read. Reuses market_routes._fetch_global_cues
    (foreign / non-NSE feeds). Lazy import avoids an import cycle."""
    items: List[Dict[str, Any]] = []
    source = "yfinance"
    try:
        from ...api.market_routes import _fetch_global_cues
        payload = _fetch_global_cues() or {}
        items = payload.get("items") or []
        source = payload.get("source") or "yfinance"
    except Exception as e:  # noqa: BLE001
        logger.debug("briefing global fetch failed: %s", e)

    gift = next((it for it in items if it.get("key") == "giftnifty"), None)
    gap_read = None
    if gift and gift.get("change_pct") is not None:
        c = gift["change_pct"]
        tone = "a soft open" if c <= -0.3 else "a firm open" if c >= 0.3 else "a flat open"
        gap_read = f"GIFT Nifty {'+' if c >= 0 else ''}{c}% points to {tone}."
    return {"items": items, "gift_nifty": gift, "gap_read": gap_read, "source": source}


# ── India context (derived / EOD only) ────────────────────────────────────
def _india_facts() -> Dict[str, Any]:
    """Reuse the market_explainer fact assembler: nifty %, breadth, sectors,
    regime + VIX. All derived / EOD — no live intraday quote."""
    try:
        from ..explain.market_explainer import _assemble_facts
        return _assemble_facts() or {}
    except Exception as e:  # noqa: BLE001
        logger.debug("briefing india facts failed: %s", e)
        return {}


def _india_eod_levels() -> Dict[str, Any]:
    """NIFTY 50 + Bank Nifty PREVIOUS close — the settled prior-session close, a
    published EOD statistic (SEBI-safe, shown on every finance site). This is
    NEVER the live intraday tick: yfinance fast_info.previous_close is always the
    last *settled* session's close, so it does not leak a real-time NSE quote."""
    out: Dict[str, Any] = {}
    try:
        import yfinance as yf  # noqa: WPS433
        for key, sym, label in (("nifty", "^NSEI", "NIFTY 50"), ("banknifty", "^NSEBANK", "Bank Nifty")):
            try:
                prev = getattr(yf.Ticker(sym).fast_info, "previous_close", None)
                if prev:
                    out[key] = {"label": label, "prev_close": round(float(prev), 2)}
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        logger.debug("india eod levels failed: %s", e)
    return out


def _india_section(facts: Dict[str, Any]) -> Dict[str, Any]:
    regime = facts.get("regime") or {}
    breadth = facts.get("breadth") or {}
    sectors = facts.get("sectors") or {}
    return {
        "regime": regime.get("market"),
        "vix": regime.get("vix"),
        # NIFTY 50 + Bank Nifty prior-session close (EOD, safe) — the key Indian
        # reference levels alongside the global setup.
        "eod": _india_eod_levels(),
        "breadth": {
            "adv": breadth.get("adv"),
            "dec": breadth.get("dec"),
            "adv_pct": breadth.get("adv_pct"),
        } if breadth else None,
        "sectors": {
            "leading": sectors.get("leading") or [],
            "lagging": sectors.get("lagging") or [],
        } if sectors else None,
        "note": "Derived · prev close (EOD)",
    }


# ── headlines (deterministic, 0 tokens) ───────────────────────────────────
def _premarket_headline(g: Dict[str, Any], flows: Dict[str, Any], india: Dict[str, Any]) -> str:
    gift = g.get("gift_nifty") or {}
    items = {it.get("key"): it for it in (g.get("items") or [])}
    lead = gift if gift.get("change_pct") is not None else items.get("sp500", {})
    c = lead.get("change_pct")
    bias = "Cautious open" if (c is not None and c <= -0.3) else \
           "Firm open" if (c is not None and c >= 0.3) else "Flat open"
    parts: List[str] = []
    if gift.get("change_pct") is not None:
        parts.append(f"GIFT Nifty {'+' if gift['change_pct'] >= 0 else ''}{gift['change_pct']}%")
    elif items.get("sp500", {}).get("change_pct") is not None:
        s = items["sp500"]["change_pct"]
        parts.append(f"S&P 500 {'+' if s >= 0 else ''}{s}%")
    asia = [items[k]["change_pct"] for k in ("nikkei", "hangseng")
            if items.get(k, {}).get("change_pct") is not None]
    if asia:
        avg = sum(asia) / len(asia)
        parts.append("firm Asia" if avg >= 0.2 else "weak Asia" if avg <= -0.2 else "mixed Asia")
    fii = (flows.get("fii") or {}).get("cash_net")
    if fii is not None:
        parts.append("FIIs net buyers" if fii > 0 else "FIIs net sellers" if fii < 0 else "FIIs flat")
    tail = " · ".join(parts)
    return f"{bias}{(' — ' + tail) if tail else ''}"


def _postmarket_headline(facts: Dict[str, Any], flows: Dict[str, Any]) -> str:
    n = facts.get("nifty") or {}
    b = facts.get("breadth") or {}
    parts: List[str] = []
    if n.get("change_pct") is not None:
        c = float(n["change_pct"])
        seg = f"NIFTY {'+' if c >= 0 else '−'}{abs(c):g}%"
        if n.get("ltp") is not None:
            seg += f" to {round(float(n['ltp'])):,}"
        parts.append(seg)
    if b.get("adv") is not None and b.get("dec") is not None:
        parts.append("breadth positive" if b["adv"] >= b["dec"] else "breadth negative")
    fii = (flows.get("fii") or {}).get("cash_net")
    if fii is not None:
        parts.append("FIIs net buyers" if fii > 0 else "FIIs net sellers" if fii < 0 else "FIIs flat")
    return " · ".join(parts) if parts else "Market wrap"


def _premarket_drivers(g, flows, india, events) -> List[str]:
    out: List[str] = []
    if g.get("gap_read"):
        out.append(g["gap_read"])
    fii = (flows.get("fii") or {}).get("cash_net")
    dii = (flows.get("dii") or {}).get("cash_net")
    if fii is not None or dii is not None:
        fs = f"FIIs {'+' if (fii or 0) >= 0 else ''}{fii}" if fii is not None else "FIIs n/a"
        ds = f"DIIs {'+' if (dii or 0) >= 0 else ''}{dii}" if dii is not None else "DIIs n/a"
        out.append(f"Prior-session flows (₹ Cr): {fs}, {ds}.")
    if india.get("regime"):
        tail = f" (VIX {india['vix']})" if india.get("vix") is not None else ""
        out.append(f"Regime: {str(india['regime']).capitalize()}{tail}.")
    sc = india.get("sectors") or {}
    if sc.get("leading"):
        out.append(f"Sector leadership: {', '.join(sc['leading'])}.")
    exp = (events or {}).get("expiry") or {}
    if exp.get("weekly"):
        out.append(f"Weekly index expiry: {exp['weekly']}.")
    return out


def _postmarket_drivers(facts, flows, sectors, events) -> List[str]:
    out: List[str] = []
    n = facts.get("nifty") or {}
    if n.get("change_pct") is not None:
        seg = f"NIFTY {'+' if n['change_pct'] >= 0 else ''}{n['change_pct']}%"
        if n.get("ltp") is not None:
            seg += f" at {round(float(n['ltp']))}"
        out.append(seg + ".")
    b = facts.get("breadth") or {}
    if b.get("adv") is not None and b.get("dec") is not None:
        pct = f" ({b['adv_pct']}% advancing)" if b.get("adv_pct") is not None else ""
        out.append(f"Breadth {b['adv']} adv / {b['dec']} dec{pct}.")
    if sectors and sectors.get("leading"):
        out.append(f"Leaders: {', '.join(sectors['leading'])}.")
    if sectors and sectors.get("lagging"):
        out.append(f"Laggards: {', '.join(sectors['lagging'])}.")
    fii = (flows.get("fii") or {}).get("cash_net")
    dii = (flows.get("dii") or {}).get("cash_net")
    if fii is not None or dii is not None:
        out.append(f"Provisional flows (₹ Cr): FIIs {fii}, DIIs {dii}.")
    return out


# ── LLM narrative (cached daily, budget-guarded) ──────────────────────────
def _narrative(session: str, trading_date: str, facts: Dict[str, Any]) -> Optional[str]:
    """Cached grounded narrative — first visitor triggers, everyone shares.
    Returns None on failure / when the LLM is disabled (callers fall back to
    the deterministic drivers)."""
    if session == "postmarket":
        question = (
            f"Write the post-market desk wrap for {trading_date}. Summarise how "
            "the session went (index close, breadth, sector leadership, "
            "institutional flows) and give a neutral read on the next session's "
            "bias and what to watch."
        )
    else:
        question = (
            f"Write the pre-market desk note for {trading_date}. Cover the "
            "likely open bias from global cues and the GIFT-Nifty gap, the key "
            "index zones to watch, the sectors or themes in focus, and the main "
            "risks into the session."
        )
    try:
        from ...ai.agents.grounded import grounded_reason
        return grounded_reason(
            facts,
            question,
            # v2: deep-reasoning tier (role market_brief → LLM_DEEP_MODEL) —
            # one shared call per session per day, so the strongest model
            # costs effectively nothing here. Key bumped so the upgraded
            # prompt takes effect immediately, not tomorrow.
            cache_key=f"briefing:v2:{session}:{trading_date}",
            role="market_brief",
            system=_BRIEFING_SYSTEM,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("briefing narrative failed: %s", e)
        return None


def _earnings_items(days: int = 7) -> List[Dict[str, Any]]:
    """Upcoming earnings in the forward window → briefing event items. SAME
    source as GET /api/earnings/upcoming (the cached ``earnings_predictions``
    table, DB-first). Public calendar dates → safe to surface to everyone."""
    out: List[Dict[str, Any]] = []
    try:
        from ...ai.earnings import fetch_upcoming_earnings
        for r in (fetch_upcoming_earnings(days=days) or [])[:8]:
            sym = getattr(r, "symbol", None)
            dt = getattr(r, "announce_date", None)
            if sym and dt:
                out.append({"type": "Earnings", "label": str(sym), "date": str(dt)})
    except Exception as e:  # noqa: BLE001
        logger.debug("briefing earnings items failed: %s", e)
    return out


# ── the centrepiece ───────────────────────────────────────────────────────
def build_briefing(session: str, *, use_llm: bool = True) -> Dict[str, Any]:
    """Build a structured daily briefing for ``session`` (premarket / intraday /
    postmarket). Every section is always present (honest-empty when a factor is
    missing). ``session`` labels the response; 'intraday' renders the
    pre-market read (the day's plan still stands during the session).
    """
    sess = session if session in ("premarket", "intraday", "postmarket") else "premarket"
    is_post = sess == "postmarket"
    trading_date = _trading_date_for(sess)
    generated_at = datetime.now(IST).isoformat()

    g = _global()
    flows = fii_dii_eod(5)
    india_facts = _india_facts()
    events = {"items": _earnings_items(7), "expiry": _expiry()}

    if is_post:
        india = _india_section(india_facts)
        sectors_sec = india.get("sectors")
        # SEBI guard: the NIFTY level in ``india_facts`` comes from a LIVE
        # provider quote (``get_quote``). It is only an honest "EOD close" — and
        # only safe to surface to everyone — when the market is actually CLOSED
        # at generation time. If a postmarket briefing is generated while the
        # market is open (e.g. an explicit ?session=postmarket request at 11:00
        # IST), suppress the index level so we never leak a live intraday NSE
        # quote to non-entitled users or mislabel a live value as EOD. Breadth /
        # VIX / sectors are derived-or-EOD (safe) and stay either way.
        market_closed = current_session() == "postmarket"
        nifty_eod = india_facts.get("nifty") if market_closed else None
        # Facts copy with the live-quote-derived NIFTY level gated out when the
        # market is open, so the headline / drivers / narrative never cite it.
        post_facts = dict(india_facts)
        post_facts["nifty"] = nifty_eod
        tape = {
            "nifty": {
                "ltp": (nifty_eod or {}).get("ltp"),
                "change_pct": (nifty_eod or {}).get("change_pct"),
            } if nifty_eod else None,
            "vix": india.get("vix"),
            "breadth": india.get("breadth"),
            "note": _EOD_LABEL,
        }
        llm_facts = {
            "session": sess, "date": trading_date,
            "nifty": nifty_eod, "breadth": india_facts.get("breadth"),
            "sectors": india_facts.get("sectors"), "regime": india_facts.get("regime"),
            "flows": {"fii_cash_net": (flows.get("fii") or {}).get("cash_net"),
                      "dii_cash_net": (flows.get("dii") or {}).get("cash_net"),
                      "trend": flows.get("trend")},
            "expiry": events.get("expiry"),
        }
        narrative = _narrative(sess, trading_date, llm_facts) if use_llm else None
        drivers = _postmarket_drivers(post_facts, flows, sectors_sec, events)
        return {
            "session": sess,
            "generated_at": generated_at,
            "trading_date": trading_date,
            "headline": _postmarket_headline(post_facts, flows),
            "tape": tape,
            "sectors": sectors_sec,
            "flows": flows,
            "movers": (movers_block := {"items": _eod_movers(), "note": "EOD · settled close"}),
            "tomorrow": events,
            "gameplan": _gameplan(flows, events, india_facts, movers_block["items"]),
            "wrap": {"narrative": narrative, "drivers": drivers, "disclaimer": _DISCLAIMER},
        }

    # premarket / intraday
    india = _india_section(india_facts)
    llm_facts = {
        "session": sess, "date": trading_date,
        "global": g.get("items"), "gift_nifty": g.get("gift_nifty"),
        "flows": {"fii_cash_net": (flows.get("fii") or {}).get("cash_net"),
                  "dii_cash_net": (flows.get("dii") or {}).get("cash_net"),
                  "trend": flows.get("trend")},
        "india": {"regime": india.get("regime"), "vix": india.get("vix"),
                  "breadth": india.get("breadth"), "sectors": india.get("sectors")},
        "expiry": events.get("expiry"),
    }
    narrative = _narrative(sess, trading_date, llm_facts) if use_llm else None
    drivers = _premarket_drivers(g, flows, india, events)
    return {
        "session": sess,
        "generated_at": generated_at,
        "trading_date": trading_date,
        "headline": _premarket_headline(g, flows, india),
        "global": g,
        "flows": flows,
        "india": india,
        "events": events,
        "plan": {"narrative": narrative, "drivers": drivers, "disclaimer": _DISCLAIMER},
    }


__all__ = ["build_briefing", "current_session", "fii_dii_eod"]

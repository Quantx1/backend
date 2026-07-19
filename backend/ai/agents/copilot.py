"""
AI Copilot graph (N1) — context-aware chat embedded on every platform page.

Graph shape (linear):

    Classifier → ToolPlanner → ToolCaller → Responder

- **Classifier** rejects out-of-scope prompts early (EarlyExit).
- **ToolPlanner** asks the LLM "which tool(s) does this user request need?"
  and emits a tiny JSON plan. No tools need calling is a valid plan.
- **ToolCaller** executes the plan against ``tool_registry`` and records
  each tool's output in scratch for the Responder.
- **Responder** synthesizes the final reply using current context +
  route + user + tool results.

Senior-analyst voice per Step 4 §1: numbers first, no fluff, cite the
tool data.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from ...core.config import settings
from .base import Agent, EarlyExit, GraphRunner
from .llm import llm_for, LLM
from .state import AgentState
from .tools import tool_registry

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- classifier

# Fast-path: if the message contains any of these, skip the classifier LLM
# call (~4.6s) and assume in-scope. Catches the >95% of real user queries
# that are obviously about markets without paying an LLM round-trip.
_FINANCE_KEYWORDS = re.compile(
    r"\b("
    r"stock|share|price|market|nifty|banknifty|sensex|finnifty|midcpnifty|"
    r"trade|trading|buy|sell|long|short|hold|exit|entry|target|stoploss|sl|"
    r"rsi|macd|sma|ema|vwap|bollinger|adx|atr|fibonacci|fib|signal|alpha|"
    r"portfolio|position|holdings|watchlist|strategy|backtest|breakout|"
    r"support|resistance|candlestick|candle|chart|pattern|profit|loss|pnl|"
    r"p&l|sip|dividend|earnings|broker|demat|kite|zerodha|upstox|angel|"
    r"bse|nse|futures|options|fo|f&o|call|put|premium|strike|iv|theta|gamma|"
    r"delta|hedge|swing|intraday|gap|volume|ohlc|ipo|smallcase|mutual fund|mf|"
    r"sebi|rbi|fed|interest rate|inflation|gdp|cpi|wpi|earning|quarterly|"
    r"q1|q2|q3|q4|fy|return|sharpe|drawdown|risk|reward|stoplos|"
    r"reliance|tcs|infy|hdfc|hdfcbank|icici|icicibank|sbi|sbin|kotak|axis|"
    r"itc|wipro|maruti|tata|adani|asian|hindustan|bharti|airtel|jio|larsen|"
    r"bajaj|titan|nestle|britannia|sun|cipla|dr ?reddy|divis|apollo|ultra|"
    r"ambuja|jsw|tata steel|hindalco|coal india|ongc|ioc|bpcl|gail|powergrid|"
    r"\₹|rs\.|inr|crore|lakh|cr |"
    r"copilot|engine|regime|sentiment|news"
    r")\b",
    re.IGNORECASE,
)
# @SYMBOL references in user message — always in-scope.
_AT_SYMBOL = re.compile(r"@[A-Z][A-Z0-9_&-]{1,15}")
# Short greetings — no need to classify or run tools.
_GREETING = re.compile(
    r"^\s*(hi|hello|hey|yo|hola|good (morning|afternoon|evening)|namaste|thanks?|thank you|ok|okay|sup|wassup)[!?.\s]*$",
    re.IGNORECASE)
# When these terms appear the responder will likely need live data; force the
# planner LLM hop so it can call portfolio/positions/signals tools.
_NEEDS_TOOLS = re.compile(
    r"\b(my (portfolio|positions?|holdings?|trades?|watchlist)|"
    r"current (regime|signals?|prices?|positions?)|"
    r"today'?s (signals?|movers?|news|trades?)|"
    r"latest (signals?|news)|"
    r"recent (trades?|signals?)|"
    r"open positions?|exit (now|today)|"
    r"top (gainers?|losers?|movers?)|"
    # live-market formulations MUST hit real tools — a zero-tool reply here
    # makes the responder fabricate levels/citations (seen live 2026-06-12)
    r"market (status|breadth|today|now|regime|doing|outlook|sentiment|trend|view)|"
    r"bull(ish)?|bear(ish)?|overbought|oversold|"
    r"regime|vix|"
    r"how('s| is) the market|"
    r"nifty (level|support|resistance|now|today))\b",
    re.IGNORECASE,
)

# Classifier intents that inherently need live data — these always reach the
# tool planner (never the pure-knowledge skip), so a stock/market/portfolio/
# signal question fetches real data instead of answering blind. The planner
# itself returns an empty plan for any that turn out to be definitional.
_DATA_INTENTS = frozenset({
    "stock_research", "market_context", "regime_ask",
    "portfolio_review", "signal_explain",
})


# ----------------------------------------------------------- bounded re-plan
#
# After the first tool pass, the copilot may discover its plan was wrong (a
# tool errored, nothing was planned for a tool-needing query, or the planner
# explicitly flagged a follow-up). ``_replan_loop`` re-runs planner→caller up
# to ``_MAX_REPLAN_ROUNDS`` times — but only while the monthly LLM budget has
# headroom. Re-plans accumulate tool_results (the caller appends, never
# clobbers) so the responder sees the full picture.

_MAX_REPLAN_ROUNDS = 2


def _replan_round(state) -> int:
    return int(state.get("replan", "round", 0) or 0)


def _bump_replan_round(state) -> None:
    state.put("replan", round=_replan_round(state) + 1)


def needs_more(state) -> bool:
    plan = state.get("tool_planner", "plan") or []
    plan_source = state.get("tool_planner", "plan_source", "")
    if not plan or plan_source in ("regex_greeting", "regex_pure_knowledge"):
        return False
    results = state.get("tool_caller", "tool_results") or []
    if not results:
        return True
    if any(isinstance(r.get("result"), dict) and r["result"].get("error") for r in results):
        return True
    if state.get("tool_planner", "follow_up", False):
        return True
    return False


def _replan_affordable() -> bool:
    try:
        from ...observability.llm_budget import get_meter
        return not get_meter().over_budget(settings.LLM_MONTHLY_BUDGET_USD)
    except Exception:  # noqa: BLE001
        return True


async def _replan_loop(state, planner, caller) -> None:
    while _replan_round(state) < _MAX_REPLAN_ROUNDS and needs_more(state):
        if not _replan_affordable():
            break
        _bump_replan_round(state)
        try:
            await planner.run(state)
            await caller.run(state)
        except EarlyExit:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("copilot re-plan round %d failed: %s", _replan_round(state), exc)
            break


def _quick_classify(message: str) -> Dict[str, Any] | None:
    """Returns {intent, in_scope} if we can decide without the LLM, else None."""
    if not message:
        return None
    if _GREETING.match(message):
        return {"intent": "greeting", "in_scope": True, "via": "regex"}
    if _AT_SYMBOL.search(message):
        return {"intent": "stock_research", "in_scope": True, "via": "regex"}
    if _FINANCE_KEYWORDS.search(message):
        return {"intent": "market_context", "in_scope": True, "via": "regex"}
    return None


class CopilotClassifier(Agent):
    name = "classifier"

    async def _run(self, state: AgentState) -> None:
        message = state.inputs.get("message", "")
        if not message:
            state.output = {"reply": "Please send a message.", "refused": True}
            raise EarlyExit

        # ── Fast path: deterministic in-scope detection ──
        quick = _quick_classify(message)
        if quick is not None:
            state.put(self.name, intent=quick["intent"], in_scope=True, raw=quick)
            return

        # ── Slow path: LLM-classified for ambiguous messages ──
        system = (
            "You are a strict classifier for a finance-only trading assistant "
            "serving Indian retail + prosumer traders (NSE / BSE). Return JSON only."
        )
        schema = (
            '{"in_scope": true|false, "intent": "portfolio_review|signal_explain|'
            'market_context|stock_research|regime_ask|greeting|other", '
            '"reason": "short string"}'
        )
        prompt = (
            f"User message: {message}\n"
            "In scope: stocks, F&O, trading, portfolio, market news, macro, "
            "regime, signals, paper-trading, tax-on-trades questions.\n"
            "Out of scope: coding help, personal advice, medical, entertainment."
        )
        parsed = await self.llm.generate_json(prompt, schema, system=system)
        in_scope = bool(parsed.get("in_scope", True))
        intent = str(parsed.get("intent", "other"))

        state.put(self.name, intent=intent, in_scope=in_scope, raw=parsed)

        if not in_scope:
            state.output = {
                "reply": (
                    "I can only help with trading, markets, portfolio, and "
                    "signals. Ask me about a stock, a signal, or your "
                    "positions and I'll dig in."
                ),
                "refused": True,
                "intent": intent,
            }
            raise EarlyExit


# ---------------------------------------------------------------- tool planner


class CopilotToolPlanner(Agent):
    name = "tool_planner"

    async def _run(self, state: AgentState) -> None:
        message = state.inputs.get("message", "")
        intent = state.get("classifier", "intent", "other")
        route = state.inputs.get("route", "")
        mentioned_symbols = state.inputs.get("mentioned_symbols") or []

        # ── Fast path: greetings need no tools ──
        if intent == "greeting":
            state.put(self.name, plan=[], plan_source="regex_greeting")
            return

        # Pure-knowledge skip: only when the question is NOT data-requiring.
        # Data intents must reach the planner even without an @SYMBOL or a
        # _NEEDS_TOOLS keyword — otherwise "analyze RELIANCE" / "is the market
        # bullish" answer blind ("data wasn't fetched", reported 2026-06-22). The
        # planner still returns [] for genuinely definitional questions (e.g.
        # "what is RSI" tagged market_context), so the only cost is one cheap
        # planner call on those; correctness over a saved ~1.3s.
        if (intent not in _DATA_INTENTS
                and not mentioned_symbols
                and not _AT_SYMBOL.search(message)
                and not _NEEDS_TOOLS.search(message)):
            state.put(self.name, plan=[], plan_source="regex_pure_knowledge")
            return

        # ── Slow path: LLM plans the tool calls ──
        schema_json = json.dumps(tool_registry.schema(), indent=2)
        schema_hint = (
            '{"tool_calls": [{"tool": "<name>", "args": {...}}, ...], '
            '"follow_up": false} '
            "— empty array if no tools needed; set follow_up true only when one "
            "more tool round is genuinely required to answer."
        )
        system = (
            "You plan which tools to call to answer a trading question. "
            "Call the fewest tools that cover the question. Prefer one call. "
            "Never call more than 3. Return JSON only."
        )
        # On a re-plan round, show the prior results so the LLM avoids repeating
        # failed calls and picks DIFFERENT tools / corrected args.
        observation = ""
        if _replan_round(state) > 0:
            prior = state.get("tool_caller", "tool_results") or []
            observation = (
                "\n\nThis is a RE-PLAN. The previous tool round returned:\n"
                f"{json.dumps(prior, ensure_ascii=False, default=str)[:1500]}\n"
                "Do NOT repeat calls that errored or returned nothing useful. "
                "Pick DIFFERENT tools or corrected arguments that actually "
                "answer the question. Return an empty array if no further tool "
                "can help."
            )
        prompt = (
            f"Available tools:\n{schema_json}\n\n"
            f"Current route: {route}\n"
            f"Intent: {intent}\n"
            f"Mentioned symbols: {mentioned_symbols}\n"
            f"User message: {message}\n\n"
            "Produce the plan."
            f"{observation}"
        )
        parsed = await self.llm.generate_json(prompt, schema_hint, system=system)
        raw_calls = parsed.get("tool_calls") or []
        plan: List[Dict[str, Any]] = []
        for c in raw_calls[:3]:  # hard cap
            name = c.get("tool")
            args = c.get("args") or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            plan.append({"tool": name, "args": args})
        state.put(
            self.name,
            plan=plan,
            plan_source=("llm_replan" if _replan_round(state) else "llm"),
            follow_up=bool(parsed.get("follow_up", False)),
        )


# ----------------------------------------------------------------- tool caller


class CopilotToolCaller(Agent):
    name = "tool_caller"

    async def _run(self, state: AgentState) -> None:
        plan = state.get("tool_planner", "plan") or []
        user_id = state.user_id or state.inputs.get("user_id")
        results: List[Dict[str, Any]] = []
        for item in plan:
            args = dict(item.get("args") or {})
            # Inject user_id automatically when a tool needs it.
            spec = tool_registry.get(item["tool"])
            if spec and "user_id" in spec.params and user_id and "user_id" not in args:
                args["user_id"] = user_id
            out = await tool_registry.call(state, item["tool"], **args)
            results.append({"tool": item["tool"], "args": args, "result": out})
        prior = state.get(self.name, "tool_results") or []
        state.put(self.name, tool_results=prior + results)


# ------------------------------------------------------- responder prompt + artifacts

# Shared so both the (non-streaming) Responder agent and the streaming runner
# build an identical prompt — one source of truth for the analyst voice.
_RESPONDER_SYSTEM = (
    "You are Quant X Copilot — a senior NSE/BSE analyst on an institutional "
    "quant desk. Write like a professional research note: precise, calm, "
    "evidence-led. The reader is a serious trader who wants a fast, high-signal "
    "read, not a chatbot.\n"
    "\n"
    "VOICE & STRUCTURE (thesis-first, like a top-tier research desk):\n"
    "- OPEN with a one-sentence thesis that directly answers the question — the "
    "verdict/stance up front, with the decisive figure in **bold**. No preamble, "
    "no restating the question.\n"
    "- Then the EVIDENCE in short paragraphs: the dominant driver first, with "
    "exact numbers (level, %, date). Bold the single most important figure per "
    "idea.\n"
    "- Where two sides matter, use SHORT bold lead-in labels on their own lines — "
    "e.g. `**Bull case:** …`, `**Bear case:** …`, `**Risk:** …`, `**Catalyst:** "
    "…`, `**Setup:** …`. These labels replace headings; keep them tight.\n"
    "- Close analytical answers with a `**My read:**` line — your synthesis / "
    "which way the balance tips and why. This is the analyst's take, stated "
    "plainly.\n"
    "- Use a `## ` heading ONLY for a genuinely long multi-part answer; most "
    "answers need no headings at all — bold labels carry the structure.\n"
    "- Tabular data (comparisons, holdings, key stats, levels, option chain, "
    "screener/IPO rows) → a GFM Markdown table with a header row and a "
    "`|---|---|` separator; first column = the label/symbol.\n"
    "\n"
    "NO EMOJI — this is an institutional desk, not a social feed. Do NOT use "
    "emoji or decorative symbols anywhere (no 📊 📈 ✅ ⚠️ 🎯 🟢 🔴, no colored "
    "dots, no ▲/▼ glyphs). All emphasis comes from **bold**, clear labels, and "
    "tables. A single answer must contain zero emoji.\n"
    "\n"
    "PRECISION & DEPTH (be exact, complete, clean):\n"
    "- Be EXACT, never vague: the specific number / level / date, not 'around', "
    "'a few', 'near-term'. Prefer 'support 24,320, target 25,100 (+3.2%), stop "
    "24,050' over 'some upside with a stop below support'.\n"
    "- Be COMPLETE on what matters, nothing more: a trade idea covers setup → "
    "entry / stop / target → risk:reward → the single key risk. A stock read "
    "covers trend → the decisive metric → context (sector / news / event). "
    "Answer the actual question first; skip generic definitions the user didn't "
    "ask for.\n"
    "- Be NEAT: one idea per line, whitespace between blocks, a table for any 3+ "
    "comparable rows. No wall of text, no repetition, no hedging filler ('it "
    "depends', 'markets are uncertain') unless you immediately say on WHAT it "
    "depends. Every sentence adds a fact or an implication — cut the rest. Do "
    "NOT write the literal words 'so what'.\n"
    "\n"
    "GROUNDING (strict):\n"
    "- Write every price/percent as a plain number (24,850 or +3.2%) — no font "
    "names or markup around the number itself.\n"
    "- Only state a specific price/level/ratio if it appears in the tool data. "
    "If NO tool data is provided you must NOT state any current level, price, "
    "VIX, ratio or percent move — answer qualitatively from general knowledge "
    "and say the data wasn't fetched.\n"
    "- NEVER write bracketed tool/citation markup — no [tool:...], [source], "
    "[get_...]. The app shows consulted data separately.\n"
    "\n"
    "Read like a sharp human analyst. Never say 'as an AI'. No fluff preamble. "
    "Keep it tight. Educational, not an execution recommendation; end anything "
    "actionable with a one-line **Risk:** note."
)


def _build_responder_prompt(state: AgentState) -> str:
    message = state.inputs.get("message", "")
    history = state.inputs.get("history") or []
    route = state.inputs.get("route", "")
    memory = (state.inputs.get("memory") or "").strip()
    tool_results = state.get("tool_caller", "tool_results") or []
    memory_block = f"Known about this trader (memory):\n{memory}\n\n" if memory else ""
    return (
        f"{memory_block}"
        f"Route: {route}\n"
        f"Recent conversation (last 6 turns): "
        f"{json.dumps(history[-6:], ensure_ascii=False)}\n\n"
        f"Tool results JSON: {json.dumps(tool_results, ensure_ascii=False, default=str)[:4000]}\n\n"
        f"User message: {message}\n\n"
        "Write the reply."
    )


_FALLBACK_REPLY = (
    "I couldn't put together a reply just now — the assistant may be "
    "temporarily unavailable or the request timed out. Please try again."
)


# ----------------------------------------------------------- structured artifacts

def _num(v: Any) -> float | None:
    """Coerce to float, dropping None / NaN / junk."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def _pct(v: Any) -> float | None:
    """Probability (0..1) or percent (0..100) → a 0..100 number, 1dp."""
    n = _num(v)
    if n is None:
        return None
    return round(n * 100, 1) if abs(n) <= 1 else round(n, 1)


def _urlq(s: Any) -> str:
    """URL-encode a value for an artifact deep-link query param."""
    from urllib.parse import quote
    return quote(str(s or "").strip())


def _fmt_money(v: Any) -> str:
    n = _num(v)
    if n is None:
        return "—"
    a = abs(n)
    if a >= 1e7:
        return f"₹{n / 1e7:.2f}Cr"
    if a >= 1e5:
        return f"₹{n / 1e5:.2f}L"
    if a >= 1e3:
        return f"₹{n / 1e3:.1f}K"
    return f"₹{n:,.0f}"


def _build_payoff(proposal: Dict[str, Any]) -> Dict[str, Any] | None:
    """Compute an option strategy's expiry payoff curve from its legs. Each leg:
    {action BUY/SELL, option_type CE/PE, strike, premium}. Returns points +
    breakevens + spot, or None when legs are unusable. Deterministic."""
    legs = proposal.get("legs") or []
    parsed = []
    for l in legs:
        strike = _num(l.get("strike"))
        prem = _num(l.get("premium"))
        ot = str(l.get("option_type") or "").upper()
        act = str(l.get("action") or l.get("side") or "").upper()
        if strike is None or prem is None or ot not in ("CE", "PE") or act not in ("BUY", "SELL"):
            return None
        parsed.append((act, ot, strike, prem))
    if not parsed:
        return None
    lot = int(_num(proposal.get("lot_size")) or 1)
    strikes = [p[2] for p in parsed]
    lo, hi = min(strikes) * 0.90, max(strikes) * 1.10
    n = 41
    step = (hi - lo) / (n - 1) if hi > lo else 1.0

    def pnl_at(s: float) -> float:
        total = 0.0
        for act, ot, strike, prem in parsed:
            intrinsic = max(s - strike, 0.0) if ot == "CE" else max(strike - s, 0.0)
            leg = (intrinsic - prem) if act == "BUY" else (prem - intrinsic)
            total += leg
        return round(total * lot, 0)

    points = [{"x": round(lo + i * step, 1), "y": pnl_at(lo + i * step)} for i in range(n)]
    spot = None
    # Spot ≈ the middle strike (ATM anchor) when not otherwise given.
    spot = round(sorted(strikes)[len(strikes) // 2], 1)
    return {
        "points": points,
        "breakevens": [round(b, 1) for b in (proposal.get("breakevens") or []) if _num(b) is not None],
        "spot": spot,
        "maxProfit": _num(proposal.get("max_profit")),
        "maxLoss": _num(proposal.get("max_loss")),
    }


def build_artifacts(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn real tool outputs into renderable chart/stat artifacts.

    Every field is sourced from live tool data — no synthetic values. Caps at
    3 artifacts so the chat stays readable. Frontend renders by ``type``:
    ``sparkline`` (price), ``bars`` (regime probs), ``stat`` (snapshot pills).
    """
    arts: List[Dict[str, Any]] = []
    for tr in tool_results:
        if len(arts) >= 3:
            break
        tool = tr.get("tool")
        res = tr.get("result")
        if not isinstance(res, dict) or res.get("error"):
            continue

        if tool == "get_stock_snapshot":
            series = [x for x in (_num(v) for v in (res.get("series") or [])) if x is not None]
            if len(series) >= 2:
                # Real interactive area chart (axes + tooltip) — richer than the
                # old sparkline for the primary price artifact.
                arts.append({
                    "type": "linechart",
                    "title": res.get("symbol") or "Price",
                    "subtitle": "Last 3 months · daily close",
                    "series": series,
                    "last": _num(res.get("last_close")),
                    "changePct": _num(res.get("pct_change_3m")),
                    "yLabel": "₹",
                })

        elif tool == "get_current_regime":
            probs = {
                "bull": (_pct(res.get("prob_bull")), "up"),
                "sideways": (_pct(res.get("prob_sideways")), "neutral"),
                "bear": (_pct(res.get("prob_bear")), "down"),
            }
            regime = str(res.get("regime") or "").lower()
            # Gauge of the dominant regime's confidence (state-of-the-art visual).
            dom = regime if regime in probs and probs[regime][0] is not None else \
                max((k for k in probs if probs[k][0] is not None), key=lambda k: probs[k][0], default=None)
            if dom is not None:
                conf, tone = probs[dom]
                caption_bits = []
                nifty = _num(res.get("nifty_close"))
                vix = _num(res.get("vix"))
                if nifty is not None:
                    caption_bits.append(f"NIFTY {nifty:,.0f}")
                if vix is not None:
                    caption_bits.append(f"VIX {vix:.1f}")
                arts.append({
                    "type": "gauge",
                    "title": "Market regime",
                    "subtitle": " · ".join(caption_bits) or None,
                    "value": conf,
                    "valueLabel": f"{dom.title()} · {conf:.0f}% confidence",
                    "tone": tone,
                })

        elif tool == "get_todays_signals":
            sigs = res.get("signals") or []
            if sigs:
                longs = sum(1 for s in sigs if str(s.get("direction", "")).lower() in ("long", "buy"))
                top = sigs[0]
                arts.append({
                    "type": "stat",
                    "title": "Today's signals",
                    "stats": [
                        {"label": "Active", "value": str(len(sigs))},
                        {"label": "Long", "value": str(longs), "tone": "up"},
                        {"label": "Short", "value": str(len(sigs) - longs), "tone": "down"},
                        {"label": "Top", "value": str(top.get("symbol", "—"))},
                    ],
                })

        elif tool == "suggest_options_strategy":
            proposal = res.get("proposal") or {}
            payoff = _build_payoff(proposal)
            if payoff:
                # State-of-the-art: the actual expiry payoff diagram.
                arts.append({
                    "type": "payoff",
                    "title": res.get("template") or "Options structure",
                    "subtitle": (f"{res.get('lots_suggestion')} lots" if res.get("lots_suggestion") else None),
                    **payoff,
                })
            else:
                stats = []
                for label, key, tone in (
                    ("Net premium", "net_premium", None),
                    ("Max profit", "max_profit", "up"),
                    ("Max loss", "max_loss", "down"),
                ):
                    if proposal.get(key) is not None:
                        stats.append({"label": label, "value": _fmt_money(proposal.get(key)), "tone": tone})
                if stats:
                    lots = res.get("lots_suggestion")
                    arts.append({
                        "type": "stat",
                        "title": res.get("template") or "Options structure",
                        "subtitle": f"{lots} lots" if lots else None,
                        "stats": stats,
                    })

        elif tool == "get_portfolio":
            live = res.get("live_positions") or []
            paper = res.get("paper_positions") or []
            n = len(live) + len(paper)
            if n:
                pnl = sum((_num(p.get("pnl")) or 0.0) for p in live)
                arts.append({
                    "type": "stat",
                    "title": "Your book",
                    "stats": [
                        {"label": "Open", "value": str(n)},
                        {"label": "Live", "value": str(len(live))},
                        {"label": "Paper", "value": str(len(paper))},
                        {"label": "Live P&L", "value": _fmt_money(pnl), "tone": "up" if pnl >= 0 else "down"},
                    ],
                })

        # ── Phase 2 artifacts (2026-07-11) ──

        elif tool == "run_screen":
            rows = res.get("results") or []
            if rows and res.get("recognized"):
                table_rows = []
                for r in rows[:12]:
                    chg = _num(r.get("change_pct"))
                    rsi = _num(r.get("rsi"))
                    table_rows.append({
                        "symbol": r.get("symbol"),
                        "cells": [
                            {"value": f"₹{_num(r.get('last_price')):,.1f}" if _num(r.get("last_price")) is not None else "—"},
                            {"value": (f"{chg:+.1f}%" if chg is not None else "—"),
                             "tone": ("up" if (chg or 0) >= 0 else "down")},
                            # RSI==0 means "not computed" (real RSI never hits 0) → show —
                            {"value": (f"{rsi:.0f}" if rsi else "—")},
                        ],
                    })
                names = ", ".join(s.get("name", "") for s in (res.get("scanners_used") or [])[:3])
                arts.append({
                    "type": "table",
                    "title": f"{res.get('count', len(rows))} matches",
                    "subtitle": names or None,
                    "columns": ["LTP", "Chg", "RSI"],
                    "rows": table_rows,
                    "cta": {"label": "Open full screener", "href": f"/scanner/new?q={_urlq(res.get('query'))}"},
                })

        elif tool == "get_fno_snapshot":
            pcr = _num(res.get("pcr_oi"))
            mp = _num(res.get("max_pain"))
            spot = _num(res.get("spot"))
            stats = []
            if pcr is not None:
                stats.append({"label": "PCR (OI)", "value": f"{pcr:.2f}",
                              "tone": "up" if pcr >= 1 else "down"})
            if mp is not None:
                stats.append({"label": "Max pain", "value": f"{mp:,.0f}"})
            if spot is not None:
                stats.append({"label": "Spot", "value": f"{spot:,.0f}"})
            top_call = (res.get("top_call_oi_strikes") or [])[:1]
            top_put = (res.get("top_put_oi_strikes") or [])[:1]
            if top_call:
                stats.append({"label": "Resistance", "value": f"{_num(top_call[0]):,.0f}" if _num(top_call[0]) is not None else "—", "tone": "down"})
            if top_put:
                stats.append({"label": "Support", "value": f"{_num(top_put[0]):,.0f}" if _num(top_put[0]) is not None else "—", "tone": "up"})
            if stats:
                arts.append({
                    "type": "stat",
                    "title": f"{res.get('symbol', 'Chain')} option chain",
                    "subtitle": res.get("pcr_tag") or None,
                    "stats": stats[:4],
                })

        elif tool == "get_fundamentals":
            f = res.get("fundamentals") or {}
            stats = []
            for label, key in (("PE", "pe"), ("ROE", "roe"), ("ROCE", "roce")):
                v = _num(f.get(key))
                if v is not None:
                    suffix = "" if label == "PE" else "%"
                    stats.append({"label": label, "value": f"{v:.1f}{suffix}"})
            mcap = _num(f.get("market_cap_cr"))  # already in ₹ crore
            if mcap is not None:
                mcap_str = (f"₹{mcap / 1e5:.2f}L Cr" if mcap >= 1e5
                            else f"₹{mcap / 1e3:.1f}K Cr" if mcap >= 1e3
                            else f"₹{mcap:,.0f} Cr")
                stats.append({"label": "M-cap", "value": mcap_str})
            if stats:
                arts.append({
                    "type": "stat",
                    "title": f"{res.get('symbol', 'Stock')} fundamentals",
                    "subtitle": res.get("as_of") or None,
                    "stats": stats[:4],
                })

        elif tool == "get_sector_performance":
            secs = [s for s in (res.get("sectors") or []) if _num(s.get("rs_long")) is not None]
            secs = sorted(secs, key=lambda s: _num(s.get("rs_long")) or 0, reverse=True)
            if secs:
                top = secs[:5]
                lo = min((_num(s.get("rs_long")) or 0) for s in top)
                hi = max((_num(s.get("rs_long")) or 0) for s in top)
                span = (hi - lo) or 1.0
                items = [{
                    "label": str(s.get("sector", ""))[:12],
                    "value": round((( _num(s.get("rs_long")) or 0) - lo) / span * 100, 0),
                    "tone": "up" if (_num(s.get("rs_long")) or 0) >= 0 else "down",
                } for s in top]
                arts.append({
                    "type": "bars",
                    "title": "Sector strength",
                    "subtitle": "20d relative strength",
                    "items": items,
                })

        elif tool == "get_fii_dii_flow":
            if res.get("available"):
                fii = _num(res.get("fii_net"))
                dii = _num(res.get("dii_net"))
                stats = []
                if fii is not None:
                    stats.append({"label": "FII net", "value": _fmt_money(fii * 1e7),
                                  "tone": "up" if fii >= 0 else "down"})
                if dii is not None:
                    stats.append({"label": "DII net", "value": _fmt_money(dii * 1e7),
                                  "tone": "up" if dii >= 0 else "down"})
                if stats:
                    arts.append({
                        "type": "stat",
                        "title": "FII / DII flow",
                        "subtitle": res.get("as_of") or None,
                        "stats": stats,
                    })

        elif tool == "run_fundamental_screen":
            rows = res.get("results") or []
            if rows:
                table_rows = []
                for r in rows[:12]:
                    pe = _num(r.get("pe"))
                    roce = _num(r.get("roce"))
                    q = _num(r.get("quality_score"))
                    table_rows.append({
                        "symbol": r.get("symbol"),
                        "cells": [
                            {"value": f"{pe:.1f}" if pe is not None else "—"},
                            {"value": f"{roce:.0f}%" if roce is not None else "—",
                             "tone": "up" if (roce or 0) >= 15 else "neutral"},
                            {"value": f"{q:.0f}/5" if q is not None else "—",
                             "tone": "up" if (q or 0) >= 4 else "neutral"},
                        ],
                    })
                arts.append({
                    "type": "table",
                    "title": f"{res.get('count', len(rows))} {res.get('name', 'matches')}",
                    "subtitle": "Fundamental screen",
                    "columns": ["PE", "ROCE", "Quality"],
                    "rows": table_rows,
                    "cta": {"label": "Open Screener", "href": "/scanner"},
                })

        elif tool == "compile_strategy":
            if res.get("dsl") and not res.get("needs_clarification"):
                rows = []
                if res.get("entry_rule"):
                    rows.append({"label": "Buy when", "value": str(res["entry_rule"])})
                if res.get("exit_rule"):
                    rows.append({"label": "Sell when", "value": str(res["exit_rule"])})
                risk_bits = []
                if res.get("stop_loss_pct") is not None:
                    risk_bits.append(f"SL {res['stop_loss_pct']}%")
                if res.get("take_profit_pct") is not None:
                    risk_bits.append(f"target {res['take_profit_pct']}%")
                if res.get("trailing_stop_pct") is not None:
                    risk_bits.append(f"trail {res['trailing_stop_pct']}%")
                if res.get("square_off_time"):
                    risk_bits.append(f"square-off {res['square_off_time']}")
                if risk_bits:
                    rows.append({"label": "Risk", "value": " · ".join(risk_bits)})
                tf = res.get("timeframe")
                sym = res.get("symbol") or res.get("universe")
                arts.append({
                    "type": "strategy",
                    "title": res.get("name") or "Strategy",
                    "subtitle": " · ".join(str(x) for x in (tf, sym) if x) or None,
                    "rules": rows,
                    "cta": {"label": "Backtest & deploy in Studio",
                            "href": f"/strategies?prompt={_urlq(res.get('name') or '')}"},
                })

        elif tool == "get_ipo_calendar":
            open_ipos = res.get("open") or []
            upcoming = res.get("upcoming") or []
            listing = open_ipos or upcoming
            if listing:
                is_open = bool(open_ipos)
                table_rows = []
                for x in listing[:10]:
                    sub = _num(x.get("subscription_x"))
                    cells = [
                        {"value": x.get("price_band") or "—"},
                        {"value": x.get("close_date") or x.get("open_date") or "—"},
                    ]
                    if is_open:
                        cells.append({"value": f"{sub:.1f}x" if sub is not None else "—",
                                      "tone": "up" if (sub or 0) >= 1 else "neutral"})
                    table_rows.append({"symbol": x.get("symbol") or x.get("company") or "—",
                                       "cells": cells})
                arts.append({
                    "type": "table",
                    "title": f"{len(open_ipos)} open · {len(upcoming)} upcoming",
                    "subtitle": "IPO calendar",
                    "columns": ["Price band", "Closes"] + (["Sub"] if is_open else []),
                    "rows": table_rows,
                    "cta": {"label": "Open IPO center", "href": "/ipo"},
                })

        elif tool == "get_my_strategies":
            strategies = res.get("strategies") or []
            if strategies:
                arts.append({
                    "type": "stat",
                    "title": "Your strategies",
                    "stats": [
                        {"label": "Total", "value": str(res.get("count", len(strategies)))},
                        {"label": "Live/Paper", "value": str(res.get("live_or_paper", 0)),
                         "tone": "up" if res.get("live_or_paper") else None},
                    ],
                    "cta": {"label": "Manage strategies", "href": "/strategies"},
                })

    return arts[:3]


# --------------------------------------------------- transparent-agent rails
#
# ``build_progress`` + ``build_references`` are the honest-telemetry siblings of
# ``build_artifacts``. They project the raw ``AgentState`` (turns + tool_trace)
# — already captured every run — into two compact, BRAND-SAFE, whitelisted
# shapes the copilot UI renders as a Progress timeline + a References panel.
#
# BRAND FIREWALL (highest-frequency failure): raw tool names, ``tool_trace``
# arg/result rows, and any model/provider slugs must NEVER surface. Every label
# routes through ``_tool_public_label``; every reference pulls ONLY whitelisted
# display fields (symbol / direction / template / regime / vix / nifty_close /
# signal id) — never a raw result row; tool errors collapse to a generic string.

# Public, brand-safe labels — mirrors the frontend ``TOOL_LABEL`` map. Unknown
# tools fall back to a generic label so a raw ``get_*`` identifier never leaks.
_TOOL_PUBLIC_LABEL: Dict[str, str] = {
    "get_portfolio": "Your book",
    "get_watchlist": "Watchlist",
    "get_signal": "Signal detail",
    "get_todays_signals": "Today's signals",
    "get_stock_snapshot": "Price data",
    "explain_move": "Move drivers",
    "get_current_regime": "Market regime",
    "suggest_options_strategy": "Options structure",
    # Phase 2 (2026-07-11)
    "get_fundamentals": "Fundamentals",
    "get_technicals": "Technicals",
    "get_news_sentiment": "News mood",
    "get_fno_snapshot": "Option chain",
    "get_sector_performance": "Sector rotation",
    "get_fii_dii_flow": "FII/DII flow",
    "run_screen": "Screener",
    "run_fundamental_screen": "Fundamentals",
    "compile_strategy": "Strategy Studio",
    "get_ipo_calendar": "IPO calendar",
    "get_my_strategies": "Your strategies",
}

_STAGE_LABEL: Dict[str, str] = {
    "classifier": "Understanding your question",
    "tool_planner": "Planning the data pull",
    "tool_caller": "Fetching live data",
    "responder": "Composing the answer",
}

# Parses the responder's ``[tool:<name>]`` citations so a reference can be marked
# cited-vs-merely-consulted. The responder cites with the raw tool name (it sees
# it in the results JSON), which matches ``ToolCall.name`` — intersection only.
_TOOL_CITE_RE = re.compile(r"\[tool:([a-zA-Z0-9_]+)\]")

# Defensive strip of any bracketed tool/citation markup the model emits despite
# the system prompt (e.g. "[tool:get_stock_snapshot]", "[source]"). Attribution
# lives in the References rail + CONSULTED chips, never inline in the prose.
_CITE_STRIP_RE = re.compile(r"\s*\[(?:tool:[a-zA-Z0-9_]+|source|get_[a-zA-Z0-9_]+)\]")


_SOWHAT_LABEL_RE = re.compile(r"(?im)\*\*\s*so[\s-]?what\??\s*\*\*\s*[:：.\-—]?\s*")

# Emoji / decorative-symbol ranges. Deliberately excludes the arrow block
# (U+2190–U+21FF) so the analyst's "setup → entry" arrows and ₹ survive; the
# desk voice bans emoji, and this is the hard guarantee if the model drifts.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, supplemental, extended-A/B, geometric-ext (🟢🔴🚀…)
    "\U00002600-\U000026FF"   # miscellaneous symbols (☀ ⚠ ⚡ …)
    "\U00002700-\U000027BF"   # dingbats (✅ ✂ ✓ ✗ …)
    "\U0001F1E6-\U0001F1FF"   # regional-indicator flags
    "\U00002B00-\U00002BFF"   # misc symbols & arrows (⬆ ⭐ …) — no plain → here
    "\uFE0F\u200D"          # variation selector + zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    """Hard-remove emoji/decorative glyphs — institutional desk voice, no emoji."""
    if not text:
        return text
    text = _EMOJI_RE.sub("", text)
    # Tidy the whitespace an emoji used to occupy (leading "· "/"  " it sat on).
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)([:*]) +", r"\1 ", text)
    return text


def _strip_citations(text: str) -> str:
    if not text:
        return text
    # Drop the forbidden "**So what**:" lead-in label the model occasionally
    # emits (the prompt bans it) — keep the takeaway sentence that follows.
    text = _SOWHAT_LABEL_RE.sub("", text)
    if "[" in text:
        text = _CITE_STRIP_RE.sub("", text).replace("  ", " ")
    text = _strip_emoji(text)
    return text.strip()


def _tool_public_label(name: Any) -> str:
    return _TOOL_PUBLIC_LABEL.get(str(name), "Market data")


def build_references(tool_results: List[Dict[str, Any]], reply: str = "") -> List[Dict[str, Any]]:
    """Project real tool outputs into a compact, brand-safe list of the market-
    data ENTITIES the agent touched (symbols / signals / regime / positions /
    watchlist / options — F4: no news/URL). WHITELISTS display fields; ships NO
    raw result rows. Caps at 8. When ``reply`` is given, each entity is marked
    ``cited`` if the responder cited its tool via ``[tool:<name>]``.

    Reconciliation: errored / empty tool results are SKIPPED entirely, so a tool
    the responder cited but that returned nothing never renders a phantom ref.
    """
    cited: set[str] | None = None
    if reply:
        cited = {m.group(1) for m in _TOOL_CITE_RE.finditer(reply)}

    refs: List[Dict[str, Any]] = []
    seen: set = set()

    def add(kind: str, label: Any, tool: Any, *, sublabel: Any = None, ref_id: Any = None) -> None:
        if label is None:
            return
        text = str(label).strip()[:24]
        if not text:
            return
        key = (kind, text)
        if key in seen:
            return
        seen.add(key)
        entry: Dict[str, Any] = {"kind": kind, "label": text, "tool": _tool_public_label(tool)}
        if sublabel:
            entry["sublabel"] = str(sublabel).strip()[:32]
        if ref_id:
            entry["id"] = str(ref_id)
        if cited is not None:
            entry["cited"] = str(tool) in cited
        refs.append(entry)

    for tr in tool_results:
        if len(refs) >= 8:
            break
        tool = tr.get("tool")
        res = tr.get("result")
        args = tr.get("args") or {}
        if not isinstance(res, dict) or res.get("error"):
            continue

        if tool in ("get_stock_snapshot", "explain_move"):
            add("symbol", res.get("symbol") or args.get("symbol"), tool)
        elif tool == "get_todays_signals":
            for s in (res.get("signals") or [])[:6]:
                if len(refs) >= 8:
                    break
                direction = str(s.get("direction") or "").upper() or None
                add("signal", s.get("symbol"), tool, sublabel=direction)
        elif tool == "get_signal":
            sig = res.get("signal") or {}
            add("signal", sig.get("symbol") or "Signal", tool,
                sublabel=(str(sig.get("direction") or "").upper() or None),
                ref_id=args.get("signal_id") or sig.get("id"))
        elif tool == "get_current_regime":
            regime = str(res.get("regime") or "").title() or "Regime"
            bits: List[str] = []
            nifty = _num(res.get("nifty_close"))
            vix = _num(res.get("vix"))
            if nifty is not None:
                bits.append(f"NIFTY {nifty:,.0f}")
            if vix is not None:
                bits.append(f"VIX {vix:.1f}")
            add("regime", f"{regime} regime", tool, sublabel=(" · ".join(bits) or None))
        elif tool == "get_portfolio":
            for p in ((res.get("live_positions") or []) + (res.get("paper_positions") or [])):
                if len(refs) >= 8:
                    break
                add("position", p.get("symbol"), tool)
        elif tool == "get_watchlist":
            for w in (res.get("watchlist") or []):
                if len(refs) >= 8:
                    break
                add("watch", w.get("symbol"), tool)
        elif tool == "suggest_options_strategy":
            add("options", res.get("template") or "Options structure", tool,
                sublabel=res.get("symbol") or args.get("symbol"))
        # ── Phase 2 references (2026-07-11) ──
        elif tool in ("get_fundamentals", "get_technicals", "get_news_sentiment"):
            add("symbol", res.get("symbol") or args.get("symbol"), tool)
        elif tool == "get_fno_snapshot":
            add("options", res.get("symbol") or args.get("symbol"), tool,
                sublabel=res.get("pcr_tag"))
        elif tool in ("run_screen", "run_fundamental_screen"):
            for r in (res.get("results") or [])[:6]:
                if len(refs) >= 8:
                    break
                add("symbol", r.get("symbol"), tool)
        elif tool == "get_my_strategies":
            for s in (res.get("strategies") or [])[:6]:
                if len(refs) >= 8:
                    break
                add("strategy", s.get("name"), tool,
                    sublabel=(str(s.get("status") or "").title() or None))
        elif tool == "compile_strategy":
            if res.get("name") and not res.get("needs_clarification"):
                add("strategy", res.get("name"), tool,
                    sublabel=(res.get("timeframe") or None))
        elif tool == "get_ipo_calendar":
            for x in ((res.get("open") or []) + (res.get("upcoming") or []))[:6]:
                if len(refs) >= 8:
                    break
                add("ipo", x.get("symbol") or x.get("company"), tool,
                    sublabel=(x.get("status") or None))
        elif tool == "get_sector_performance":
            for s in (res.get("sectors") or [])[:4]:
                if len(refs) >= 8:
                    break
                add("sector", s.get("sector"), tool, sublabel=s.get("quadrant"))

    return refs[:8]


def _primary_symbol(tool_results: List[Dict[str, Any]]) -> str | None:
    """Best-guess the stock the turn was about — for context-aware follow-ups."""
    for tr in tool_results:
        res = tr.get("result")
        args = tr.get("args") or {}
        if isinstance(res, dict) and not res.get("error"):
            sym = res.get("symbol") or args.get("symbol")
            if sym:
                return str(sym).upper()
        if args.get("symbol"):
            return str(args["symbol"]).upper()
    return None


# Static fallback offered when nothing more specific fits.
_GENERIC_FOLLOWUPS = ["Explain this simply", "Show the risks", "What would change this?"]


def build_followups(
    intent: str,
    tool_results: List[Dict[str, Any]],
    message: str = "",
) -> List[str]:
    """Context-aware follow-up chips — deterministic (0 tokens), keyed on the
    intent, the tools that ran, and the primary symbol. Feels dynamic without a
    second LLM call. Falls back to the generic three."""
    tools = {t.get("tool") for t in tool_results}
    sym = _primary_symbol(tool_results)

    if "run_screen" in tools or "run_fundamental_screen" in tools:
        return ["Which of these looks strongest?", "Make it stricter", "Show more results"]
    if "get_ipo_calendar" in tools:
        return ["Which IPO looks worth watching?", "How does IPO subscription work?", "Show upcoming IPOs"]
    if "compile_strategy" in tools:
        return ["Backtest this strategy", "Tweak the entry rule", "Add a trailing stop"]
    if "get_fno_snapshot" in tools or "suggest_options_strategy" in tools:
        return ["Suggest an option structure", "Explain max pain simply", "What's the PCR telling us?"]
    if "get_fundamentals" in tools and sym:
        return [f"Show {sym}'s technicals", f"Any recent news on {sym}?", f"Is {sym} a buy here?"]
    if ("get_stock_snapshot" in tools or "get_technicals" in tools or "explain_move" in tools) and sym:
        return [f"{sym} fundamentals", f"Key levels for {sym}", f"Bull vs bear case for {sym}"]
    if "get_portfolio" in tools or intent == "portfolio_review":
        return ["Where is my book most fragile?", "How should I rebalance?", "Which position is riskiest?"]
    if "get_current_regime" in tools or "get_sector_performance" in tools or intent == "market_context":
        return ["Which sectors are leading?", "What did FII/DII do today?", "Strongest signals right now"]
    return _GENERIC_FOLLOWUPS


def build_progress(state: AgentState) -> List[Dict[str, Any]]:
    """Project the pipeline stages (``state.turns``) + tool calls
    (``state.tool_trace``) into an ordered, brand-safe timeline
    ``[{stage, label, tool?, status, duration_ms, error?}]``.

    Returns ``[]`` when no tool ran (greeting / refusal / pure-knowledge) so the
    UI shows no empty timeline. Reasoning stages lead in; each tool call is its
    own step (in true execution order, so replan retries + errors render
    coherently — an errored call keeps its own step with ``status='error'`` and
    a generic message, never the raw exception).
    """
    tools = state.tool_trace
    if not tools:
        return []

    steps: List[Dict[str, Any]] = []
    planner_seen = 0
    for turn in state.turns:
        agent = turn.agent
        if agent == "classifier":
            steps.append({"stage": "classifier", "label": _STAGE_LABEL["classifier"],
                          "status": "ok", "duration_ms": int(turn.duration_ms or 0)})
        elif agent == "tool_planner":
            planner_seen += 1
            steps.append({
                "stage": "planner",
                "label": "Re-checking the plan" if planner_seen > 1 else _STAGE_LABEL["tool_planner"],
                "status": "ok",
                "duration_ms": int(turn.duration_ms or 0),
            })
        # ``tool_caller`` + ``responder`` turns are intentionally omitted: the
        # tool calls below ARE the caller's work, and the responder streams as
        # the visible answer.

    for tc in tools:
        label = _tool_public_label(tc.name)
        step: Dict[str, Any] = {
            "stage": "tool",
            "label": label,
            "tool": label,
            "status": "error" if tc.error else "ok",
            "duration_ms": int(tc.duration_ms or 0),
        }
        if tc.error:
            step["error"] = "Couldn't fetch this"
        steps.append(step)

    return steps[:12]


# --------------------------------------------------------- grounding self-check
#
# Deterministic, 0-token sanity check that the numeric claims in the responder's
# reply are actually backed by the tool data it was given. FLAG-ONLY in v1: we
# attach a {grounded, unsupported} flag + log it, but NEVER mutate the reply
# (clean_reply is computed but not used as the outgoing text).

_NUM_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])([+-]?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)(%?)(?![A-Za-z0-9])"
)
_GROUNDING_SAFE = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0})
_GROUNDING_REL_TOL = 0.01
_GROUNDING_ABS_TOL = 0.5


def _flatten_numbers(obj, out) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        n = _num(obj)
        if n is not None:
            out.add(round(n, 4))
        return
    if isinstance(obj, str):
        for m in _NUM_TOKEN_RE.finditer(obj):
            n = _num(m.group(1).replace(",", ""))
            if n is not None:
                out.add(round(n, 4))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)


def build_facts_corpus(tool_results):
    corpus = set()
    _flatten_numbers(tool_results, corpus)
    for n in list(corpus):
        if 0.0 < abs(n) <= 1.0:
            corpus.add(round(n * 100, 4))
        if abs(n) >= 1.0:
            corpus.add(round(n, 0))
    return corpus


def _matches_fact(value, corpus) -> bool:
    if value in _GROUNDING_SAFE:
        return True
    tol = max(_GROUNDING_ABS_TOL, abs(value) * _GROUNDING_REL_TOL)
    return any(abs(f - value) <= tol for f in corpus)


def validate_grounding(reply, tool_results):
    """0-token check that numeric claims in `reply` are backed by tool data.
    FLAG-ONLY: returns {grounded, unsupported, clean_reply}; callers attach the
    flag + log but do NOT mutate the reply in v1."""
    if not reply:
        return {"grounded": True, "unsupported": [], "clean_reply": reply}
    if not tool_results:
        # Pure-knowledge bypass — EXCEPT a reply that cites tools that never
        # ran is fabricating evidence (seen live 2026-06-12: invented
        # [tool:market_breadth] + made-up index levels on a zero-tool turn).
        if "[tool:" in reply:
            return {"grounded": False, "unsupported": ["[tool:] citation with no tool run"],
                    "clean_reply": reply}
        return {"grounded": True, "unsupported": [], "clean_reply": reply}
    corpus = build_facts_corpus(tool_results)
    unsupported = []

    def _repl(m):
        raw, pct = m.group(1), m.group(2)
        val = _num(raw.replace(",", ""))
        if val is None:
            return m.group(0)
        scaled = val * 100 if (pct == "%" and abs(val) <= 1) else val
        if _matches_fact(val, corpus) or (pct == "%" and _matches_fact(scaled, corpus)):
            return m.group(0)
        unsupported.append(m.group(0))
        return "[unverified]"

    clean = _NUM_TOKEN_RE.sub(_repl, reply)
    return {"grounded": not unsupported, "unsupported": unsupported, "clean_reply": clean}


# ------------------------------------------------------------------- responder


# Adaptive responder model — quick/simple turns stay on the fast chat model
# (~2-3s); analytical/complex turns escalate to LLM_STRONG_MODEL for depth. The
# strong client is built once (lazily) and is budget-gated inside LLM.complete
# (spills to the free tier when over the monthly cap).
_STRONG_RESPONDER: "LLM | None" = None


def _strong_responder_llm() -> "LLM":
    global _STRONG_RESPONDER
    if _STRONG_RESPONDER is None:
        _STRONG_RESPONDER = LLM(model=settings.LLM_STRONG_MODEL)
    return _STRONG_RESPONDER


_DEEP_INTENTS = {
    "strategy", "strategy_gen", "analysis", "analyze", "compare", "comparison",
    "portfolio", "options", "research", "recommendation", "screen", "backtest",
    "doctor", "debate", "thesis", "outlook",
}
_ANALYTICAL_HINTS = (
    "why", "analyze", "analyse", "compare", " vs ", "versus", "strategy",
    "should i", "evaluate", "break down", "breakdown", "deep dive", "detailed",
    "thesis", "outlook", "forecast", "explain", "pros and cons", "risk reward",
    "risk-reward", "in detail",
)


def _needs_strong_model(state: AgentState) -> bool:
    """True when the turn is analytical enough to justify the strong model."""
    if not settings.COPILOT_ADAPTIVE_MODEL:
        return False
    intent = str(state.get("classifier", "intent", "") or "").lower()
    if intent in _DEEP_INTENTS:
        return True
    tool_results = state.get("tool_caller", "tool_results") or []
    if len(tool_results) >= 2:  # multi-tool synthesis → analytical
        return True
    msg = str(state.inputs.get("message", "") or "")
    if len(msg) > 220 or msg.count("?") >= 2:  # long / multi-part question
        return True
    ml = f" {msg.lower()} "
    return any(h in ml for h in _ANALYTICAL_HINTS)


class CopilotResponder(Agent):
    name = "responder"

    async def _run(self, state: AgentState) -> None:
        system = _RESPONDER_SYSTEM
        prompt = _build_responder_prompt(state)
        # Adaptive: strong model for analytical turns, fast model otherwise.
        llm = _strong_responder_llm() if _needs_strong_model(state) else self.llm
        reply = await llm.complete(prompt, system=system, temperature=0.25)
        if not reply:
            reply = _FALLBACK_REPLY
        tool_results = state.get("tool_caller", "tool_results") or []
        grounding = validate_grounding(reply, tool_results)  # on RAW (sees [tool:])
        if not grounding["grounded"]:
            logger.info("grounding: %d unsupported numeric claim(s): %s",
                        len(grounding["unsupported"]), grounding["unsupported"][:5])
        reply = _strip_citations(reply)  # clean for display/persist
        state.put(self.name, response=reply, grounding=grounding)
        state.output = {
            "reply": reply,
            "refused": False,
            "intent": state.get("classifier", "intent", "other"),
            "tools_used": [t["tool"] for t in tool_results],
            "grounding": {"grounded": grounding["grounded"], "unsupported": grounding["unsupported"]},
        }


# --------------------------------------------------------------------- runner


# Singleton nodes — shared by the explicit stage driver (which can re-plan)
# AND the back-compat GraphRunner. One instance each keeps the LLM clients +
# any node state consistent between the two entry points.
_CLASSIFIER = CopilotClassifier(llm=llm_for("classifier"))
_PLANNER = CopilotToolPlanner(llm=llm_for("tool_planner"))
_CALLER = CopilotToolCaller()
_RESPONDER = CopilotResponder(llm=llm_for("responder"))

# Back-compat: the linear graph (no re-plan loop). run_copilot drives the
# stages explicitly below so it can insert the bounded re-plan loop.
COPILOT_GRAPH = GraphRunner(
    "copilot",
    [_CLASSIFIER, _PLANNER, _CALLER, _RESPONDER],
)


async def run_copilot(
    *,
    user_id: str,
    message: str,
    route: str = "",
    history: list | None = None,
    mentioned_symbols: list | None = None,
    memory: str = "",
) -> Dict[str, Any]:
    """Single-turn Copilot call. Returns ``state.output`` + trace."""
    state = AgentState(
        inputs={
            "message": message,
            "route": route,
            "history": history or [],
            "mentioned_symbols": mentioned_symbols or [],
            "memory": memory or "",
        },
        user_id=user_id,
        graph_name="copilot",
    )
    try:
        await _CLASSIFIER.run(state)
        await _PLANNER.run(state)
        await _CALLER.run(state)
        await _replan_loop(state, _PLANNER, _CALLER)
        await _RESPONDER.run(state)
    except EarlyExit:
        pass
    tool_results = state.get("tool_caller", "tool_results") or []
    return {
        **state.output,
        "trace": [
            {"agent": t.agent, "duration_ms": t.duration_ms}
            for t in state.turns
        ],
        "tool_calls": [
            {"name": tc.name, "duration_ms": tc.duration_ms, "error": tc.error}
            for tc in state.tool_trace
        ],
        # Transparent-agent rails (brand-safe, whitelisted projections).
        "progress": build_progress(state),
        "references": build_references(tool_results, reply=state.output.get("reply", "")),
    }


# Pre-responder stages, run in order so we can token-stream the final reply.
_PRE_RESPONDER = (_CLASSIFIER, _PLANNER, _CALLER)


async def run_copilot_stream(
    *,
    user_id: str,
    message: str,
    route: str = "",
    history: list | None = None,
    mentioned_symbols: list | None = None,
    memory: str = "",
):
    """Streaming Copilot turn — async-generator of transport-agnostic events.

    Runs classifier → planner → caller eagerly (fast, non-streamed), emits a
    ``meta`` event with the consulted tools + structured artifacts + the honest
    Progress timeline + References (F1 batched), then token-streams the
    responder. Event shapes (the HTTP layer wraps these in SSE)::

        {"type": "meta",  "tools_used": [...], "artifacts": [...], "intent": "...",
                          "progress": [{"stage","label","tool?","status","duration_ms","error?"}],
                          "references": [{"kind","label","sublabel?","tool","id?","cited?"}]}
        {"type": "token", "text": "..."}                       # repeated
        {"type": "done",  "reply": "<full>", "intent": "...", "tools_used": [...],
                          "refused": false, "references": [...]}
    """
    state = AgentState(
        inputs={
            "message": message,
            "route": route,
            "history": history or [],
            "mentioned_symbols": mentioned_symbols or [],
            "memory": memory or "",
        },
        user_id=user_id,
        graph_name="copilot",
    )

    # ── pre-responder stages (honor EarlyExit → canned reply) ──
    early: Dict[str, Any] | None = None
    for agent in _PRE_RESPONDER:
        try:
            await agent.run(state)
        except EarlyExit:
            early = state.output or {}
            break
        except Exception as exc:  # a stage failed — degrade to a one-shot reply
            logger.warning("copilot stream stage %s failed: %s", agent.name, exc)
            early = {
                "reply": "I hit a snag pulling live data just now. Try that again in a moment.",
                "refused": False,
            }
            break

    # ── bounded re-plan loop (only when no early exit fired) ──
    if early is None:
        try:
            await _replan_loop(state, _PLANNER, _CALLER)
        except EarlyExit:
            early = state.output or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("copilot stream re-plan failed: %s", exc)

    intent = state.get("classifier", "intent", "other")
    tool_results = state.get("tool_caller", "tool_results") or []
    tools_used = [t["tool"] for t in tool_results]
    artifacts = build_artifacts(tool_results)
    # Honest telemetry (F1 batched — projected from the already-run stages, NOT a
    # per-step live pipeline). ``progress`` is complete here (all pre-responder
    # stages have run); ``references`` are the consulted entities (``cited`` is
    # filled at the ``done`` event once the reply text is known).
    progress = build_progress(state)
    references = build_references(tool_results)

    # Charts + tool chips + the Progress timeline + References render before the
    # prose streams in. New fields are additive — ``meta`` stays back-compatible.
    yield {
        "type": "meta",
        "tools_used": tools_used,
        "artifacts": artifacts,
        "intent": intent,
        "progress": progress,
        "references": references,
    }

    # ── early exit: stream the canned reply as one chunk ──
    if early is not None:
        reply = early.get("reply", "") or _FALLBACK_REPLY
        yield {"type": "token", "text": reply}
        yield {
            "type": "done",
            "reply": reply,
            "intent": early.get("intent", intent),
            "tools_used": tools_used,
            "refused": bool(early.get("refused", False)),
            # Empty on greeting/refusal (no tools) — the UI renders no rails.
            "references": build_references(tool_results, reply=reply),
        }
        return

    # ── stream the responder ──
    prompt = _build_responder_prompt(state)
    chunks: List[str] = []
    try:
        async for piece in llm_for("responder").complete_stream(
            prompt,
            system=_RESPONDER_SYSTEM,
            temperature=0.25,
            user_id=user_id,
            feature="copilot",
        ):
            chunks.append(piece)
            yield {"type": "token", "text": piece}
    except Exception as exc:
        logger.warning("copilot responder stream failed: %s", exc)

    reply = "".join(chunks).strip()
    if not reply:
        reply = _FALLBACK_REPLY
        yield {"type": "token", "text": reply}

    grounding = validate_grounding(reply, tool_results)  # on RAW (sees [tool:])
    if not grounding["grounded"]:
        logger.info("grounding(stream): %d unsupported claim(s): %s",
                    len(grounding["unsupported"]), grounding["unsupported"][:5])

    clean_reply = _strip_citations(reply)  # clean for display/persist
    yield {
        "type": "done",
        "reply": clean_reply,
        "intent": intent,
        "tools_used": tools_used,
        "refused": False,
        "grounding": {"grounded": grounding["grounded"], "unsupported": grounding["unsupported"]},
        # Re-emit references now that the reply is known → ``cited`` flags set.
        "references": build_references(tool_results, reply=reply),
        # Context-aware follow-up chips (deterministic, 0 tokens).
        "followups": build_followups(intent, tool_results, message),
    }

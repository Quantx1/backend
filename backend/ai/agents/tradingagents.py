"""
TradingAgents Bull/Bear debate (B1) — 7-agent graph.

Triggered on high-stakes signals (Elite tier) per Step 1 §B1. Surface:
signal detail page → "Debate" tab → renders the `DebateTranscript`
component with each agent's vote + reasoning.

Graph:

    ┌── FundamentalsAnalyst ──┐                  ┌─ BullResearcher ─┐
    ├── TechnicalAnalyst ─────┼─→ Manager ─→    ┤                  ├─→ RiskManager ─→ Trader
    └── SentimentAnalyst ─────┘                  └─ BearResearcher ─┘

- Three analysts produce fact-grounded reports (parallel).
- Manager distills analyst reports into a briefing for both researchers.
- Bull + Bear researchers debate (parallel, seeded with the manager brief).
- RiskManager reads both sides + VIX + regime → position-size cap.
- Trader writes the final verdict: enter / skip / half-size / wait.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .base import Agent, GraphRunner
from .llm import llm_for
from .state import AgentState

logger = logging.getLogger(__name__)


def _signal(state: AgentState) -> Dict[str, Any]:
    return state.inputs.get("signal") or {}


# ============================================================================
# LAYER 1 — 3 ANALYSTS (parallel)
# ============================================================================


class FundamentalsAnalyst(Agent):
    name = "fundamentals_analyst"

    async def _run(self, state: AgentState) -> None:
        signal = _signal(state)
        fundamentals = state.inputs.get("fundamentals") or {}
        schema = (
            '{"stance": "support|oppose|neutral", "confidence": 0-100, '
            '"points": ["string"]}'
        )
        system = (
            "You are the fundamentals analyst on a TradingAgents debate. "
            "Do the fundamentals support this trade? Return JSON only."
        )
        signal_keys = [
            'symbol', 'direction', 'entry_price',
            'target_1', 'stop_loss', 'risk_reward',
        ]
        signal_json = json.dumps({k: signal.get(k) for k in signal_keys})
        fundamentals_json = json.dumps(fundamentals, default=str)[:3000]
        prompt = (
            f"Signal: {signal_json}\n"
            f"Fundamentals JSON: {fundamentals_json}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)


class TechnicalAnalyst(Agent):
    name = "technical_analyst"

    async def _run(self, state: AgentState) -> None:
        signal = _signal(state)
        snapshot = state.inputs.get("stock_snapshot") or {}
        schema = (
            '{"stance": "support|oppose|neutral", "confidence": 0-100, '
            '"points": ["string"]}'
        )
        system = (
            "You are the technicals analyst. Judge whether chart structure, "
            "momentum, and volume agree with the proposed entry. Return JSON only."
        )
        prompt = (
            f"Signal: {json.dumps(signal, default=str)[:1500]}\n"
            f"Stock snapshot JSON: {json.dumps(snapshot, default=str)}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)


class SentimentAnalyst(Agent):
    name = "sentiment_analyst"

    async def _run(self, state: AgentState) -> None:
        signal = _signal(state)
        news = state.inputs.get("news_headlines") or []
        schema = (
            '{"stance": "support|oppose|neutral", "confidence": 0-100, '
            '"points": ["string"]}'
        )
        system = (
            "You are the sentiment analyst. Use recent news + FinBERT "
            "sentiment scores (if provided) to judge whether crowd sentiment "
            "supports this trade. Return JSON only."
        )
        prompt = (
            f"Signal: {json.dumps(signal, default=str)[:1000]}\n"
            f"Recent headlines JSON: {json.dumps(news)[:3000]}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)


# ============================================================================
# LAYER 2 — MANAGER (distills analyst briefings)
# ============================================================================


class DebateManager(Agent):
    name = "manager"

    async def _run(self, state: AgentState) -> None:
        fa = state.get("fundamentals_analyst")
        ta = state.get("technical_analyst")
        sa = state.get("sentiment_analyst")

        system = (
            "You are the debate manager. Produce a 3-sentence briefing for "
            "the Bull and Bear researchers. Neutral tone; list the strongest "
            "support and strongest objection surfaced by the analysts. "
            "Return JSON only."
        )
        schema = '{"briefing": "string", "top_supports": ["string"], "top_objections": ["string"]}'
        prompt = (
            f"Fundamentals: {json.dumps(fa)}\n"
            f"Technical: {json.dumps(ta)}\n"
            f"Sentiment: {json.dumps(sa)}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)


# ============================================================================
# LAYER 3 — 2 RESEARCHERS (parallel, seeded with manager briefing)
# ============================================================================


class BullResearcher(Agent):
    name = "bull_researcher"

    async def _run(self, state: AgentState) -> None:
        manager = state.get("manager")
        signal = _signal(state)
        system = (
            "You are the Bull researcher. Steelman the case for entering "
            "this trade. Include expected return, why now, what confirms the "
            "thesis. Return JSON only."
        )
        schema = (
            '{"verdict": "strong_buy|buy|neutral", "confidence": 0-100, '
            '"argument": "2-3 sentences", "key_evidence": ["string"]}'
        )
        prompt = (
            f"Signal: {json.dumps(signal, default=str)[:1500]}\n"
            f"Manager briefing: {json.dumps(manager)}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)


class BearResearcher(Agent):
    name = "bear_researcher"

    async def _run(self, state: AgentState) -> None:
        manager = state.get("manager")
        signal = _signal(state)
        system = (
            "You are the Bear researcher. Steelman the case for skipping or "
            "fading this trade. Include downside risk, what invalidates, and "
            "the hidden assumptions the bull must prove. Return JSON only."
        )
        schema = (
            '{"verdict": "strong_skip|skip|neutral", "confidence": 0-100, '
            '"argument": "2-3 sentences", "key_evidence": ["string"]}'
        )
        prompt = (
            f"Signal: {json.dumps(signal, default=str)[:1500]}\n"
            f"Manager briefing: {json.dumps(manager)}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)


# ============================================================================
# LAYER 4 — RISK MANAGER (reads both sides + VIX/regime)
# ============================================================================


class RiskManager(Agent):
    name = "risk_manager"

    async def _run(self, state: AgentState) -> None:
        bull = state.get("bull_researcher")
        bear = state.get("bear_researcher")
        regime = state.inputs.get("regime") or {}
        vix = state.inputs.get("vix")
        system = (
            "You are the risk manager. Recommend a position-size multiplier "
            "between 0.0 and 1.0 (0 = skip, 1 = full-size) and a kill-switch "
            "trigger rule. Return JSON only."
        )
        schema = (
            '{"size_multiplier": 0.0-1.0, "rationale": "string", '
            '"kill_switch_trigger": "string"}'
        )
        prompt = (
            f"Bull case: {json.dumps(bull)}\n"
            f"Bear case: {json.dumps(bear)}\n"
            f"Regime: {json.dumps(regime, default=str)}\n"
            f"India VIX: {vix}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)


# ============================================================================
# LAYER 5 — TRADER (final verdict)
# ============================================================================


class Trader(Agent):
    name = "trader"

    async def _run(self, state: AgentState) -> None:
        bull = state.get("bull_researcher")
        bear = state.get("bear_researcher")
        risk = state.get("risk_manager")
        signal = _signal(state)

        system = (
            "You are the trader. Combine the bull case, bear case, and risk "
            "rule into a single decision. Return JSON only."
        )
        schema = (
            '{"decision": "enter|skip|half_size|wait", '
            '"confidence": 0-100, "summary": "2-sentence verdict"}'
        )
        prompt = (
            f"Signal: {json.dumps(signal, default=str)[:1000]}\n"
            f"Bull: {json.dumps(bull)}\n"
            f"Bear: {json.dumps(bear)}\n"
            f"Risk: {json.dumps(risk)}"
        )
        parsed = await self.llm.generate_json(
            prompt, schema, system=system,
            user_id=state.user_id, feature="debate",
            metadata={"agent": self.name},
        )
        state.put(self.name, prompt=prompt, response=json.dumps(parsed), **parsed)

        state.output = {
            "signal_id": signal.get("id"),
            "symbol": signal.get("symbol"),
            "decision": parsed.get("decision", "skip"),
            "confidence": int(parsed.get("confidence", 0) or 0),
            "summary": parsed.get("summary", ""),
            "transcript": {
                "fundamentals": state.get("fundamentals_analyst"),
                "technical": state.get("technical_analyst"),
                "sentiment": state.get("sentiment_analyst"),
                "manager_briefing": state.get("manager"),
                "bull": bull,
                "bear": bear,
                "risk": risk,
                "trader": parsed,
            },
        }


# --------------------------------------------------------------------- runner


DEBATE_GRAPH = GraphRunner(
    "tradingagents_debate",
    [
        # Parallel: 3 analysts run concurrently.
        [
            FundamentalsAnalyst(llm=llm_for("debate")),
            TechnicalAnalyst(llm=llm_for("debate")),
            SentimentAnalyst(llm=llm_for("debate")),
        ],
        # Sequential: manager distills.
        DebateManager(llm=llm_for("debate")),
        # Parallel: bull + bear researchers debate from manager brief.
        [BullResearcher(llm=llm_for("debate")), BearResearcher(llm=llm_for("debate"))],
        # Sequential: risk manager reads both sides.
        RiskManager(llm=llm_for("debate")),
        # Sequential: trader decides.
        Trader(llm=llm_for("debate")),
    ],
)


def _build_debate_graph(*, deep: bool, tier: Optional[str]) -> GraphRunner:
    """Per-run graph whose agents resolve to the deep model (Elite + flag on)."""
    def mk():
        return llm_for("debate", deep=deep, tier=tier)
    return GraphRunner("tradingagents_debate", [
        [FundamentalsAnalyst(llm=mk()), TechnicalAnalyst(llm=mk()),
         SentimentAnalyst(llm=mk())],
        DebateManager(llm=mk()),
        [BullResearcher(llm=mk()), BearResearcher(llm=mk())],
        RiskManager(llm=mk()),
        Trader(llm=mk()),
    ])


async def run_trading_debate(
    *,
    user_id: str,
    signal: Dict[str, Any],
    fundamentals: Dict[str, Any] | None = None,
    stock_snapshot: Dict[str, Any] | None = None,
    news_headlines: list | None = None,
    regime: Dict[str, Any] | None = None,
    vix: float | None = None,
    deep: bool = False,
    tier: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the 7-agent Bull/Bear debate on one signal."""
    state = AgentState(
        inputs={
            "signal": signal,
            "fundamentals": fundamentals or {},
            "stock_snapshot": stock_snapshot or {},
            "news_headlines": news_headlines or [],
            "regime": regime or {},
            "vix": vix,
        },
        user_id=user_id,
        graph_name="tradingagents_debate",
    )
    graph = _build_debate_graph(deep=deep, tier=tier) if deep else DEBATE_GRAPH
    await graph.run(state)
    return {
        **state.output,
        "trace": [
            {"agent": t.agent, "duration_ms": t.duration_ms}
            for t in state.turns
        ],
    }

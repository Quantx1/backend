"""
FinRobot Chain-of-Thought graph — 4 specialist agents + 1 synthesizer.

Used by:
- **F5 AI SIP** — pick / rebalance candidate stocks in the long-term
  portfolio.
- **F7 Portfolio Doctor** — audit the user's holdings. Each agent runs
  per holding; the synthesizer produces the shareable PDF report.

Graph (parallel fan-out, then sequential synthesize):

    ┌─── FundamentalAgent ───┐
    │                        │
    ├─── ManagementAgent ────┤
    │                        ├──→ SynthesizerAgent ──→ report
    ├─── PromoterAgent ──────┤
    │                        │
    └─── PeerAgent ──────────┘

Data sources in v1: yfinance fundamentals + FinBERT-India sentiment on
news (when PR 11 ships) + concall-tone analysis stubbed (real transcript
scraping lands with the F9 PR). Every agent gracefully handles missing
fields.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import Agent, GraphRunner
from .llm import llm_for
from .state import AgentState

logger = logging.getLogger(__name__)


def _symbol(state: AgentState) -> str:
    return str(state.inputs.get("symbol", "")).upper()


# -------------------------------------------------------------- fundamental


class FundamentalAgent(Agent):
    name = "fundamental"

    async def _run(self, state: AgentState) -> None:
        symbol = _symbol(state)
        fundamentals = state.inputs.get("fundamentals") or {}
        schema = (
            '{"grade": "A+|A|B+|B|C|D|F", "verdict": "short string", '
            '"flags": ["string"], "score": 0-100}'
        )
        system = (
            "You are an NSE-literate equity fundamentals analyst. Grade the "
            "company on ROE, Debt/Equity, earnings trend, and valuation "
            "(P/E vs sector). Return JSON only. No prose."
        )
        prompt = (
            f"Symbol: {symbol}\n"
            f"Fundamentals JSON: {json.dumps(fundamentals, default=str)}\n\n"
            "Produce the grade."
        )
        parsed = await self.llm.generate_json(prompt, schema, system=system)
        state.put(
            self.name,
            prompt=prompt,
            response=json.dumps(parsed),
            grade=parsed.get("grade", "C"),
            score=int(parsed.get("score", 50) or 50),
            verdict=parsed.get("verdict", ""),
            flags=parsed.get("flags", []) or [],
        )


# -------------------------------------------------------------- management


class ManagementAgent(Agent):
    name = "management_tone"

    async def _run(self, state: AgentState) -> None:
        symbol = _symbol(state)
        transcript = state.inputs.get("concall_transcript", "")
        headlines = state.inputs.get("management_headlines") or []

        if not transcript and not headlines:
            # Nothing to analyze — output neutral placeholder.
            state.put(
                self.name,
                prompt="(no concall / headlines)",
                response='{"tone": "neutral", "confidence": 0, "flags": []}',
                tone="neutral",
                confidence=0,
                flags=[],
            )
            return

        schema = (
            '{"tone": "bullish|neutral|bearish", "confidence": 0-100, '
            '"forward_guidance": "string", "flags": ["string"]}'
        )
        system = (
            "You analyze Indian company management commentary for tone and "
            "forward guidance nuance. Look for hedging, numerical revisions, "
            "and strategic pivots. Return JSON only."
        )
        prompt = (
            f"Symbol: {symbol}\n"
            f"Recent management headlines: {json.dumps(headlines)[:2000]}\n\n"
            f"Concall transcript excerpt:\n{(transcript or '')[:4000]}\n"
        )
        parsed = await self.llm.generate_json(prompt, schema, system=system)
        state.put(
            self.name,
            prompt=prompt,
            response=json.dumps(parsed),
            tone=parsed.get("tone", "neutral"),
            confidence=int(parsed.get("confidence", 0) or 0),
            guidance=parsed.get("forward_guidance", ""),
            flags=parsed.get("flags", []) or [],
        )


# ----------------------------------------------------------------- promoter


class PromoterAgent(Agent):
    name = "promoter_holding"

    async def _run(self, state: AgentState) -> None:
        symbol = _symbol(state)
        holding = state.inputs.get("promoter_holding") or {}
        schema = (
            '{"trend": "rising|stable|falling", "pledged_pct": 0-100, '
            '"concern_level": "low|medium|high", "verdict": "string"}'
        )
        system = (
            "You assess promoter shareholding patterns in Indian listed "
            "companies. Falling promoter holding + rising pledges = red flag. "
            "Return JSON only."
        )
        prompt = (
            f"Symbol: {symbol}\n"
            f"Promoter holding history JSON: {json.dumps(holding, default=str)}"
        )
        parsed = await self.llm.generate_json(prompt, schema, system=system)
        state.put(
            self.name,
            prompt=prompt,
            response=json.dumps(parsed),
            trend=parsed.get("trend", "stable"),
            pledged_pct=parsed.get("pledged_pct", 0),
            concern_level=parsed.get("concern_level", "low"),
            verdict=parsed.get("verdict", ""),
        )


# -------------------------------------------------------------------- peer


class PeerAgent(Agent):
    name = "peer_comparison"

    async def _run(self, state: AgentState) -> None:
        symbol = _symbol(state)
        peers = state.inputs.get("peers") or []
        schema = (
            '{"rank_percentile": 0-100, "strengths": ["string"], '
            '"weaknesses": ["string"], "verdict": "string"}'
        )
        system = (
            "You rank a stock against its sector peers across ROE, margin, "
            "valuation, and growth. Output percentile (100 = best in sector). "
            "Return JSON only."
        )
        prompt = (
            f"Subject: {symbol}\n"
            f"Peers with metrics JSON: {json.dumps(peers, default=str)[:4000]}"
        )
        parsed = await self.llm.generate_json(prompt, schema, system=system)
        state.put(
            self.name,
            prompt=prompt,
            response=json.dumps(parsed),
            rank_percentile=int(parsed.get("rank_percentile", 50) or 50),
            strengths=parsed.get("strengths", []) or [],
            weaknesses=parsed.get("weaknesses", []) or [],
            verdict=parsed.get("verdict", ""),
        )


# ---------------------------------------------------------------- synthesizer


class SynthesizerAgent(Agent):
    name = "synthesizer"

    async def _run(self, state: AgentState) -> None:
        symbol = _symbol(state)
        f = state.get("fundamental")
        m = state.get("management_tone")
        p = state.get("promoter_holding")
        peer = state.get("peer_comparison")

        system = (
            "You synthesize four specialist reports into one decision-ready "
            "verdict for a retail investor. Senior-analyst voice. No emoji. "
            "Max 4 short paragraphs + a 1-line 'Action' at the end."
        )
        prompt = (
            f"Symbol: {symbol}\n\n"
            f"Fundamental report: {json.dumps(f)}\n"
            f"Management report: {json.dumps(m)}\n"
            f"Promoter report: {json.dumps(p)}\n"
            f"Peer report: {json.dumps(peer)}\n\n"
            "Write the synthesis. End with exactly one line that starts "
            "'Action:' and is either 'add', 'hold', 'trim', or 'exit'."
        )
        narrative = await self.llm.complete(prompt, system=system, temperature=0.2)
        narrative = (narrative or "").strip()

        action = "hold"
        for candidate in ("add", "hold", "trim", "exit"):
            marker = f"Action: {candidate}"
            if marker in narrative.lower():
                action = candidate
                break

        # Composite score: simple weighted blend.
        fund_score = int((f or {}).get("score", 50) or 50)
        tone_score = 50 + int((m or {}).get("confidence", 0) or 0) // 4
        if (m or {}).get("tone") == "bearish":
            tone_score = 100 - tone_score
        promoter_penalty = {"low": 0, "medium": -10, "high": -25}.get(
            (p or {}).get("concern_level", "low"), 0
        )
        peer_score = int((peer or {}).get("rank_percentile", 50) or 50)
        composite = max(
            0, min(100, int(
                0.40 * fund_score
                + 0.20 * tone_score
                + 0.15 * peer_score
                + 0.25 * 50  # neutral base for promoter
                + promoter_penalty
            ))
        )

        state.put(self.name, prompt=prompt, response=narrative)
        state.output = {
            "symbol": symbol,
            "narrative": narrative,
            "action": action,
            "composite_score": composite,
            "agents": {
                "fundamental": f,
                "management_tone": m,
                "promoter_holding": p,
                "peer_comparison": peer,
            },
        }


# --------------------------------------------------------------------- runner


DOCTOR_GRAPH = GraphRunner(
    "finrobot_doctor",
    [
        # Parallel fan-out — 4 specialists run concurrently.
        [
            FundamentalAgent(llm=llm_for("doctor")),
            ManagementAgent(llm=llm_for("doctor")),
            PromoterAgent(llm=llm_for("doctor")),
            PeerAgent(llm=llm_for("doctor")),
        ],
        # Sequential synthesize.
        SynthesizerAgent(llm=llm_for("doctor")),
    ],
)


def _build_doctor_graph(*, deep: bool, tier: Optional[str]) -> GraphRunner:
    """Per-run graph whose agents resolve to the deep model (Elite + flag on)."""
    def mk():
        return llm_for("doctor", deep=deep, tier=tier)
    return GraphRunner("finrobot_doctor", [
        [FundamentalAgent(llm=mk()), ManagementAgent(llm=mk()),
         PromoterAgent(llm=mk()), PeerAgent(llm=mk())],
        SynthesizerAgent(llm=mk()),
    ])


async def run_finrobot_doctor(
    *,
    user_id: str,
    symbol: str,
    fundamentals: Dict[str, Any] | None = None,
    concall_transcript: str | None = None,
    management_headlines: List[str] | None = None,
    promoter_holding: Dict[str, Any] | None = None,
    peers: List[Dict[str, Any]] | None = None,
    deep: bool = False,
    tier: Optional[str] = None,
) -> Dict[str, Any]:
    """Run FinRobot CoT on a single symbol. Missing inputs are fine —
    each agent falls back to a neutral output."""
    state = AgentState(
        inputs={
            "symbol": symbol,
            "fundamentals": fundamentals or {},
            "concall_transcript": concall_transcript or "",
            "management_headlines": management_headlines or [],
            "promoter_holding": promoter_holding or {},
            "peers": peers or [],
        },
        user_id=user_id,
        graph_name="finrobot_doctor",
    )
    graph = _build_doctor_graph(deep=deep, tier=tier) if deep else DOCTOR_GRAPH
    await graph.run(state)
    return {
        **state.output,
        "trace": [
            {"agent": t.agent, "duration_ms": t.duration_ms}
            for t in state.turns
        ],
    }

"""
Agent + GraphRunner abstractions.

An ``Agent`` is a unit of work that reads ``AgentState``, optionally calls
tools + LLM, and writes back into ``state.scratch[self.name]``. Graphs are
just ordered lists of Agents (or lists-of-lists for parallel fan-out).

Composition patterns we support in practice:

- Sequential chain (FinRobot CoT):
    [fundamental_agent, management_tone_agent, promoter_holding_agent,
     peer_comparison_agent, synthesizer_agent]

- Parallel fan-out + aggregate (TradingAgents):
    [[fundamentals_analyst, technical_analyst, sentiment_analyst],
     [bull_researcher, bear_researcher],
     [risk_manager],
     [trader]]

- Linear tool-use (Copilot):
    [classifier_agent, tool_planner_agent, tool_caller_agent, responder_agent]

Any agent may return early by setting ``state.output`` + raising
``EarlyExit`` — the runner short-circuits the rest of the graph.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Sequence, Union

from .llm import LLM, get_llm
from .state import AgentState, AgentTurn

logger = logging.getLogger(__name__)


class EarlyExit(Exception):
    """An agent can raise this to short-circuit the graph (e.g., a
    classifier deciding the message is out of scope).
    """


class Agent(ABC):
    """Base class for every graph node.

    Subclass hooks:
        name      : stable identifier written into state.scratch[name]
        _run      : the actual work. Receives state, returns nothing
                    (side-effects via state.put / state.output).
    """

    name: str = "agent"

    def __init__(self, llm: LLM | None = None):
        self._llm_override = llm

    @property
    def llm(self) -> LLM:
        return self._llm_override or get_llm()

    @abstractmethod
    async def _run(self, state: AgentState) -> None:
        ...

    async def run(self, state: AgentState) -> None:
        """Wrap ``_run`` with turn-level telemetry on the shared state."""
        t0 = time.monotonic()
        try:
            await self._run(state)
        except EarlyExit:
            raise
        except Exception as e:
            state.put(self.name, error=str(e))
            logger.warning("Agent %s failed: %s", self.name, e)
            raise
        finally:
            duration_ms = int((time.monotonic() - t0) * 1000)
            bucket = state.scratch.get(self.name, {}) or {}
            prompt = bucket.get("prompt", "")
            response = bucket.get("response", "") or bucket.get("result", "")
            state.turns.append(
                AgentTurn(
                    agent=self.name,
                    prompt=str(prompt)[:500] if prompt else "",
                    response=str(response)[:1500] if response else "",
                    duration_ms=duration_ms,
                )
            )


# --------------------------------------------------------------- graph runner

GraphNode = Union[Agent, Sequence[Agent]]  # bare agent = serial step; list = parallel


class GraphRunner:
    """Execute a graph of agents over shared ``AgentState``.

    Serial step: ``Agent``  → awaited to completion before next step.
    Parallel step: ``[Agent, Agent, ...]`` → awaited with
    ``asyncio.gather``; state is mutated concurrently (each agent writes
    its own scratch slot — keep agent writes disjoint).
    """

    def __init__(self, name: str, nodes: List[GraphNode]):
        self.name = name
        self.nodes = nodes

    async def run(self, state: AgentState) -> AgentState:
        state.graph_name = state.graph_name or self.name
        state.started_at = state.started_at or datetime.utcnow()

        for step, node in enumerate(self.nodes):
            try:
                if isinstance(node, Agent):
                    await node.run(state)
                else:
                    await asyncio.gather(*(agent.run(state) for agent in node))
            except EarlyExit:
                logger.info(
                    "Graph %s short-circuited at step %d", self.name, step
                )
                break

        return state

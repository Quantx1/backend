"""
Shared agent state — typed dataclass threaded through every graph.

Each graph reads `inputs`, writes to `scratch` (intermediate artifacts
keyed by agent name) and `output` (final response payload). Tool calls
append to `tool_trace` for observability + debugging + the Copilot
"Context" tab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    result: Any
    started_at: datetime
    duration_ms: int
    error: Optional[str] = None


@dataclass
class AgentTurn:
    """One agent node's contribution to the graph run."""
    agent: str
    prompt: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0


@dataclass
class AgentState:
    """Shared state for one graph run.

    - ``inputs``    : everything the caller passed in (user query, signal_id,
                      symbol, portfolio_csv, etc.). Read-only during the run.
    - ``scratch``   : working memory written by each agent; keyed by agent
                      name.
    - ``turns``     : every agent run recorded in order.
    - ``tool_trace``: every tool invocation recorded in order.
    - ``output``    : final response payload the caller returns to the user.
    """

    inputs: Dict[str, Any]
    scratch: Dict[str, Any] = field(default_factory=dict)
    turns: List[AgentTurn] = field(default_factory=list)
    tool_trace: List[ToolCall] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    graph_name: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)

    def put(self, agent: str, **fields: Any) -> None:
        """Merge fields into this agent's scratch slot."""
        bucket = self.scratch.setdefault(agent, {})
        bucket.update(fields)

    def get(self, agent: str, key: Optional[str] = None, default: Any = None) -> Any:
        bucket = self.scratch.get(agent, {})
        if key is None:
            return bucket
        return bucket.get(key, default)

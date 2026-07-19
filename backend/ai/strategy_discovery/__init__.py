"""Strategy Discovery Engine — generate, evaluate, and persist new
trading strategies via systematic search over the DSL parameter space.

LOCKED: candidates come from deterministic search (random / Sobol /
genetic), NEVER from LLM authoring. See
``project_agents_decision_2026_05_10`` in memory.
"""

from .search_space import EquitySearchSpace, FOSearchSpace
from .evaluator import evaluate_candidate, CandidateScore
from .runner import run_discovery, DiscoveryConfig

__all__ = [
    "EquitySearchSpace",
    "FOSearchSpace",
    "evaluate_candidate",
    "CandidateScore",
    "run_discovery",
    "DiscoveryConfig",
]

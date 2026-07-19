"""Genetic-algorithm sampler for the Strategy Discovery Engine.

Wraps any `SearchSpace` (EquitySearchSpace / FOSearchSpace) and turns its
`sample`/`mutate` primitives into a multi-generation search:

    Gen 0: draw N random candidates                  (search_space.sample)
    Score, take top-K survivors                       (elitism)
    Gen 1: each survivor spawns C mutated children    (search_space.mutate)
            + fill the rest with fresh random candidates for diversity
    Repeat for G generations

Each candidate persists `parent_id` and `generation` (0..G-1) so the UI
can render lineage. The composite score from the scorer drives selection.

LOCKED: mutation is bounded knob-perturbation in the DSL parameter space.
LLMs never author or modify strategies — see project_agents_decision_2026_05_10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple

from backend.ai.strategy.dsl import Strategy

logger = logging.getLogger(__name__)


class _Space(Protocol):
    """Subset of EquitySearchSpace / FOSearchSpace the GA needs."""

    def sample(self, idx: int = 0) -> Strategy: ...
    def mutate(self, parent: Strategy, idx: int = 0) -> Strategy: ...


@dataclass
class GAConfig:
    """Inputs that fully determine a GA-search batch.

    `total_budget` caps how many candidates we ever score — this is the
    most expensive knob (each candidate = M backtests). A typical run is
    pop_size=12, generations=3, elite=4, children=2 → 12 + 12 + 12 = 36
    scored candidates.
    """
    pop_size: int = 12                 # candidates scored per generation
    generations: int = 3               # number of generations
    elite: int = 4                     # top-K survivors that breed next gen
    children_per_elite: int = 2        # mutations per survivor (rest = random)

    def total_budget(self) -> int:
        """Upper bound on the number of scorings a run will perform."""
        return self.pop_size * self.generations


@dataclass
class _ScoredCandidate:
    """Internal — pairs a Strategy with its score for ranking."""
    strategy: Strategy
    score: float
    generation: int
    parent_label: Optional[str] = None      # name of the parent strategy


def evolve(
    space: _Space,
    score_fn: Callable[[Strategy], float],
    *,
    config: GAConfig,
    on_candidate: Optional[Callable[[Strategy, float, int, Optional[str]], None]] = None,
) -> List[Tuple[Strategy, float, int, Optional[str]]]:
    """Run the GA over a given search space.

    `score_fn(strategy) -> float`: caller-supplied scorer. The GA only
    cares about the comparable score number; how it's computed
    (per-symbol mean / walk-forward / etc.) is the caller's choice.

    `on_candidate(strategy, score, generation, parent_label)`: optional
    callback fired ONCE per candidate as it's scored — lets the runner
    persist rows incrementally so the UI shows progress mid-run.

    Returns ALL scored candidates (best to worst by score), tagged with
    (strategy, score, generation, parent_label) for the runner to
    persist with lineage.

    Failure isolation: a scorer exception logs a warning and the
    candidate is assigned score=-inf. The GA carries on so one broken
    candidate doesn't kill the run.
    """
    if config.pop_size < 2:
        raise ValueError("pop_size must be >= 2")
    if config.elite < 1 or config.elite >= config.pop_size:
        raise ValueError(f"elite must be in [1, {config.pop_size - 1}], got {config.elite}")
    if config.generations < 1:
        raise ValueError("generations must be >= 1")

    all_scored: List[_ScoredCandidate] = []
    sample_counter = 0          # unique idx fed to space.sample/mutate

    # ── Generation 0: random ─────────────────────────────────────────
    logger.info("GA gen 0 — sampling %d random candidates", config.pop_size)
    gen0: List[_ScoredCandidate] = []
    for _ in range(config.pop_size):
        try:
            s = space.sample(sample_counter)
            sample_counter += 1
        except Exception as e:
            logger.warning("GA sample failed: %s", e)
            continue
        score = _score_safely(s, score_fn)
        sc = _ScoredCandidate(strategy=s, score=score, generation=0, parent_label=None)
        gen0.append(sc)
        if on_candidate is not None:
            on_candidate(s, score, 0, None)

    all_scored.extend(gen0)
    current_gen = gen0

    # ── Generations 1..G-1: elitism + mutation ───────────────────────
    for gen_idx in range(1, config.generations):
        survivors = sorted(current_gen, key=lambda c: c.score, reverse=True)[:config.elite]
        if not survivors:
            logger.warning("GA gen %d — no survivors; stopping early", gen_idx)
            break

        logger.info(
            "GA gen %d — %d survivors (best score %.3f), spawning children",
            gen_idx, len(survivors), survivors[0].score,
        )

        next_gen: List[_ScoredCandidate] = []

        # Each survivor breeds `children_per_elite` mutated children
        for parent_sc in survivors:
            for _ in range(config.children_per_elite):
                try:
                    child = space.mutate(parent_sc.strategy, sample_counter)
                    sample_counter += 1
                except Exception as e:
                    logger.warning("GA mutate failed: %s", e)
                    continue
                score = _score_safely(child, score_fn)
                sc = _ScoredCandidate(
                    strategy=child, score=score, generation=gen_idx,
                    parent_label=parent_sc.strategy.name,
                )
                next_gen.append(sc)
                if on_candidate is not None:
                    on_candidate(child, score, gen_idx, parent_sc.strategy.name)

        # Diversity: fill remaining slots with fresh random draws so the
        # population doesn't collapse into a single neighborhood.
        slots_left = max(0, config.pop_size - len(next_gen))
        for _ in range(slots_left):
            try:
                s = space.sample(sample_counter)
                sample_counter += 1
            except Exception as e:
                logger.warning("GA sample failed: %s", e)
                continue
            score = _score_safely(s, score_fn)
            sc = _ScoredCandidate(strategy=s, score=score, generation=gen_idx, parent_label=None)
            next_gen.append(sc)
            if on_candidate is not None:
                on_candidate(s, score, gen_idx, None)

        all_scored.extend(next_gen)
        current_gen = next_gen

    # Sort descending by score for caller convenience
    all_scored.sort(key=lambda c: c.score, reverse=True)
    return [
        (sc.strategy, sc.score, sc.generation, sc.parent_label)
        for sc in all_scored
    ]


def _score_safely(strategy: Strategy, score_fn: Callable[[Strategy], float]) -> float:
    """Wrap the scorer with belt-and-braces error handling."""
    try:
        return float(score_fn(strategy))
    except Exception as e:
        logger.warning("GA scorer exception for %s: %s", strategy.name, e)
        return float("-inf")

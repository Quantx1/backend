"""RL exit agent — COMPLETE implementation, ENABLED by default — PR-MODELS.

═══════════════════════════════════════════════════════════════════════
NARROW SCOPE — USER EXPLICITLY APPROVED 2026-05-25
═══════════════════════════════════════════════════════════════════════

This is a TIGHT exit-timing-only Q-learning agent. It is structurally
NOT the same thing as the entry-side full-decision RL that was removed
in ``memory/project_rl_removed_2026_05_23.md``:

  Previous (removed): FinRL-X PPO+SAC+A2C ensemble for portfolio entry
                      decisions. Huge action space. Sharpe -0.16 in shadow.

  This (enabled):     Tabular Q-learning. 3 actions: HOLD/EXIT/TIGHTEN.
                      8 features. Operates ONLY on already-open positions.
                      Hard SL/target safety rails remain authoritative.
                      aaryansinha16 runs this exact pattern at ~90%
                      profitability on RL_EXIT actions.

User explicit OK 2026-05-25 — see memory note
``project_rl_exit_enabled_2026_05_25.md``.

═══════════════════════════════════════════════════════════════════════

State space (8 features):
  unrealized_pnl_pct, bars_held_norm, premium_momentum,
  premium_volatility, distance_to_sl, distance_to_tgt,
  trailing_active, peak_gain_pct

Action space: HOLD | EXIT | TIGHTEN

Reward function:
  EXIT  → realized P&L % at that bar (positive = win)
  HOLD  → 0 + small negative hold_penalty (opportunity cost of theta)
  TIGHTEN → 0 (immediate); SL ratchets up, affecting future state

Training: Q-learning with bootstrapped value estimate
  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

Persistence: JSON-serialised Q-table → ``artifacts/rl/q_table.json``
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Enabled by default per user direction 2026-05-25
# ═══════════════════════════════════════════════════════════════════

ENABLE_RL_EXIT: bool = os.getenv("ENABLE_RL_EXIT", "true").lower() == "true"

RL_MODEL_DIR = Path(os.getenv("RL_MODELS_DIR", "artifacts/rl"))
RL_QTABLE_PATH = RL_MODEL_DIR / "q_table.json"


STATE_FEATURES = (
    "unrealized_pnl_pct",
    "bars_held_norm",
    "premium_momentum",
    "premium_volatility",
    "distance_to_sl",
    "distance_to_tgt",
    "trailing_active",
    "peak_gain_pct",
)

ACTIONS = ("HOLD", "EXIT", "TIGHTEN")
N_ACTIONS = len(ACTIONS)


_BINS = {
    "unrealized_pnl_pct": (-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30, 0.50),
    "bars_held_norm": (0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0),
    "premium_momentum": (-0.05, -0.02, -0.005, 0.0, 0.005, 0.02, 0.05),
    "premium_volatility": (0.005, 0.01, 0.02, 0.03, 0.05, 0.08),
    "distance_to_sl": (0.0, 0.05, 0.10, 0.20, 0.35, 0.50),
    "distance_to_tgt": (0.0, 0.10, 0.25, 0.40, 0.60, 0.80),
    "trailing_active": (0.5,),
    "peak_gain_pct": (0.0, 0.05, 0.10, 0.20, 0.35, 0.50),
}


@dataclass
class RLExitState:
    unrealized_pnl_pct: float
    bars_held_norm: float
    premium_momentum: float
    premium_volatility: float
    distance_to_sl: float
    distance_to_tgt: float
    trailing_active: float
    peak_gain_pct: float


def discretize_state(state: RLExitState) -> Tuple[int, ...]:
    indices = []
    for feat in STATE_FEATURES:
        val = getattr(state, feat, 0.0)
        bins = _BINS[feat]
        idx = 0
        for threshold in bins:
            if val < threshold:
                break
            idx += 1
        indices.append(idx)
    return tuple(indices)


def compute_rl_state(
    *,
    entry_price: float,
    current_price: float,
    bars_held: int,
    max_hold_bars: int,
    sl: float,
    target: float,
    trailing_active: bool,
    peak_price: float,
    price_history: List[float],
) -> RLExitState:
    import numpy as np

    if entry_price <= 0:
        return RLExitState(0, 0, 0, 0, 1, 1, 0, 0)

    unrealized = (current_price - entry_price) / entry_price
    held_norm = bars_held / max(max_hold_bars, 1)
    peak_gain = (peak_price - entry_price) / entry_price

    if len(price_history) >= 3:
        momentum = (price_history[-1] - price_history[-3]) / entry_price
    else:
        momentum = 0.0

    if len(price_history) >= 5:
        changes = np.diff(price_history[-5:]) / entry_price
        vol = float(np.std(changes))
    else:
        vol = 0.0

    dist_sl = (current_price - sl) / entry_price if sl > 0 else 1.0
    dist_tgt = (target - current_price) / entry_price if target > 0 else 1.0

    return RLExitState(
        unrealized_pnl_pct=unrealized,
        bars_held_norm=held_norm,
        premium_momentum=momentum,
        premium_volatility=vol,
        distance_to_sl=max(dist_sl, 0.0),
        distance_to_tgt=max(dist_tgt, 0.0),
        trailing_active=1.0 if trailing_active else 0.0,
        peak_gain_pct=max(peak_gain, 0.0),
    )


class RLExitAgent:
    """Tabular Q-learning exit agent. JSON-persistable Q-table.

    Hard safety: even when this agent says EXIT, the caller's regular
    SL/target/timeout exits still happen first. RL only ADDS an early-exit
    pathway — it never overrides hard stops.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        hold_penalty: float = -0.001,
    ):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.hold_penalty = hold_penalty
        self.q_table: Dict[tuple, List[float]] = {}
        self.is_loaded = False
        self.is_enabled = ENABLE_RL_EXIT
        self.training_episodes = 0

    def _get_q(self, state_key: tuple) -> List[float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.01, 0.0, 0.0]    # slight HOLD bias
        return self.q_table[state_key]

    def decide(self, state: RLExitState, *, explore: bool = False) -> str:
        if not self.is_enabled or not self.is_loaded:
            return "HOLD"
        import numpy as np
        key = discretize_state(state)
        if explore and np.random.random() < self.epsilon:
            action_idx = int(np.random.randint(N_ACTIONS))
        else:
            q_vals = self.q_table.get(key)
            if q_vals is None:
                return "HOLD"  # cold-start safety
            action_idx = int(np.argmax(q_vals))
        return ACTIONS[action_idx]

    def update(
        self,
        state: RLExitState,
        action: str,
        reward: float,
        next_state: Optional[RLExitState] = None,
        done: bool = False,
    ) -> None:
        """Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q(s', a') - Q(s, a)]"""
        state_key = discretize_state(state)
        action_idx = ACTIONS.index(action)
        q_vals = self._get_q(state_key)
        if done or next_state is None:
            target = reward
        else:
            next_key = discretize_state(next_state)
            next_q = self._get_q(next_key)
            target = reward + self.gamma * max(next_q)
        q_vals[action_idx] += self.lr * (target - q_vals[action_idx])
        self.q_table[state_key] = q_vals

    def fit_trajectory(
        self,
        *,
        trajectory: List[Dict[str, Any]],
        entry_price: float,
        sl_pct: float = 0.08,
        target_pct: float = 0.15,
        max_hold_bars: int = 40,
    ) -> None:
        """Train on one trade journey. Each intermediate bar trains a
        HOLD step; the final bar trains an EXIT step with realized PnL."""
        if len(trajectory) < 2:
            return

        sl = entry_price * (1 - sl_pct)
        target = entry_price * (1 + target_pct)
        peak = entry_price
        price_history: List[float] = [entry_price]

        for i, point in enumerate(trajectory):
            price = float(point.get("price") or 0)
            if price <= 0:
                continue
            price_history.append(price)
            if price > peak:
                peak = price

            bars_held = int(point.get("bars_held") or i + 1)
            state = compute_rl_state(
                entry_price=entry_price, current_price=price,
                bars_held=bars_held, max_hold_bars=max_hold_bars,
                sl=sl, target=target, trailing_active=False,
                peak_price=peak, price_history=price_history,
            )

            is_last = (i == len(trajectory) - 1)
            if is_last:
                realized_pnl = (price - entry_price) / entry_price
                self.update(state, "EXIT", realized_pnl, next_state=None, done=True)
            else:
                next_pt = trajectory[i + 1]
                next_price = float(next_pt.get("price") or 0)
                if next_price > 0:
                    next_state = compute_rl_state(
                        entry_price=entry_price, current_price=next_price,
                        bars_held=bars_held + 1, max_hold_bars=max_hold_bars,
                        sl=sl, target=target, trailing_active=False,
                        peak_price=max(peak, next_price),
                        price_history=price_history + [next_price],
                    )
                    self.update(state, "HOLD", self.hold_penalty,
                                next_state=next_state, done=False)
            self.training_episodes += 1

    # ── Persistence (JSON; tuples encoded as comma-separated strings) ──

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or RL_QTABLE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # Encode tuple keys as strings for JSON
        encoded_q = {
            ",".join(str(x) for x in key): [float(v) for v in vals]
            for key, vals in self.q_table.items()
        }
        with open(path, "w") as f:
            json.dump({
                "q_table": encoded_q,
                "training_episodes": self.training_episodes,
                "n_states": len(self.q_table),
                "action_space": list(ACTIONS),
                "state_features": list(STATE_FEATURES),
            }, f, indent=2)
        return path

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or RL_QTABLE_PATH
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.q_table = {
                tuple(int(x) for x in key_str.split(",")): list(vals)
                for key_str, vals in (data.get("q_table") or {}).items()
            }
            self.training_episodes = int(data.get("training_episodes") or 0)
            self.is_loaded = len(self.q_table) > 0
            if self.is_loaded:
                logger.info(
                    "RLExitAgent loaded: %d states, %d episodes",
                    len(self.q_table), self.training_episodes,
                )
            return self.is_loaded
        except Exception as exc:
            logger.warning("RLExitAgent load failed: %s", exc)
            return False

    def load_from_dict(self, q_table: Dict[tuple, Any]) -> None:
        """Legacy injection — direct dict insert (used by old tests)."""
        self.q_table = {k: list(v) for k, v in (q_table or {}).items()}
        self.is_loaded = len(self.q_table) > 0


_global_agent: Optional[RLExitAgent] = None


def get_rl_exit_agent() -> RLExitAgent:
    global _global_agent
    if _global_agent is None:
        _global_agent = RLExitAgent()
        try:
            _global_agent.load()
        except Exception:
            pass
    return _global_agent


def rl_exit_status() -> Dict[str, Any]:
    agent = get_rl_exit_agent()
    return {
        "enabled_flag": ENABLE_RL_EXIT,
        "is_loaded": agent.is_loaded,
        "scope": "narrow_exit_only (HOLD/EXIT/TIGHTEN on open positions)",
        "user_approval": "2026-05-25 explicit user OK — see project_rl_exit_enabled_2026_05_25.md",
        "memory_lock_note": "project_rl_removed_2026_05_23 was about entry-side FinRL-X. This is different scope.",
        "q_table_size": len(agent.q_table),
        "training_episodes": agent.training_episodes,
        "safety": "Hard SL + target stay authoritative. RL adds early-exit pathway only.",
    }

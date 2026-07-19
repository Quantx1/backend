"""Kite Connect tick collector — PR-DEPTH.

Real-time tick ingestion via Kite WebSocket → Supabase ``tick_data`` table.

Why this exists:
  Kite Connect doesn't expose historical tick data (unlike TrueData). To
  enable per-tick backtests, premium-confirmation gates, and micro-feature
  computation, we have to collect our own tick history starting from
  day 1 of deployment.

  Pattern adapted from aaryansinha16/AI-trader's scripts/collect_ticks.py.

Symbols collected (initial set, configurable via env):
  - NIFTY 50 / BANKNIFTY / FINNIFTY (underlier ticks)
  - ATM ±3 strikes × CE+PE = up to ~14 option contracts per index
  - Dynamic re-subscription when the underlier drifts > 100 points

Scaling: tick volume @ peak ~50K ticks/min across 20 symbols. Buffered
flush every 500 ticks or 5 seconds. Supabase free-tier sufficient for
a few users; for 100+ users we'd switch to a dedicated TimescaleDB pod.

Memory locks honoured: pure data ingestion, no LLM, no broker order
issuance. Read-only side of Kite Connect.
"""

from .collector import (
    TickCollector,
    TickCollectorConfig,
    TickCollectorStatus,
)

__all__ = [
    "TickCollector",
    "TickCollectorConfig",
    "TickCollectorStatus",
]

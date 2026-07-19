"""Pure dataclasses for the signal pipeline. No logic."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional


@dataclass
class EnsembleVoter:
    """One contributor to the ensemble confidence score.

    Per the no-fallbacks rule (locked 2026-04-19): every voter in this
    list MUST come from a real loaded model. There is no ``available``
    field and no graceful skip — if a required model can't load,
    ``SignalGenerator.__init__`` raises and the service refuses to start.

    ``score`` is normalized to 0..1. ``direction_agrees`` feeds the
    ``model_agreement`` count surfaced on the signal card.
    """
    name: str
    weight: float
    score: float
    direction_agrees: bool


@dataclass
class GeneratedSignal:
    """
    Generated trading signal.

    DB column mapping (legacy names retained for Supabase compatibility):
      catboost_score    → ML meta-labeler (RandomForest) breakout probability (0-1)
      tft_score         → TFT price-forecast bullish score (0-1)
      stockformer_score → Raw strategy confidence (0-100)
    """
    symbol: str
    exchange: str
    segment: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    target_3: Optional[float]
    risk_reward: float
    catboost_score: float      # DB: catboost_score → actually ML meta-labeler (RandomForest) (0-1)
    tft_score: float           # DB: tft_score → TFT forecast bullish score (0-1)
    stockformer_score: float   # DB: stockformer_score → raw strategy confidence (0-100)
    lgbm_score: float          # LightGBM BUY probability (0-1)
    model_agreement: int       # 1=strategy only, 2+=more models agree
    reasons: List[str]
    is_premium: bool
    strategy_name: str = ""    # Primary strategy that generated this signal
    tft_prediction: Optional[Dict] = None  # TFT quantile forecast (p10, p50, p90)
    lot_size: Optional[int] = None
    expiry_date: Optional[date] = None
    strategy_catalog_id: Optional[str] = None  # Links to strategy_catalog for marketplace deployments
    strike_price: Optional[float] = None
    option_type: Optional[str] = None
    # PR 4 — HMM/shadow-model columns (see PR 2 migration).
    regime_at_signal: Optional[str] = None    # 'bull' / 'sideways' / 'bear'
    regime_snapshot: Optional[Dict] = None    # full regime_info dict at signal time
    lgbm_buy_prob: Optional[float] = None     # LGBMGate SHADOW buy probability (0-1)

"""Supabase persistence for generated signals.

All Supabase writes for the signal pipeline live here. Public API:

    await save_signals(supabase, signals, signal_date=None, catalog_cache=None)
    await save_universe(supabase, candidates, trade_date, source, scan_type, run_id=None)
    await cache_candles(supabase, symbol, df)
    resolve_catalog_id(supabase, strategy_name, cache=None) -> Optional[str]

``catalog_cache`` is a mutable dict the caller (typically
``SignalGenerator``) owns so the catalog→UUID lookup is hit once per
process, not once per signal.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .types import GeneratedSignal

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Universe + candles
# ────────────────────────────────────────────────────────────────────


async def save_universe(
    supabase,
    candidates: List[str],
    trade_date: date,
    source: str,
    scan_type: str,
    run_id: Optional[str] = None,
) -> None:
    """Persist EOD candidate universe for transparency."""
    try:
        rows = []
        for symbol in candidates:
            rows.append({
                "trade_date": trade_date.isoformat(),
                "symbol": symbol,
                "source": source,
                "scan_type": scan_type,
                "run_id": run_id,
            })
        if rows:
            supabase.table("daily_universe").upsert(
                rows,
                on_conflict="trade_date,symbol",
            ).execute()
    except Exception as e:
        logger.warning(f"Failed to save daily universe: {e}")


async def cache_candles(supabase, symbol: str, df: pd.DataFrame) -> None:
    try:
        rows = []
        for idx, row in df.tail(200).iterrows():
            rows.append({
                "stock_symbol": symbol,
                "exchange": "NSE",
                "interval": "1d",
                "timestamp": idx.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "source": "kite",
            })
        supabase.table("candles").upsert(
            rows, on_conflict="stock_symbol,interval,timestamp"
        ).execute()
    except Exception as e:
        logger.debug(f"Failed to cache candles for {symbol}: {e}")


# ────────────────────────────────────────────────────────────────────
# Strategy catalog lookup
# ────────────────────────────────────────────────────────────────────


def resolve_catalog_id(
    supabase,
    strategy_name: str,
    cache: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Map ``strategy_name`` (e.g. 'Consolidation_Breakout') →
    ``strategy_catalog.id`` UUID.

    If ``cache`` is provided, it is treated as a mutable lookup table:
    populated on first call, hit on subsequent calls. Pass the same
    dict from the calling class to avoid repeated full-table scans.
    """
    if cache is None:
        cache = {}

    if not cache:
        try:
            result = supabase.table("strategy_catalog").select(
                "id, slug, name"
            ).execute()
            for row in (result.data or []):
                cache[row["slug"]] = row["id"]
                normalized = row["name"].lower().replace(" ", "_").replace("-", "_")
                cache[normalized] = row["id"]
        except Exception as e:
            logger.warning(f"Failed to load strategy catalog map: {e}")

    normalized_name = strategy_name.lower().replace(" ", "_").replace("-", "_")
    return cache.get(normalized_name)


# ────────────────────────────────────────────────────────────────────
# Signal rows
# ────────────────────────────────────────────────────────────────────


def _sanitize(v):
    """Recursively convert numpy scalars/arrays into JSON-safe types."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k2: _sanitize(v2) for k2, v2 in v.items()}
    if isinstance(v, list):
        return [_sanitize(i) for i in v]
    return v


def _safe_float(v, default=0.0):
    if v is None:
        return default
    fv = float(v)
    if fv == float("inf") or fv == float("-inf"):
        return default
    return fv


async def save_signals(
    supabase,
    signals: List[GeneratedSignal],
    signal_date: Optional[date] = None,
    catalog_cache: Optional[Dict[str, str]] = None,
) -> None:
    """Save generated signals to database.

    ``catalog_cache`` is forwarded to ``resolve_catalog_id`` for any
    signal that doesn't already carry a ``strategy_catalog_id``.
    """
    today = (signal_date or date.today()).isoformat()

    for signal in signals:
        try:
            # Check for existing signal (prevent re-run duplicates)
            existing = supabase.table("signals").select("id").eq(
                "date", today
            ).eq("symbol", signal.symbol).eq("direction", signal.direction).eq(
                "status", "active"
            ).execute()
            if existing.data:
                continue

            # Build TFT prediction payload (strip internal keys for DB)
            tft_pred_payload = {}
            if signal.tft_prediction:
                tft_pred_payload = {
                    k: v for k, v in signal.tft_prediction.items()
                    if k in ("p10", "p50", "p90", "direction", "horizon",
                             "current_close", "predicted_close")
                }
            tft_pred_payload = _sanitize(tft_pred_payload)

            data = {
                "symbol": signal.symbol,
                "exchange": signal.exchange,
                "segment": signal.segment,
                "direction": signal.direction,
                "confidence": _safe_float(signal.confidence),
                "entry_price": _safe_float(signal.entry_price),
                "stop_loss": _safe_float(signal.stop_loss),
                "target_1": _safe_float(signal.target_1),
                "target_2": _safe_float(signal.target_2),
                "target_3": _safe_float(signal.target_3),
                "risk_reward": _safe_float(signal.risk_reward),
                "catboost_score": float(signal.catboost_score) if signal.catboost_score is not None else None,
                "tft_score": float(signal.tft_score) if signal.tft_score is not None else None,
                "stockformer_score": float(signal.stockformer_score) if signal.stockformer_score is not None else None,
                "model_agreement": int(signal.model_agreement) if signal.model_agreement is not None else 1,
                "reasons": signal.reasons or [],
                "is_premium": bool(signal.is_premium) if signal.is_premium is not None else False,
                "lot_size": int(signal.lot_size) if signal.lot_size is not None else 1,
                "strategy_names": [signal.strategy_name],
                "tft_prediction": tft_pred_payload,
                "date": today,
                "status": "active",
                "generated_at": datetime.utcnow().isoformat(),
                # PR 4 — HMM + shadow-model columns (see PR 2 migration).
                "regime_at_signal": signal.regime_at_signal,
                "lgbm_buy_prob": (
                    float(signal.lgbm_buy_prob)
                    if signal.lgbm_buy_prob is not None else None
                ),
                "tft_p10": (
                    _safe_float(tft_pred_payload.get("p10")[-1])
                    if isinstance(tft_pred_payload.get("p10"), list) and tft_pred_payload["p10"]
                    else None
                ),
                "tft_p50": (
                    _safe_float(tft_pred_payload.get("p50")[-1])
                    if isinstance(tft_pred_payload.get("p50"), list) and tft_pred_payload["p50"]
                    else None
                ),
                "tft_p90": (
                    _safe_float(tft_pred_payload.get("p90")[-1])
                    if isinstance(tft_pred_payload.get("p90"), list) and tft_pred_payload["p90"]
                    else None
                ),
            }

            # Tag with marketplace strategy catalog ID
            catalog_id = signal.strategy_catalog_id or resolve_catalog_id(
                supabase, signal.strategy_name, cache=catalog_cache,
            )
            if catalog_id:
                data["strategy_catalog_id"] = catalog_id

            supabase.table("signals").insert(data).execute()

        except Exception as e:
            logger.error(f"Failed to save signal for {signal.symbol}: {e}")

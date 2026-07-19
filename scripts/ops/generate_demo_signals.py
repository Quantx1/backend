#!/usr/bin/env python
"""
Demo signal generator — uses HMM + TFT + LGBM (no Qlib needed) to produce
real signals on a small NSE universe. Useful for local verification
before the full Qlib data provider is ingested.

Run: python scripts/ops/generate_demo_signals.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Project root on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("demo_signals")


# Small but liquid demo universe — top NSE names that always have data.
DEMO_UNIVERSE = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
    "BHARTIARTL", "SBIN", "ITC", "LT", "KOTAKBANK",
]


def main():
    print("═" * 70)
    print("Quant X — Demo Signal Generator")
    print("Uses HMM + TFT + LGBM (skip Qlib for this demo)")
    print("═" * 70)

    # ── 1. Fetch recent OHLCV ─────────────────────────────────────────
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=200)
    tickers = [f"{s}.NS" for s in DEMO_UNIVERSE]
    print(f"\n[1/4] Fetching {len(tickers)} NSE symbols from yfinance ({start}..{end})...")
    px = yf.download(
        tickers,
        start=str(start),
        end=str(end),
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    if px is None or px.empty:
        print("  FAIL: yfinance returned empty")
        return 1
    print(f"  OK — {len(px)} bars × {len(tickers)} symbols")

    # ── 2. Load PROD models from B2 ───────────────────────────────────
    print("\n[2/4] Loading PROD models from B2 + Supabase...")
    from backend.core.database import get_supabase_admin
    sb = get_supabase_admin()

    from backend.ai.registry.model_registry import get_registry
    reg = get_registry()

    # HMM regime detector — download from B2 via registry
    from ml.regime_detector import MarketRegimeDetector, compute_regime_features
    hmm = MarketRegimeDetector()
    hmm_dir = reg.resolve("regime_hmm")
    hmm_path = next(hmm_dir.glob("*.pkl"))
    hmm.load(str(hmm_path))
    print(f"  ✓ HMM regime detector loaded")

    # LGBM gate (from disk fallback — no B2 PROD row for it yet)
    from backend.ai.model_registry import LGBMGate
    lgbm_path = Path(ROOT) / "artifacts" / "models" / "lgbm_signal_gate.txt"
    lgbm_gate = None
    if lgbm_path.exists():
        try:
            lgbm_gate = LGBMGate.load(str(lgbm_path), meta_path=str(lgbm_path).replace(".txt", ".meta.json"))
            print(f"  ✓ LGBM gate loaded")
        except Exception as exc:
            print(f"  ⚠ LGBM gate skipped: {exc}")

    # ── 3. Compute regime + per-stock features ────────────────────────
    print("\n[3/4] Computing current market regime + per-stock features...")
    nifty = yf.download("^NSEI", start=str(start), end=str(end), progress=False, auto_adjust=True)
    vix = yf.download("^INDIAVIX", start=str(start), end=str(end), progress=False, auto_adjust=True)
    # Flatten columns if multi-index
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [c[0] for c in nifty.columns]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]
    nifty.columns = [c.lower() for c in nifty.columns]
    vix.columns = [c.lower() for c in vix.columns]

    try:
        regime_feats = compute_regime_features(nifty, vix.reindex(nifty.index).ffill())
        regime_pred = hmm.predict(regime_feats.tail(20))
        regime_id = int(regime_pred[-1]) if len(regime_pred) else 1
        regime_map = hmm.regime_labels
        regime_name = regime_map.get(regime_id, "sideways")
        print(f"  ✓ Current market regime: {regime_name.upper()}")
    except Exception as exc:
        regime_name = "sideways"
        print(f"  ⚠ regime fallback to sideways: {exc}")

    # ── 4. Per-stock signal generation ────────────────────────────────
    print(f"\n[4/4] Generating signals for {len(DEMO_UNIVERSE)} stocks...")
    print()

    signals = []
    for sym in DEMO_UNIVERSE:
        ticker = f"{sym}.NS"
        try:
            bar = px[ticker].dropna()
        except KeyError:
            continue
        if len(bar) < 50:
            continue

        last_close = float(bar["Close"].iloc[-1])
        prev_close = float(bar["Close"].iloc[-2])
        change_pct = (last_close - prev_close) / prev_close * 100

        # Simple technical features (we skip Qlib's 158 alpha factors here)
        ret_5d = (bar["Close"].iloc[-1] / bar["Close"].iloc[-5] - 1) * 100
        ret_20d = (bar["Close"].iloc[-1] / bar["Close"].iloc[-20] - 1) * 100
        vol_20d = bar["Close"].pct_change().tail(20).std() * np.sqrt(252) * 100
        rsi_proxy = 50 + ret_5d * 2  # crude RSI proxy for demo

        # Naive signal: positive momentum + low vol + bull regime
        score = 0.0
        if ret_20d > 0:
            score += 0.3
        if ret_5d > 0:
            score += 0.2
        if vol_20d < 30:
            score += 0.1
        if regime_name == "bull":
            score += 0.3
        elif regime_name == "sideways":
            score += 0.1

        # Regime bear gate
        if regime_name == "bear" and score > 0.3:
            score *= 0.4  # halve confidence in bear

        action = "BUY" if score >= 0.5 else ("HOLD" if score >= 0.3 else "SKIP")
        signals.append({
            "symbol": sym,
            "last": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "ret_5d_pct": round(float(ret_5d), 2),
            "ret_20d_pct": round(float(ret_20d), 2),
            "vol_20d_ann_pct": round(float(vol_20d), 2),
            "rsi_proxy": round(float(rsi_proxy), 1),
            "regime": regime_name,
            "score": round(score, 2),
            "action": action,
        })

    signals.sort(key=lambda x: x["score"], reverse=True)

    # ── 5. Display ────────────────────────────────────────────────────
    print(f"{'Symbol':<12} {'Last':>8} {'Δ%':>6} {'5d%':>6} {'20d%':>6} {'Vol':>5} {'Score':>5} {'Action':<6}")
    print("─" * 70)
    for s in signals:
        print(
            f"{s['symbol']:<12} {s['last']:>8.2f} {s['change_pct']:>+6.2f} "
            f"{s['ret_5d_pct']:>+6.2f} {s['ret_20d_pct']:>+6.2f} "
            f"{s['vol_20d_ann_pct']:>5.1f} {s['score']:>5.2f} {s['action']:<6}"
        )

    print()
    buy = [s for s in signals if s["action"] == "BUY"]
    print(f"═══ Summary ═══")
    print(f"  Universe: {len(DEMO_UNIVERSE)} liquid NSE stocks")
    print(f"  Market regime: {regime_name.upper()}")
    print(f"  BUY signals: {len(buy)}")
    print(f"  HOLD: {len([s for s in signals if s['action'] == 'HOLD'])}")
    print(f"  SKIP: {len([s for s in signals if s['action'] == 'SKIP'])}")
    if buy:
        print(f"\n  Top BUY: {', '.join([s['symbol'] for s in buy[:5]])}")
    print()
    print("⚠ This is a DEMO signal generator. Production SignalGenerator")
    print("  uses TFT (68% directional accuracy) + Qlib (cross-sectional)")
    print("  + LGBM gate + FinBERT sentiment, not this simple score.")
    print("  Production version is in backend/services/signal_generator.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

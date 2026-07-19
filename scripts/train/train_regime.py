#!/usr/bin/env python3
"""
================================================================================
TRAIN HMM MARKET REGIME DETECTOR
================================================================================
Downloads 10 years of Nifty 50 + India VIX data, computes regime features,
trains a 3-state Gaussian HMM, and saves to artifacts/models/regime_hmm.pkl.

Usage:
    python scripts/train/train_regime.py
================================================================================
"""

import sys
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from ml.regime_detector import MarketRegimeDetector, compute_regime_features


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from newer yfinance to lowercase strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def download_nifty(period: str = "10y") -> pd.DataFrame:
    """Download Nifty 50 daily data."""
    print(f"Downloading Nifty 50 data ({period})...")
    df = yf.download("^NSEI", period=period, interval="1d", progress=False)
    if df.empty:
        raise RuntimeError("Failed to download Nifty 50 data")
    df = _flatten_columns(df)
    print(f"  Got {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def download_vix(period: str = "10y") -> pd.DataFrame | None:
    """Download India VIX data. Returns None if unavailable."""
    print(f"Downloading India VIX data ({period})...")
    try:
        df = yf.download("^INDIAVIX", period=period, interval="1d", progress=False)
        if df.empty or len(df) < 100:
            print("  India VIX data insufficient, will use realized vol proxy")
            return None
        df = _flatten_columns(df)
        print(f"  Got {len(df)} VIX observations")
        return df
    except Exception as e:
        print(f"  VIX download failed: {e}. Using realized vol proxy.")
        return None


def print_regime_stats(detector: MarketRegimeDetector, features: pd.DataFrame):
    """Print regime distribution and transition matrix."""
    if not detector.is_trained:
        print("Model not trained!")
        return

    X = features.dropna().values.astype(np.float64)
    X_scaled = detector.scaler.transform(X)
    states = detector.model.predict(X_scaled)

    print("\n" + "=" * 60)
    print("REGIME DISTRIBUTION")
    print("=" * 60)
    total = len(states)
    for sid, name in detector.REGIMES.items():
        count = int((states == sid).sum())
        pct = count / total * 100
        print(f"  {name:>10s} (state {sid}): {count:5d} days ({pct:5.1f}%)")

    print(f"\n  Total: {total} trading days")

    print("\n" + "=" * 60)
    print("TRANSITION MATRIX")
    print("=" * 60)
    transmat = detector.model.transmat_
    header = "".join(f"{detector.REGIMES[i]:>12s}" for i in range(transmat.shape[1]))
    print(f"  {'From \\ To':>12s}{header}")
    for i in range(transmat.shape[0]):
        row = "".join(f"{transmat[i, j]:12.4f}" for j in range(transmat.shape[1]))
        print(f"  {detector.REGIMES[i]:>12s}{row}")

    # Strategy weights per regime
    print("\n" + "=" * 60)
    print("STRATEGY WEIGHTS BY REGIME")
    print("=" * 60)
    for sid, name in detector.REGIMES.items():
        weights = detector.get_strategy_weights(sid)
        print(f"\n  {name.upper()} regime:")
        for strat, w in weights.items():
            status = "SKIP" if w == 0.0 else f"{w:.1f}x"
            print(f"    {strat:30s} -> {status}")

    # Current regime (last observation)
    current_state = int(states[-1])
    posteriors = detector.model.predict_proba(X_scaled)
    last_probs = posteriors[-1]
    print("\n" + "=" * 60)
    print("CURRENT REGIME (last observation)")
    print("=" * 60)
    print(f"  Regime: {detector.REGIMES[current_state].upper()}")
    print(f"  Confidence: {last_probs[current_state]:.2%}")
    for sid, name in detector.REGIMES.items():
        print(f"    P({name}): {last_probs[sid]:.4f}")


def main():
    # Download data
    nifty_df = download_nifty("10y")
    vix_df = download_vix("10y")

    # Compute features
    print("\nComputing regime features...")
    features = compute_regime_features(nifty_df, vix_df)
    features_clean = features.dropna()
    print(f"  {len(features_clean)} valid observations after NaN removal")

    # Train
    print("\nTraining 3-state Gaussian HMM...")
    detector = MarketRegimeDetector()
    detector.train(features)

    # Stats
    print_regime_stats(detector, features_clean)

    # Save
    model_dir = ROOT / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "regime_hmm.pkl"
    detector.save(str(save_path))
    print(f"\nModel saved to {save_path}")
    print(f"File size: {save_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

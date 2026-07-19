"""
Train LightGBM 3-class Signal Classifier (BUY / SELL / HOLD).

Downloads 10y OHLCV for ~500+ liquid NSE stocks, computes 15 technical features
via feature_engineering.compute_features() + split_feature_sets(), labels bars
by forward 10-bar return, and trains a LightGBM classifier with 5-fold
TimeSeriesSplit cross-validation.

Usage:
    python scripts/train/train_lgbm.py

Output:
    artifacts/models/lgbm_signal_gate.txt  (native LightGBM format)
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — allow importing from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

from backend.ai.feature_engineering import (
    compute_features,
    build_feature_row,
    split_feature_sets,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "models", "lgbm_signal_gate.txt")
SYMBOLS_PATH = os.path.join(PROJECT_ROOT, "data", "nse_all_symbols.json")
UNIVERSE_PATH = os.path.join(PROJECT_ROOT, "data", "full_backtest_universe.txt")

MIN_BARS = 500        # Minimum trading days (~2 years)
MIN_AVG_VOL = 200_000  # Average daily volume filter (strong liquidity)
BATCH_SIZE = 50        # yfinance batch size

# Labelling
FORWARD_BARS = 10
BUY_THRESHOLD = 0.03   # +3%
SELL_THRESHOLD = -0.03  # -3%

# LightGBM hyper-parameters
LGBM_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# The 15 features used by LGBMGate (matches split_feature_sets xgb_keys)
FEATURE_ORDER = [
    "close", "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_percent",
    "ema_20", "ema_50", "atr_14",
    "volume_ratio", "obv", "vwap_diff",
    "body_pct", "wick_pct",
]

# Label mapping
LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_symbols() -> list:
    """Load NSE symbols from JSON or universe text file."""
    if os.path.exists(SYMBOLS_PATH):
        with open(SYMBOLS_PATH) as f:
            data = json.load(f)
        symbols = data["symbols"]
        print(f"Loaded {len(symbols)} NSE symbols from {SYMBOLS_PATH}")
        return symbols

    if os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH) as f:
            symbols = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(symbols)} symbols from {UNIVERSE_PATH}")
        return symbols

    # Hardcoded top-200 fallback
    fallback = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL",
        "SBIN", "KOTAKBANK", "LT", "AXISBANK", "TATASTEEL", "JSWSTEEL",
        "HINDALCO", "VEDL", "ABB", "SIEMENS", "HAL", "BEL", "IRCTC",
        "TRENT", "POLYCAB", "PERSISTENT", "DIXON", "TATAELXSI", "ASTRAL",
        "COFORGE", "LALPATHLAB", "MUTHOOTFIN", "INDHOTEL", "TATAMOTORS",
        "MARUTI", "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
        "TITAN", "NESTLEIND", "ASIANPAINT", "ULTRACEMCO", "GRASIM",
        "ADANIENT", "ADANIPORTS", "WIPRO", "TECHM", "HCLTECH",
        "POWERGRID", "NTPC", "ONGC", "BPCL", "COALINDIA",
        "ITC", "HINDUNILVR", "BRITANNIA", "DABUR", "MARICO",
        "PIDILITIND", "BERGEPAINT", "GODREJCP", "COLPAL", "HAVELLS",
        "VOLTAS", "CROMPTON", "WHIRLPOOL", "EICHERMOT", "HEROMOTOCO",
        "BAJAJ-AUTO", "M&M", "TATAPOWER", "TORNTPOWER", "INDUSINDBK",
        "BANKBARODA", "PNB", "CANBK", "FEDERALBNK", "IDFCFIRSTB",
        "LICHSGFIN", "CHOLAFIN", "SHRIRAMFIN", "MFSL", "SBICARD",
        "ICICIPRULI", "ICICIGI", "RECLTD", "PFC", "IRFC",
        "DLF", "GODREJPROP", "PRESTIGE", "OBEROIRLTY", "SOBHA",
        "SRF", "PIIND", "NAVINFLUOR", "DEEPAKNTR", "ATUL",
        "LUPIN", "AUROPHARMA", "BIOCON", "GLENMARK", "ALKEM",
        "JUBLFOOD", "TATACONSUM", "VBL", "UBL", "MCDOWELL-N",
        "NAUKRI", "ZOMATO", "PAYTM", "POLICYBZR", "DMART",
        "LTIM", "LTTS", "MPHASIS", "KPITTECH", "HAPPSTMNDS",
        "MRF", "PAGEIND", "SHREECEM", "AMBUJACEM", "DALBHARAT",
        "SAIL", "NMDC", "NATIONALUM", "JINDALSTEL", "TATACHEM",
        "INDIGO", "CONCOR", "BHEL", "GAIL", "NHPC",
        "SJVN", "RVNL", "HUDCO", "INDUSTOWER", "IDEA",
        "ACC", "RAMCOCEM", "STARHEALTH", "MAXHEALTH", "MEDANTA",
        "CDSL", "CAMS", "ANGELONE", "MOTILALOFS", "MANAPPURAM",
        "RADICO", "SKFINDIA", "TIMKEN", "SCHAEFFLER", "GRINDWELL",
        "HONAUT", "LINDEINDIA", "RELAXO", "SUPREMEIND", "RATNAMANI",
        "KAYNES", "SYRMA", "DATAPATTNS", "CLEAN", "FIVESTAR",
        "CANFINHOME", "RBLBANK", "AUBANK", "BANDHANBNK", "SUNDARMFIN",
        "AFFLE", "MAPMYINDIA", "LATENTVIEW", "INTELLECT", "NIITLTD",
        "ZYDUSLIFE", "GLAXO", "PFIZER", "SANOFI", "NATCOPHARM",
        "NYKAA", "CAMPUS", "BIKAJI", "SULA", "TARSONS",
    ]
    print(f"Using hardcoded fallback of {len(fallback)} symbols")
    return fallback


def download_batch(tickers: list) -> dict:
    """Download OHLCV for a batch of tickers."""
    ticker_str = " ".join(tickers)
    try:
        raw = yf.download(
            ticker_str, period="10y", interval="1d",
            progress=False, group_by="ticker", threads=True,
        )
        if raw.empty:
            return {}
    except Exception:
        return {}

    results = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy()

            if df.empty or len(df) < MIN_BARS:
                continue

            df = df.dropna(subset=["Close"])
            if len(df) < MIN_BARS:
                continue

            # Normalize columns
            df.columns = [
                c.lower() if isinstance(c, str) else c[0].lower()
                for c in df.columns
            ]

            # Filter by volume
            if "volume" in df.columns:
                avg_vol = df["volume"].mean()
                if avg_vol < MIN_AVG_VOL:
                    continue

            results[ticker] = df
        except Exception:
            continue
    return results


def build_dataset(stock_dfs: dict) -> tuple:
    """
    Build feature matrix X and label vector y from all downloaded stocks.
    Label: 1=BUY (fwd return > +3%), 2=SELL (fwd return < -3%), 0=HOLD.
    """
    all_X = []
    all_y = []
    skipped = 0

    for sym, df in stock_dfs.items():
        try:
            featured = compute_features(df)

            # Forward return labelling
            featured["fwd_return"] = (
                featured["close"].shift(-FORWARD_BARS) / featured["close"] - 1
            )

            # Drop rows where features or label are NaN
            featured = featured.dropna(subset=FEATURE_ORDER + ["fwd_return"])
            if len(featured) < 50:
                skipped += 1
                continue

            # Extract the 15 features
            X_stock = featured[FEATURE_ORDER].values

            # Labels
            fwd = featured["fwd_return"].values
            y_stock = np.zeros(len(fwd), dtype=int)
            y_stock[fwd > BUY_THRESHOLD] = 1   # BUY
            y_stock[fwd < SELL_THRESHOLD] = 2   # SELL
            # else 0 = HOLD

            all_X.append(X_stock)
            all_y.append(y_stock)
        except Exception as e:
            skipped += 1
            continue

    if skipped:
        print(f"  Skipped {skipped} stocks (insufficient data or errors)")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LightGBM Signal Classifier Training (BUY / SELL / HOLD)")
    print("=" * 60)
    print()

    # 1. Load symbols
    all_symbols = load_symbols()
    tickers = [f"{sym}.NS" for sym in all_symbols]

    # 2. Download OHLCV in batches
    stock_dfs = {}
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    t_download = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(tickers))
        batch = tickers[batch_start:batch_end]

        results = download_batch(batch)
        for ticker, df in results.items():
            sym = ticker.replace(".NS", "")
            stock_dfs[sym] = df

        elapsed = time.time() - t_download
        print(
            f"  Batch {batch_idx + 1}/{total_batches} | "
            f"Downloaded: {len(stock_dfs)} qualifying stocks | "
            f"Elapsed: {elapsed:.0f}s"
        )

    download_time = time.time() - t_download
    print(f"\nDownload complete: {len(stock_dfs)} stocks in {download_time:.0f}s")

    if len(stock_dfs) < 30:
        print("ERROR: Too few stocks downloaded. Check network connection.")
        sys.exit(1)

    # 3. Build dataset
    print(f"\nBuilding feature matrix from {len(stock_dfs)} stocks...")
    t0 = time.time()
    X, y = build_dataset(stock_dfs)
    build_time = time.time() - t0
    print(f"Dataset: {X.shape[0]:,} samples x {X.shape[1]} features ({build_time:.1f}s)")

    # Class distribution
    for cls_id, cls_name in LABEL_MAP.items():
        count = (y == cls_id).sum()
        pct = count / len(y) * 100
        print(f"  {cls_name}: {count:,} ({pct:.1f}%)")

    # Replace any inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. 5-fold TimeSeriesSplit CV
    print(f"\nTraining LightGBM with 5-fold TimeSeriesSplit CV...")
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],  # suppress per-iter logs
        )

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        print(f"  Fold {fold + 1}: accuracy = {acc:.4f} ({len(X_val):,} val samples)")

    avg_acc = np.mean(fold_accuracies)
    print(f"\n  Average CV accuracy: {avg_acc:.4f}")

    # 5. Train final model on ALL data
    print(f"\nTraining final model on all {X.shape[0]:,} samples...")
    t0 = time.time()
    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(X, y)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # Classification report on training data (sanity check)
    y_train_pred = final_model.predict(X)
    print("\nFull-data classification report (sanity check):")
    print(classification_report(
        y, y_train_pred,
        target_names=["HOLD", "BUY", "SELL"],
    ))

    # Feature importance
    importances = list(zip(FEATURE_ORDER, final_model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    print("Feature importances (split-based):")
    for name, imp in importances:
        print(f"  {name:>14s}: {imp}")

    # 6. Save in native LightGBM format
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    final_model.booster_.save_model(MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH} ({file_size / 1024:.0f} KB)")
    print(f"Format: native LightGBM text (load with lgb.Booster)")
    print(f"Training stocks: {len(stock_dfs)}")
    print(f"CV accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()

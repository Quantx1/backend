"""
Train Temporal Fusion Transformer (TFT) for multi-horizon price forecasting.

Downloads 10y OHLCV for top 100 liquid NSE stocks, computes technical features
via feature_engineering.compute_features(), and trains a small TFT model with
quantile outputs (10th, 50th, 90th percentile) for next 5-bar close prediction.

Usage:
    python scripts/train/train_tft.py

Output:
    artifacts/models/tft_model.ckpt   (PyTorch Lightning checkpoint)
    artifacts/models/tft_config.json  (dataset params for TFTPredictor inference)
"""

import json
import os
import sys
import time
import warnings

# ---------------------------------------------------------------------------
# Path setup — allow importing from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print("  pip install torch")
    sys.exit(1)

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
except ImportError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    except ImportError:
        print("ERROR: pytorch-lightning is not installed.")
        print("  pip install pytorch-lightning>=2.1.3")
        sys.exit(1)

try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
except ImportError:
    print("ERROR: pytorch-forecasting is not installed.")
    print("  pip install pytorch-forecasting>=1.0.0")
    sys.exit(1)

import numpy as np
import pandas as pd
import yfinance as yf

from backend.ai.feature_engineering import compute_features

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
CKPT_PATH = os.path.join(MODEL_DIR, "tft_model.ckpt")
CONFIG_PATH = os.path.join(MODEL_DIR, "tft_config.json")

# Data
MAX_STOCKS = 100
MIN_BARS = 500          # Require ~2 years minimum history
MIN_AVG_VOL = 300_000   # Strong liquidity filter
BATCH_SIZE_DL = 25      # yfinance download batch size

# TFT architecture
MAX_ENCODER_LENGTH = 120  # 120-bar lookback (leverage deeper 10y history)
MAX_PREDICTION_LENGTH = 5  # 5-bar forecast horizon
HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 2
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 16

# Training
MAX_EPOCHS = 30
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 0.001
GRADIENT_CLIP_VAL = 0.1
EARLY_STOP_PATIENCE = 5
VAL_SPLIT_FRAC = 0.15  # last 15% of each stock for validation

QUANTILES = [0.1, 0.5, 0.9]

# TFT feature columns (subset that TFT will use)
TFT_FEATURES = [
    "close", "open", "high", "low", "volume",
    "rsi_14", "macd", "ema_20", "ema_50",
    "atr_14", "volume_ratio", "bb_percent",
]

# Top 100 liquid NSE stocks (hardcoded for reproducibility)
TOP_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "SBIN", "KOTAKBANK", "LT", "AXISBANK",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "ABB", "SIEMENS",
    "HAL", "BEL", "TRENT", "POLYCAB", "PERSISTENT",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "TITAN", "NESTLEIND", "ASIANPAINT", "ULTRACEMCO", "GRASIM",
    "ADANIENT", "ADANIPORTS", "WIPRO", "TECHM", "HCLTECH",
    "POWERGRID", "NTPC", "ONGC", "BPCL", "COALINDIA",
    "ITC", "HINDUNILVR", "BRITANNIA", "DABUR", "MARICO",
    "BAJFINANCE", "BAJAJFINSV", "MARUTI", "EICHERMOT", "DLF",
    # Extended to 100 liquid stocks
    "PIDILITIND", "HAVELLS", "VOLTAS", "INDIGO", "DMART",
    "JUBLFOOD", "COLPAL", "GODREJCP", "BERGEPAINT", "CROMPTON",
    "CHOLAFIN", "SHRIRAMFIN", "MFSL", "SBICARD", "ICICIPRULI",
    "HDFCLIFE", "BAJAJ-AUTO", "M&M", "HEROMOTOCO", "TATAPOWER",
    "SRF", "PIIND", "NAVINFLUOR", "DEEPAKNTR", "ATUL",
    "LUPIN", "AUROPHARMA", "BIOCON", "GLENMARK", "ALKEM",
    "LTIM", "LTTS", "MPHASIS", "KPITTECH", "COFORGE",
    "SAIL", "NMDC", "NATIONALUM", "JINDALSTEL", "TATACHEM",
    "INDUSINDBK", "BANKBARODA", "PNB", "CANBK", "IDFCFIRSTB",
    "RECLTD", "PFC", "IRFC", "NHPC", "TORNTPOWER",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_stocks() -> dict:
    """Download 2y daily OHLCV for top-50 NSE stocks."""
    tickers = [f"{sym}.NS" for sym in TOP_50_SYMBOLS]
    stock_dfs = {}
    total_batches = (len(tickers) + BATCH_SIZE_DL - 1) // BATCH_SIZE_DL
    t0 = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE_DL
        end = min(start + BATCH_SIZE_DL, len(tickers))
        batch = tickers[start:end]
        ticker_str = " ".join(batch)

        try:
            raw = yf.download(
                ticker_str, period="10y", interval="1d",
                progress=False, group_by="ticker", threads=True,
            )
            if raw.empty:
                continue
        except Exception:
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()

                if df.empty or len(df) < MIN_BARS:
                    continue

                df = df.dropna(subset=["Close"])
                if len(df) < MIN_BARS:
                    continue

                # Normalize columns to lowercase
                df.columns = [
                    c.lower() if isinstance(c, str) else c[0].lower()
                    for c in df.columns
                ]

                # Volume filter
                if "volume" in df.columns and df["volume"].mean() < MIN_AVG_VOL:
                    continue

                sym = ticker.replace(".NS", "")
                stock_dfs[sym] = df
            except Exception:
                continue

        elapsed = time.time() - t0
        print(
            f"  Batch {batch_idx + 1}/{total_batches} | "
            f"Downloaded: {len(stock_dfs)} stocks | "
            f"Elapsed: {elapsed:.0f}s"
        )

    print(f"Download complete: {len(stock_dfs)} stocks in {time.time() - t0:.0f}s\n")
    return stock_dfs


def build_tft_dataframe(stock_dfs: dict) -> pd.DataFrame:
    """
    Build a single long-format DataFrame suitable for TimeSeriesDataSet.
    Columns: time_idx, symbol, close, + all TFT_FEATURES.
    """
    frames = []

    for sym, df in stock_dfs.items():
        try:
            featured = compute_features(df)
        except Exception as e:
            print(f"  Skipping {sym}: feature computation failed ({e})")
            continue

        # Keep only the columns we need
        cols_needed = [c for c in TFT_FEATURES if c in featured.columns]
        if len(cols_needed) < len(TFT_FEATURES):
            missing = set(TFT_FEATURES) - set(cols_needed)
            print(f"  Skipping {sym}: missing columns {missing}")
            continue

        subset = featured[TFT_FEATURES].copy()
        subset["symbol"] = sym

        # Drop NaN rows (from indicator warm-up)
        subset = subset.dropna()

        if len(subset) < MIN_BARS:
            continue

        # Integer time index (per stock, sequential)
        subset = subset.reset_index(drop=True)
        subset["time_idx"] = np.arange(len(subset))

        frames.append(subset)

    if not frames:
        raise RuntimeError("No valid stock data after feature computation.")

    combined = pd.concat(frames, ignore_index=True)

    # Ensure correct types
    combined["symbol"] = combined["symbol"].astype(str)
    combined["time_idx"] = combined["time_idx"].astype(int)
    for col in TFT_FEATURES:
        combined[col] = combined[col].astype(float)

    # Replace inf with NaN, then fill
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.fillna(0.0)

    print(f"Combined dataset: {len(combined):,} rows, {combined['symbol'].nunique()} stocks")
    return combined


def create_datasets(data: pd.DataFrame):
    """
    Create train and validation TimeSeriesDataSets.
    Validation = last VAL_SPLIT_FRAC of each stock's time series.
    """
    # Determine the training cutoff per stock
    max_time_idx = data.groupby("symbol")["time_idx"].transform("max")
    cutoff = max_time_idx * (1.0 - VAL_SPLIT_FRAC)
    training_data = data[data["time_idx"] <= cutoff].copy()
    validation_data = data.copy()  # Full data, but val only uses post-cutoff

    # Determine max time_idx for training cutoff
    training_cutoff = int(training_data["time_idx"].max())

    time_varying_unknown_reals = TFT_FEATURES.copy()

    training_dataset = TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target="close",
        group_ids=["symbol"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_unknown_reals=time_varying_unknown_reals,
        time_varying_known_reals=[],
        static_categoricals=["symbol"],
        target_normalizer=GroupNormalizer(
            groups=["symbol"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_data,
        predict=False,
        stop_randomization=True,
    )

    return training_dataset, validation_dataset


def print_sample_predictions(model, val_dataset, data: pd.DataFrame):
    """Print sample predictions for 3 stocks."""
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=TRAIN_BATCH_SIZE, num_workers=0
    )

    # Get raw predictions (quantiles)
    raw_preds = model.predict(val_loader, mode="raw", return_x=True)

    predictions = raw_preds.output
    # predictions["prediction"] shape: [batch, horizon, n_quantiles]
    pred_tensor = predictions["prediction"]
    x = raw_preds.x

    # Pick 3 random batch indices
    n_samples = min(3, pred_tensor.shape[0])
    indices = np.random.choice(pred_tensor.shape[0], size=n_samples, replace=False)

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (actual vs predicted 5-bar close with quantiles)")
    print("=" * 70)

    for i, idx in enumerate(indices):
        # Get the decoder target (actual future values)
        decoder_target = x["decoder_target"][idx].detach().cpu().numpy()
        pred_q = pred_tensor[idx].detach().cpu().numpy()  # [horizon, n_quantiles]

        print(f"\nSample {i + 1}:")
        print(f"  {'Bar':>4s}  {'Actual':>10s}  {'P10':>10s}  {'P50':>10s}  {'P90':>10s}")
        print(f"  {'-' * 4}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

        for t in range(MAX_PREDICTION_LENGTH):
            actual = decoder_target[t] if t < len(decoder_target) else float("nan")
            p10 = pred_q[t, 0] if pred_q.shape[1] >= 1 else float("nan")
            p50 = pred_q[t, 1] if pred_q.shape[1] >= 2 else float("nan")
            p90 = pred_q[t, 2] if pred_q.shape[1] >= 3 else float("nan")
            print(f"  t+{t + 1:>2d}  {actual:10.2f}  {p10:10.2f}  {p50:10.2f}  {p90:10.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("TFT Price Forecaster Training")
    print("=" * 60)
    print(f"  Stocks:            {MAX_STOCKS}")
    print(f"  Encoder length:    {MAX_ENCODER_LENGTH} bars")
    print(f"  Prediction length: {MAX_PREDICTION_LENGTH} bars")
    print(f"  Quantiles:         {QUANTILES}")
    print(f"  Hidden size:       {HIDDEN_SIZE}")
    print(f"  Attention heads:   {ATTENTION_HEAD_SIZE}")
    print(f"  Max epochs:        {MAX_EPOCHS}")
    print(f"  Device:            CPU")
    print()

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print("Step 1: Downloading OHLCV data...")
    stock_dfs = download_stocks()
    if len(stock_dfs) < 10:
        print("ERROR: Too few stocks downloaded. Check network connection.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Build feature DataFrame
    # ------------------------------------------------------------------
    print("Step 2: Computing features...")
    t0 = time.time()
    data = build_tft_dataframe(stock_dfs)
    print(f"  Feature computation: {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # 3. Create TimeSeriesDataSets
    # ------------------------------------------------------------------
    print("Step 3: Creating TimeSeriesDataSets...")
    training_dataset, validation_dataset = create_datasets(data)

    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=TRAIN_BATCH_SIZE, num_workers=0
    )
    val_loader = validation_dataset.to_dataloader(
        train=False, batch_size=TRAIN_BATCH_SIZE, num_workers=0
    )

    print(f"  Training samples:   {len(training_dataset):,}")
    print(f"  Validation samples: {len(validation_dataset):,}")
    print()

    # ------------------------------------------------------------------
    # 4. Create TFT model
    # ------------------------------------------------------------------
    print("Step 4: Building TFT model...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    param_count = sum(p.numel() for p in tft.parameters())
    print(f"  Model parameters: {param_count:,}")
    print()

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("Step 5: Training...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[early_stop],
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
    )

    t_train = time.time()
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_time = time.time() - t_train
    print(f"\nTraining completed in {train_time:.0f}s")

    # ------------------------------------------------------------------
    # 6. Validation metrics
    # ------------------------------------------------------------------
    print("\nStep 6: Validation metrics...")
    val_results = trainer.validate(tft, dataloaders=val_loader, verbose=False)
    if val_results:
        val_loss = val_results[0].get("val_loss", float("nan"))
        print(f"  Validation loss (QuantileLoss): {val_loss:.6f}")

    # Compute MAE on median predictions
    preds = tft.predict(val_loader, mode="prediction")
    actuals_loader = val_loader
    actuals_list = []
    for batch in actuals_loader:
        x, y = batch
        actuals_list.append(y[0])  # y is (target, weight) tuple
    actuals = torch.cat(actuals_list, dim=0)

    # preds shape: [N, horizon], actuals shape: [N, horizon]
    mae = (preds - actuals).abs().mean().item()
    print(f"  Validation MAE (median): {mae:.4f}")

    # ------------------------------------------------------------------
    # 7. Sample predictions
    # ------------------------------------------------------------------
    print_sample_predictions(tft, validation_dataset, data)

    # ------------------------------------------------------------------
    # 8. Save model and config
    # ------------------------------------------------------------------
    print("\nStep 7: Saving model and config...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save checkpoint via trainer's best model path, or manually
    best_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
    if best_path and os.path.exists(best_path):
        # Copy best checkpoint to our target path
        import shutil
        shutil.copy2(best_path, CKPT_PATH)
    else:
        # Save current model state
        trainer.save_checkpoint(CKPT_PATH)

    # Save dataset parameters for inference (TFTPredictor.predict_quantiles)
    dataset_params = training_dataset.get_parameters()

    # Convert to JSON-serializable config
    config = {
        "quantiles": QUANTILES,
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "max_prediction_length": MAX_PREDICTION_LENGTH,
        "time_varying_unknown_reals": TFT_FEATURES,
        "target": "close",
        "group_ids": ["symbol"],
        "features": TFT_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "attention_head_size": ATTENTION_HEAD_SIZE,
        "hidden_continuous_size": HIDDEN_CONTINUOUS_SIZE,
        "dropout": DROPOUT,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Also save full dataset params as .pt for TimeSeriesDataSet.from_parameters()
    pt_config_path = CONFIG_PATH.replace(".json", ".pt")
    torch.save(dataset_params, pt_config_path)

    ckpt_size = os.path.getsize(CKPT_PATH) / 1024
    print(f"  Checkpoint: {CKPT_PATH} ({ckpt_size:.0f} KB)")
    print(f"  Config:     {CONFIG_PATH}")
    print(f"  Params:     {pt_config_path}")

    print("\n" + "=" * 60)
    print("TFT training complete.")
    print(f"  Stocks used:      {data['symbol'].nunique()}")
    print(f"  Training samples: {len(training_dataset):,}")
    print(f"  Val MAE (median): {mae:.4f}")
    print(f"  Training time:    {train_time:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

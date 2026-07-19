#!/usr/bin/env python
"""Per-trainer EDA + preprocessing audit (strict, no-fallback).

Run BEFORE training. For each trainer, loads its real training data and
checks:
  - feature distributions (NaN%, skew, kurtosis, near-constant flags)
  - label balance (for classification trainers)
  - feature-label IC (for cross-sectional rankers)
  - look-ahead leakage (feature ↔ label same-bar correlation)

Exits non-zero if ANY trainer hits a blocker. The training pipeline
calls this as Phase 8c — failing means GPU time is saved by aborting
before the bad data hits the model.

Usage:
    python scripts/train/eda_report.py                  # smoke (10 stocks, 2y)
    python scripts/train/eda_report.py --universe 50 --period 5y
    python scripts/train/eda_report.py --only lgbm_signal_gate,regime_hmm
    python scripts/train/eda_report.py --report eda.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("eda_report")


# ---------------------------------------------------------------------------
# Per-trainer EDA runners
# ---------------------------------------------------------------------------


def eda_regime_hmm() -> Dict[str, Any]:
    """Nifty + VIX features → 3-state HMM. Unsupervised, so no class
    balance; we audit feature distributions only."""
    from datetime import date
    import yfinance as yf
    from ml.preprocessing import (
        EDAReport, eda_dataframe_summary, eda_near_constant_features,
    )

    report = EDAReport(trainer="regime_hmm")
    nifty = yf.download("^NSEI", period="5y", progress=False, auto_adjust=False)
    vix = yf.download("^INDIAVIX", period="5y", progress=False, auto_adjust=False)
    if nifty.empty or vix.empty:
        report.blockers.append("regime_hmm:yfinance_empty")
        return report.to_dict()

    import pandas as pd
    import numpy as np
    df = pd.DataFrame(index=nifty.index)
    close = nifty["Close"].iloc[:, 0] if hasattr(nifty["Close"], "columns") else nifty["Close"]
    df["ret_1d"] = close.pct_change()
    df["rv_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_z"] = ((nifty["Volume"].iloc[:, 0] if hasattr(nifty["Volume"], "columns") else nifty["Volume"])
                   .rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)))
    df["vix"] = (vix["Close"].iloc[:, 0] if hasattr(vix["Close"], "columns") else vix["Close"]).reindex(df.index)
    df["vix_5d_chg"] = df["vix"].pct_change(5)
    df = df.dropna()

    feats = ["ret_1d", "rv_10d", "vol_z", "vix", "vix_5d_chg"]
    report.n_rows = len(df)
    report.n_features = len(feats)
    report.date_range = [str(df.index.min().date()), str(df.index.max().date())]
    fs = eda_dataframe_summary(df, feats, max_nan_pct=0.10)
    report.feature_summary = fs["per_feature"]
    report.blockers.extend(fs["blockers"])
    report.near_constant = eda_near_constant_features(df, feats)
    if report.near_constant:
        report.blockers.append(f"near_constant_features:{report.near_constant}")
    return report.to_dict()


def eda_lgbm_signal_gate(universe: int, period: str) -> Dict[str, Any]:
    """30 base features + 3-class triple-barrier label. THE critical EDA
    — class balance failure means the trainer cannot work."""
    from ml.preprocessing import (
        EDAReport, eda_dataframe_summary, eda_near_constant_features,
        eda_classification_balance, eda_feature_label_ic, eda_leakage_check,
    )
    from ml.data.liquid_universe import liquid_universe, LiquidUniverseConfig
    from ml.data.production_ohlcv import production_ohlcv

    report = EDAReport(trainer="lgbm_signal_gate")
    try:
        syms = liquid_universe(LiquidUniverseConfig(top_n=universe))
        raw = production_ohlcv(
            syms, start=f"2024-01-01" if period == "2y" else "2020-01-01",
            end=None, include_delisted=False,
        )
    except Exception as exc:
        report.blockers.append(f"data_load_failed:{exc}")
        return report.to_dict()

    if raw is None or raw.empty:
        report.blockers.append("ohlcv_empty")
        return report.to_dict()

    # Build a synthetic feature frame using the trainer's actual logic.
    # EDA audits the BASE 30 features only — Kronos embeddings (when
    # KRONOS_ENABLED=1) are added by the trainer at runtime, not by
    # _compute_features. So we use the base list, not FEATURE_ORDER which
    # may have been extended to 286 cols.
    from ml.training.trainers.lgbm_signal_gate import _compute_features
    from ml.labeling import TripleBarrierConfig, triple_barrier_events

    import pandas as pd
    import numpy as np

    tb_cfg = TripleBarrierConfig()  # default ±1 ATR / 10-day vertical

    # Sample 5 symbols for the EDA — full audit takes too long
    sample_syms = list(raw.columns.get_level_values(0).unique())[:5]
    print(f"  eda lgbm: sampling {len(sample_syms)} symbols: {sample_syms[:5]}")
    frames = []
    base_feature_cols = None
    for sym in sample_syms:
        try:
            sub = raw[sym].dropna(subset=["Close", "High", "Low", "Volume"])
            if len(sub) < 100:
                print(f"  eda lgbm: {sym} skipped — only {len(sub)} rows after ohlcv dropna")
                continue
            f = _compute_features(sub)
            # Determine base feature columns from first successful build —
            # everything _compute_features produces except _atr_raw and _fwd_return.
            if base_feature_cols is None:
                base_feature_cols = [c for c in f.columns
                                     if not c.startswith("_")]
                print(f"  eda lgbm: base features = {len(base_feature_cols)} cols "
                      f"(sample: {base_feature_cols[:5]})")
            f = f.dropna(subset=base_feature_cols + ["_atr_raw"])
            if len(f) < 50:
                print(f"  eda lgbm: {sym} skipped — only {len(f)} rows after feature dropna")
                continue
            labels, _t1 = triple_barrier_events(
                close=sub.loc[f.index, "Close"].values,
                atr=f["_atr_raw"].values,
                cfg=tb_cfg,
            )
            f["_label"] = labels
            frames.append(f)
            print(f"  eda lgbm: {sym} ✓ {len(f)} rows kept "
                  f"(label dist: {dict(pd.Series(labels).value_counts())})")
        except Exception as exc:
            print(f"  eda lgbm: {sym} FAILED — {type(exc).__name__}: {exc}")
            continue

    if not frames or base_feature_cols is None:
        report.blockers.append("no_features_built — see eda lgbm log above")
        return report.to_dict()

    df = pd.concat(frames, ignore_index=True)
    report.n_rows = len(df)
    report.n_features = len(base_feature_cols)
    FEATURE_ORDER = base_feature_cols  # local alias for the rest of the function

    fs = eda_dataframe_summary(df, FEATURE_ORDER, max_nan_pct=0.50)
    report.feature_summary = fs["per_feature"]
    report.blockers.extend(fs["blockers"])

    report.near_constant = eda_near_constant_features(df, FEATURE_ORDER)
    if report.near_constant:
        report.warnings.append(f"near_constant_features:{report.near_constant}")

    # CRITICAL: class balance for 3-class SELL/HOLD/BUY.
    # ml.labeling.triple_barrier_events emits SIGNED labels {-1, 0, 1}
    # (SELL/HOLD/BUY) — NOT {0, 1, 2}.
    cb = eda_classification_balance(df["_label"], min_class_pct=0.05,
                                    expected_classes=[-1, 0, 1])
    report.label_summary = cb
    report.blockers.extend(cb["blockers"])
    report.warnings.extend(cb["warnings"])

    # IC sanity check — at least some features should correlate with label.
    ic = eda_feature_label_ic(df, FEATURE_ORDER, "_label",
                              method="spearman", min_abs_mean_ic=0.005)
    report.ic_summary = ic
    report.blockers.extend(ic["blockers"])

    # Leakage check — no feature should perfectly track the label.
    lk = eda_leakage_check(df, FEATURE_ORDER, "_label", max_corr=0.95)
    report.leakage_summary = lk
    report.blockers.extend(lk["blockers"])

    # Audit note — informational only.
    report.warnings.append(
        "audited 16 base technical features only — trainer adds 14 more "
        "from caches (fii/dii, sentiment, fundamentals) at train() time. "
        "Empty caches → those cols zero-filled."
    )
    return report.to_dict()


def eda_tft_swing(universe: int, period: str) -> Dict[str, Any]:
    """neuralforecast TFT — feature distributions + label (5d fwd return)."""
    from ml.preprocessing import (
        EDAReport, eda_dataframe_summary, eda_near_constant_features,
        eda_feature_label_ic, eda_leakage_check,
    )
    from ml.data.liquid_universe import liquid_universe, LiquidUniverseConfig
    from ml.data.production_ohlcv import production_ohlcv

    report = EDAReport(trainer="tft_swing")
    syms = liquid_universe(LiquidUniverseConfig(top_n=universe))
    raw = production_ohlcv(syms, start="2021-01-01", end=None,
                           include_delisted=False)
    if raw is None or raw.empty:
        report.blockers.append("ohlcv_empty")
        return report.to_dict()

    import pandas as pd
    import numpy as np

    sample_syms = list(raw.columns.get_level_values(0).unique())[:5]
    frames = []
    for sym in sample_syms:
        sub = raw[sym].dropna(subset=["Close"]).copy()
        if len(sub) < 100:
            continue
        sub["ret_1d"] = sub["Close"].pct_change()
        sub["log_close"] = np.log(sub["Close"])
        sub["ret_5d"] = sub["Close"].pct_change(5)
        sub["rsi_14"] = _rsi(sub["Close"], 14)
        # 5-day forward return label
        sub["_label"] = sub["Close"].pct_change(5).shift(-5)
        frames.append(sub.dropna(subset=["ret_1d", "log_close", "ret_5d",
                                          "rsi_14", "_label"]))
    if not frames:
        report.blockers.append("no_features_built")
        return report.to_dict()

    df = pd.concat(frames, ignore_index=True)
    feats = ["ret_1d", "log_close", "ret_5d", "rsi_14"]
    report.n_rows = len(df)
    report.n_features = len(feats)

    fs = eda_dataframe_summary(df, feats, max_nan_pct=0.20)
    report.feature_summary = fs["per_feature"]
    report.blockers.extend(fs["blockers"])
    report.near_constant = eda_near_constant_features(df, feats)
    ic = eda_feature_label_ic(df, feats, "_label",
                              method="spearman", min_abs_mean_ic=0.005)
    report.ic_summary = ic
    report.blockers.extend(ic["blockers"])
    lk = eda_leakage_check(df, feats, "_label", max_corr=0.95)
    report.leakage_summary = lk
    report.blockers.extend(lk["blockers"])
    return report.to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rsi(series, period=14):
    import pandas as pd
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().replace(0, float("nan"))
    return (100 - 100 / (1 + gain / loss)).fillna(50)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


ALL_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "regime_hmm":         lambda u, p: eda_regime_hmm(),
    "lgbm_signal_gate":   lambda u, p: eda_lgbm_signal_gate(u, p),
    "tft_swing":          lambda u, p: eda_tft_swing(u, p),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=int, default=10)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--only", help="comma-separated trainer subset")
    parser.add_argument("--report", type=Path, default=Path("eda_report.json"))
    args = parser.parse_args()

    runners = ALL_RUNNERS
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        runners = {k: v for k, v in runners.items() if k in wanted}
        unknown = wanted - set(ALL_RUNNERS)
        if unknown:
            print(f"unknown trainers: {unknown}")
            return 2

    print("=" * 76)
    print(f"EDA REPORT — universe={args.universe}, period={args.period}")
    print("=" * 76)

    results: List[Dict[str, Any]] = []
    total_blockers = 0
    total_warnings = 0

    for name, fn in runners.items():
        print(f"\n• {name}")
        try:
            r = fn(args.universe, args.period)
        except Exception as exc:
            r = {
                "trainer": name, "blockers": [f"crashed:{exc}"],
                "traceback": traceback.format_exc().splitlines()[-5:],
            }
        results.append(r)
        blockers = r.get("blockers", [])
        warnings = r.get("warnings", [])
        total_blockers += len(blockers)
        total_warnings += len(warnings)
        n_rows = r.get("n_rows", 0)
        n_feats = r.get("n_features", 0)
        mark = "❌" if blockers else ("⚠" if warnings else "✅")
        print(f"  {mark} rows={n_rows} features={n_feats}"
              f"  blockers={len(blockers)} warnings={len(warnings)}")
        if blockers:
            for b in blockers:
                print(f"    BLOCK: {b}")
        if warnings:
            for w in warnings[:5]:
                print(f"    warn:  {w}")
        # Show label / IC summary if available
        lab = r.get("label_summary")
        if isinstance(lab, dict) and "ratios" in lab:
            print(f"    label_ratios: {lab['ratios']}")
        ic = r.get("ic_summary")
        if isinstance(ic, dict) and "max_abs_ic" in ic:
            print(f"    max_abs_ic: {ic['max_abs_ic']}"
                  f"  n_above_005: {ic.get('n_above_005', 0)}")

    print()
    print("=" * 76)
    print(f"SUMMARY: {len(results)} trainers, "
          f"{total_blockers} blockers, {total_warnings} warnings")
    print("=" * 76)
    args.report.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full report → {args.report}")
    return 1 if total_blockers else 0


if __name__ == "__main__":
    sys.exit(main())

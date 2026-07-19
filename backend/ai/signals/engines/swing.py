"""SwingEngine — serves the trained swing_lambdarank ranker.

Resolves the model (registry-first, disk fallback), builds the swing feature
panel per the trained artifact's feature_order, predicts, ranks
cross-sectionally, and attaches ATR-derived levels. Features are built on the
FULL universe panel (so cross-sectional ranks are real), plus a NIFTY
benchmark for relative-strength columns loaded via
``ml.data.benchmark.load_nifty_benchmark`` (offline cache first, yfinance
``^NSEI`` fallback) — NOT ``_load_ohlcv(["NSEI"])``, which routes NSEI through
the equity-symbol path and 404s on yfinance. The engine then slices each
symbol's last warmed bar and scores the exact columns the loaded model expects
(``self._feature_order``). Benchmark load is fail-soft: a missing benchmark
leaves the RS columns NaN — tolerated by LightGBM at predict time. CPU-only at
serve; when the artifact's feature_order lists forecast columns (tsfm/kronos/
chronos/ens — Phase 2), the latest weekly-cached values are LEFT-merged per
symbol via ``ml.features.forecast_serving.latest_forecasts``; an absent cache
degrades (forecast cols NaN, loud log) instead of erroring. Honest-empty if
the model is missing (no heuristic fallback).
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from backend.ai.signals.style_types import SwingSignal, Style
from backend.trading.risk_engine import derive_levels

logger = logging.getLogger(__name__)

_MODEL_NAME = "swing_lambdarank"
_ROOT = Path(__file__).resolve().parents[4]  # engines -> signals -> ai -> backend -> repo
_DISK_DIR = _ROOT / "artifacts" / "models" / "swing_lambdarank"


def _default_model_loader():
    """Registry-first; disk fallback. Returns (booster, feature_order, decile_spread)."""
    import lightgbm as lgb  # noqa: PLC0415
    txt: Optional[Path] = None
    fo_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    try:
        from backend.ai.registry import get_registry  # noqa: PLC0415
        d = get_registry().resolve(_MODEL_NAME)
        txt = d / "swing_lambdarank.txt"
        fo_path = d / "feature_order.json"
        metrics_path = d / "metrics.json"
    except Exception as exc:  # registry miss -> disk fallback
        logger.info("swing registry resolve failed (%s); trying disk", exc)
    if txt is None or not txt.exists():
        txt = _DISK_DIR / "swing_lambdarank.txt"
        fo_path = _DISK_DIR / "feature_order.json"
        metrics_path = _DISK_DIR / "metrics.json"
    if not txt.exists():
        raise LookupError(f"swing model artifact not found at {txt}")
    # model_str avoids a native model_file parser crash on Windows.
    booster = lgb.Booster(model_str=txt.read_text(encoding="utf-8"))
    feature_order = json.loads(fo_path.read_text()) if fo_path and fo_path.exists() \
        else list(booster.feature_name())
    decile = 0.03
    if metrics_path and metrics_path.exists():
        decile = float(json.loads(metrics_path.read_text()).get("decile_spread_mean", decile))
    return booster, feature_order, decile


def _atr14(df: pd.DataFrame) -> float:
    """Simple ATR(14) on the latest bar from a single-symbol OHLCV frame."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


class SwingEngine:
    def __init__(
        self,
        *,
        _booster=None,
        _feature_order: Optional[List[str]] = None,
        _decile_spread: float = 0.03,
        _model_loader: Optional[Callable] = None,
        _universe: Optional[Callable] = None,
        _load_ohlcv: Optional[Callable] = None,
    ):
        self.status = "ok"
        # True when the artifact expects forecast columns but the weekly
        # cache was absent/unreadable — scoring proceeds (status stays "ok")
        # with NaN forecast cols, which degrades the model's edge.
        self.forecast_degraded = False
        self._booster = _booster
        self._feature_order = _feature_order
        self._decile = _decile_spread
        self._model_loader = _model_loader or _default_model_loader
        if _universe is None:
            # Same NSE universe as momentum — the swing trainer reuses it too.
            from ml.training.trainers.momentum_lambdarank import cached_universe  # noqa: PLC0415
            _universe = cached_universe
        self._universe = _universe
        if _load_ohlcv is None:
            from ml.data.data_loader import load_ohlcv  # noqa: PLC0415
            _load_ohlcv = load_ohlcv
        self._load_ohlcv = _load_ohlcv

    def _ensure_model(self) -> bool:
        if self._booster is not None:
            if self._feature_order is None:
                from ml.features.swing_features import SWING_FEATURE_ORDER  # noqa: PLC0415
                self._feature_order = list(SWING_FEATURE_ORDER)
            return True
        try:
            self._booster, self._feature_order, self._decile = self._model_loader()
            return True
        except Exception as exc:
            logger.warning("SwingEngine model load failed: %s", exc)
            self.status = "model_not_loaded"
            return False

    def _merge_forecasts(self, feats: pd.DataFrame) -> pd.DataFrame:
        """LEFT-merge the latest weekly-cached forecast values per symbol.

        The NEW artifacts' feature_order includes forecast columns
        (tsfm/kronos/chronos/ens for swing); serving builds only price/RS
        features, so without this merge ``dropna`` on the artifact contract
        would kill every row. The latest cached forecast per symbol applies to
        the current scoring bar — staleness up to stride + weekly-refresh
        cadence is by design (Phase 2 two-speed inference). Missing cache: the
        columns are added as NaN so the artifact contract is satisfied
        structurally (LightGBM tolerates NaN at predict), with a loud
        degraded=True log.
        """
        from ml.features.forecast_features import FORECAST_FEATURES  # noqa: PLC0415
        from ml.features.forecast_serving import latest_forecasts  # noqa: PLC0415
        fcols = [c for c in self._feature_order if c in FORECAST_FEATURES]
        if not fcols:
            return feats
        fx = None
        try:
            fx = latest_forecasts("swing")
        except Exception as exc:  # noqa: BLE001 — cache read must never kill scoring
            logger.warning("swing forecast-cache read failed: %s", exc)
        if fx is None:
            self.forecast_degraded = True
            logger.warning(
                "swing: forecast cache absent — scoring degraded=True "
                "(forecast cols NaN; the ranker loses its forecast edge). Run "
                "scripts/runpod/refresh_forecast_cache.sh to restore it.")
        else:
            keep = ["symbol"] + [c for c in fcols if c in fx.columns]
            feats = feats.merge(fx[keep], on="symbol", how="left")
            logger.info(
                "swing: merged %d forecast cols for %d symbols (age=%dd)",
                len(keep) - 1, fx["symbol"].nunique(),
                int(fx["forecast_age_days"].iloc[0]) if len(fx) else -1)
        for c in fcols:
            if c not in feats.columns:
                feats[c] = float("nan")
        return feats

    def run(self, top_n: int = 20, universe_limit: Optional[int] = None) -> List[SwingSignal]:
        from ml.features.swing_features import build_swing_features, SWING_WARMUP_BARS  # noqa: PLC0415
        from ml.features.forecast_features import FORECAST_FEATURES  # noqa: PLC0415
        from ml.data.benchmark import load_nifty_benchmark  # noqa: PLC0415
        self.forecast_degraded = False
        if not self._ensure_model():
            return []
        syms = self._universe(limit=universe_limit)
        if not syms:
            self.status = "no_data"
            return []
        end = date.today()
        # Window sizing: the 63-bar warmup is ~90 calendar days, and the RS
        # beta/corr_63 windows need a further ~63 trading days of benchmark
        # overlap on top of it — momentum's `* 2.2 + 30` sizing would give only
        # ~170 days here; `* 2 + 60` ≈ 186 days keeps the last bars fully warmed.
        # 4x warmup + buffer: beta/corr_63's paired-return chain needs ~118 trading
        # bars; 2x warmup (186cal ~ 125 trading) left ~7 valid rows/symbol and
        # served no_data (caught at deploy verification 2026-07-07).
        start = end - timedelta(days=SWING_WARMUP_BARS * 4 + 60)
        panel = self._load_ohlcv(syms, start, end, freq="eod")
        if panel is None or panel.empty:
            self.status = "no_data"
            return []

        panel = panel.sort_values(["symbol", "date"])
        cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
        # NIFTY benchmark for relative-strength features, via the shared
        # cache-first loader (NOT _load_ohlcv(["NSEI"]) — the equity-symbol
        # path 404s on yfinance for the index). Fail-soft: a missing benchmark
        # leaves the RS columns NaN, tolerated by LightGBM at predict time.
        try:
            bench = load_nifty_benchmark(start, end)
        except Exception as exc:
            logger.info("swing benchmark (NIFTY) load failed: %s", exc)
            bench = None
        try:
            feats_all = build_swing_features(panel[cols], benchmark=bench)
        except Exception as exc:
            logger.warning("swing feature build failed: %s", exc)
            self.status = "no_data"
            return []
        feats_all = self._merge_forecasts(feats_all)
        # dropna over the NON-forecast subset only: forecast cols are NaN
        # whenever the weekly cache is absent/partial, and LightGBM tolerates
        # NaN at predict — dropping on them would kill every row.
        try:
            required = [c for c in self._feature_order if c not in FORECAST_FEATURES]
            feats_all = feats_all.dropna(subset=required)
        except Exception as exc:
            logger.warning("swing feature frame vs artifact contract failed: %s", exc)
            self.status = "no_data"
            return []
        if feats_all.empty:
            self.status = "no_data"
            return []

        panel_by_sym = {s: gg.sort_values("date") for s, gg in panel.groupby("symbol")}
        rows = []  # (symbol, score, close, atr)
        for sym, gf in feats_all.groupby("symbol"):
            try:
                last = gf.sort_values("date").iloc[[-1]][self._feature_order]
                score = float(self._booster.predict(last)[0])
                g = panel_by_sym[sym]
                atr = _atr14(g)
                close = float(g["close"].iloc[-1])
                if atr > 0 and close > 0:
                    rows.append((sym, score, close, atr))
            except Exception as exc:
                logger.debug("swing feature/predict failed for %s: %s", sym, exc)

        if not rows:
            self.status = "no_data"
            return []

        rows.sort(key=lambda r: r[1], reverse=True)
        n = len(rows)
        out: List[SwingSignal] = []
        for i, (sym, score, close, atr) in enumerate(rows[:top_n]):
            rank = i + 1
            percentile = 1.0 if n == 1 else round(1.0 - (rank - 1) / (n - 1), 4)
            expected_return = round(self._decile * (2 * percentile - 1), 4)
            top_decile_prob = 1.0 if percentile >= 0.9 else round(percentile / 0.9, 4)
            confidence = round(percentile * 100, 1)
            entry, sl, target, rr = derive_levels("BUY", close, atr, Style.SWING)
            out.append(SwingSignal(
                symbol=sym, style=Style.SWING, rank=rank, percentile=percentile,
                confidence=confidence, direction="BUY", entry_price=entry, stop_loss=sl,
                target=target, risk_reward=rr,
                reasons=[f"Swing rank {rank}/{n}", f"percentile {percentile:.0%}"],
                expected_return=expected_return, top_decile_prob=top_decile_prob,
            ))
        self.status = "ok"
        return out

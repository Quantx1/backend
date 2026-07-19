# Momentum Full ML+DL Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the momentum model to the full 4-engine spec §5.1 — expand the 18-feature CPU base to ~75 features (multi-horizon momentum, quality, trend/MA, volume, volatility, RS-vs-NIFTY, cross-sectional ranks, liquidity) plus the TimesFM+Kronos DL forecast columns, then retrain and update serving.

**Architecture:** Pure feature-engineering in `ml/features/momentum_features.py` (stock-panel-only features) + a benchmark (NIFTY) RS layer + the existing GPU forecast columns (`forecast_features.py`, already wired via `with_forecasts`). The LightGBM LambdaRank ranker and the serving `MomentumEngine` consume whatever `MOMENTUM_FEATURE_ORDER` defines, so expanding it flows through automatically once the serving feature-build is updated.

**Tech Stack:** pandas/numpy, `ta` (ADX/ATR), LightGBM 4.6 (LambdaRank), TimesFM + Kronos (DL forecasters, GPU; debugged on Mac CPU), purged walk-forward CV.

**Spec:** `docs/superpowers/specs/2026-06-15-quantx-4engine-mldl-design.md` §5.1. **Approved feature list:** see session (groups A–I, ~80 incl forecasts).

**Run tests:** `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest <path> -v`
**Full DL smoke (Mac CPU):** `bash scripts/runpod/smoke_momentum_local.sh` (clones Kronos to `~/.kronos_repo`, FORECAST_DEVICE=cpu).
**Gate:** full suite stays green (882) · `lint-imports` 1 kept/0 broken.

---

## File Structure

- Modify `ml/features/momentum_features.py` — the feature factory. `MOMENTUM_FEATURE_ORDER` grows 18→~67 base (Tasks 1–2); `build_momentum_features` gains an optional `benchmark` arg for RS. One file, one responsibility (momentum features).
- Modify `ml/features/forecast_features.py` — add the `ens_fwd_ret` ensemble column (Task 3).
- Modify `ml/training/trainers/momentum_lambdarank.py` — load the NIFTY benchmark + pass it to the feature builder; the `with_forecasts` path already exists (Task 4).
- Modify `scripts/runpod/smoke_momentum_local.sh` — update the `n_features` assertion (Task 4).
- Modify `backend/ai/signals/engines/momentum.py` — serving: load the NIFTY benchmark for RS + read weekly-cached forecast columns (Task 6).
- Tests: `tests/ml/features/test_momentum_features.py` (extend), `tests/ml/features/test_momentum_rs.py` (new), `tests/ml/features/test_forecast_ensemble.py` (new).

**Sector RS (`rs_sector_*`) is OUT of this plan** — sector index series aren't cached. Noted as a follow-on data task; the feature builder is written so adding a `sector_close` series later is a drop-in.

---

### Task 1: Expand stock-only features (momentum quality, trend/MA, volume, volatility, ranks, liquidity)

These need only the stock OHLCV panel — no benchmark. Adds ~47 features to the existing 18.

**Files:**
- Modify: `ml/features/momentum_features.py`
- Test: `tests/ml/features/test_momentum_features.py`

- [ ] **Step 1: Write the failing test** (append to the existing test file)

```python
def test_expanded_feature_set_present_and_finite():
    import numpy as np, pandas as pd
    from ml.features.momentum_features import build_momentum_features, MOMENTUM_FEATURE_ORDER
    # one symbol, 400 trading days of a gentle uptrend
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    close = 100 + np.linspace(0, 40, 400)
    panel = pd.DataFrame({"date": idx, "symbol": "AAA", "open": close, "high": close*1.01,
                          "low": close*0.99, "close": close, "volume": 1_000_000})
    out = build_momentum_features(panel)
    # every declared feature column is produced
    for col in MOMENTUM_FEATURE_ORDER:
        assert col in out.columns, f"missing {col}"
    # the new feature families exist
    for col in ["ret_252_21","mom_decay","sharpe_63","dist_ema_21","sma_50_200_align",
                "vol_zscore_21","parkinson_vol_21","ulcer_index_63","adx_14",
                "turnover_21","amihud_illiq_21","xs_rank_ret_252"]:
        assert col in MOMENTUM_FEATURE_ORDER, f"{col} not in feature order"
    # last row (serving bar) is finite for every feature after warmup
    last = out.dropna(subset=MOMENTUM_FEATURE_ORDER).iloc[-1]
    assert np.isfinite(last[MOMENTUM_FEATURE_ORDER].astype(float)).all()
    assert len(MOMENTUM_FEATURE_ORDER) >= 60
```

- [ ] **Step 2: Run it, confirm FAIL** (`... missing ret_252_21` / feature count < 60)

`KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/features/test_momentum_features.py::test_expanded_feature_set_present_and_finite -v`

- [ ] **Step 3: Implement** — extend `MOMENTUM_FEATURE_ORDER` and add the computations in `build_momentum_features`, BEFORE the cross-sectional ranks block (ranks must come last so they see all symbols). Add these computations (all vectorized per-symbol via `groupby(sym).transform`, matching the existing style; `eps = 1e-9`):

```python
# --- B. momentum quality (extend) ---
df["ret_252_21"] = df.groupby(sym)["close"].transform(lambda s: s.shift(21)/s.shift(252) - 1.0)
df["mom_consistency_21"]  = df.groupby(sym)["__daily_ret"].transform(lambda s: _rolling_mean_positive(s, 21))
df["mom_consistency_126"] = df.groupby(sym)["__daily_ret"].transform(lambda s: _rolling_mean_positive(s, 126))
df["mom_consistency_252"] = df.groupby(sym)["__daily_ret"].transform(lambda s: _rolling_mean_positive(s, 252))
df["mom_accel_63_126"] = df["ret_63d"] - df["ret_126d"]
df["mom_decay"] = df["ret_21d"] - df["ret_63d"] / 3.0     # short vs long-implied
df["realized_vol_63"]  = df.groupby(sym)["__daily_ret"].transform(lambda s: s.rolling(63).std())
df["realized_vol_126"] = df.groupby(sym)["__daily_ret"].transform(lambda s: s.rolling(126).std())
df["vol_adj_mom_126"] = df["ret_126d"] / (df["realized_vol_63"] * np.sqrt(126) + eps)
df["sharpe_63"]  = df.groupby(sym)["__daily_ret"].transform(lambda s: s.rolling(63).mean()  / (s.rolling(63).std()  + eps))
df["sharpe_126"] = df.groupby(sym)["__daily_ret"].transform(lambda s: s.rolling(126).mean() / (s.rolling(126).std() + eps))
def _win_loss(s):
    up = s.clip(lower=0).rolling(63).mean(); dn = (-s.clip(upper=0)).rolling(63).mean()
    return up / (dn + eps)
df["win_loss_ratio_63"] = df.groupby(sym)["__daily_ret"].transform(_win_loss)

# --- C. trend / MA alignment (extend) ---
df["dist_sma_20"]  = df.groupby(sym)["close"].transform(lambda s: s/s.rolling(20).mean()  - 1.0)
df["dist_sma_100"] = df.groupby(sym)["close"].transform(lambda s: s/s.rolling(100).mean() - 1.0)
df["dist_ema_21"]  = df.groupby(sym)["close"].transform(lambda s: s/s.ewm(span=21, adjust=False).mean() - 1.0)
_sma20 = df.groupby(sym)["close"].transform(lambda s: s.rolling(20).mean())
_sma50 = df.groupby(sym)["close"].transform(lambda s: s.rolling(50).mean())
_sma200 = df.groupby(sym)["close"].transform(lambda s: s.rolling(200).mean())
df["sma_20_50_align"]  = (_sma20 > _sma50).astype(float)
df["sma_50_200_align"] = (_sma50 > _sma200).astype(float)
df["sma_50_slope_21"]  = df.groupby(sym)["close"].transform(lambda s: s.rolling(50).mean().pct_change(21))
df["sma_200_slope_63"] = df.groupby(sym)["close"].transform(lambda s: s.rolling(200).mean().pct_change(63))
df["pct_days_above_sma50_63"] = (df["close"] > _sma50).astype(float).groupby(sym).transform(lambda s: s.rolling(63).mean())
_hi252 = df.groupby(sym)["high"].transform(lambda s: s.rolling(252).max())
_lo252 = df.groupby(sym)["low"].transform(lambda s: s.rolling(252).min())
df["dist_high_252"] = df["close"] / (_hi252 + eps) - 1.0
df["price_vs_52w_range"] = (df["close"] - _lo252) / (_hi252 - _lo252 + eps)

# ADX(14) via `ta` (per-symbol; ta returns NaN warmup which dropna handles)
from ta.trend import ADXIndicator  # noqa: PLC0415
def _adx(g):
    return ADXIndicator(g["high"], g["low"], g["close"], window=14, fillna=False).adx()
df["adx_14"] = df.groupby(sym, group_keys=False).apply(lambda g: _adx(g)).reset_index(level=0, drop=True) \
    if False else df.groupby(sym).apply(lambda g: _adx(g)).reset_index(level=0, drop=True)

# --- D. volume confirmation (extend) ---
df["rel_volume_63"] = df.groupby(sym)["volume"].transform(lambda s: s/(s.rolling(63).mean() + eps))
df["vol_trend_21"]  = df.groupby(sym)["volume"].transform(lambda s: s.rolling(21).mean()/(s.rolling(63).mean() + eps))
df["vol_zscore_21"] = df.groupby(sym)["volume"].transform(lambda s: (s - s.rolling(21).mean())/(s.rolling(21).std() + eps))
df["volume_breakout"] = (df["rel_volume_21"] > 2.0).astype(float)
_upvol = (df["__daily_ret"] > 0).astype(float) * df["volume"]
df["up_vol_ratio_21"] = _upvol.groupby(sym).transform(lambda s: s.rolling(21).sum()) / \
    (df.groupby(sym)["volume"].transform(lambda s: s.rolling(21).sum()) + eps)
df["__pvt"] = (df["__daily_ret"].fillna(0.0) * df["volume"]).groupby(sym).cumsum()
df["pvt_slope_21"] = df.groupby(sym)["__pvt"].transform(lambda s: s.diff(21) / (s.abs().rolling(21).mean() + eps))
df["obv_slope_63"] = df.groupby(sym)["__obv"].transform(lambda s: s.diff(63) / (s.abs().rolling(63).mean() + eps))

# --- E. volatility / risk (extend) ---
df["vol_ratio"] = df["realized_vol_21"] / (df["realized_vol_63"] + eps)
df["downside_vol_63"] = df.groupby(sym)["__daily_ret"].transform(lambda s: s.clip(upper=0).rolling(63).std())
df["max_drawdown_63"] = df.groupby(sym)["close"].transform(lambda s: (s/s.rolling(63).max() - 1.0).rolling(63).min())
_hl = np.log(df["high"]/df["low"]).replace([np.inf,-np.inf], np.nan)
df["parkinson_vol_21"] = _hl.pow(2).groupby(sym).transform(lambda s: np.sqrt(s.rolling(21).mean()/(4*np.log(2))))
df["vol_of_vol_21"] = df.groupby(sym)["realized_vol_21"].transform(lambda s: s.rolling(21).std())
_atr = df.groupby(sym, group_keys=False).apply(
    lambda g: (pd.concat([(g["high"]-g["low"]),(g["high"]-g["close"].shift()).abs(),
                          (g["low"]-g["close"].shift()).abs()],axis=1).max(axis=1)).rolling(14).mean())
df["atr_pct_14"] = (_atr.reset_index(level=0, drop=True) / (df["close"] + eps))
_dd = df.groupby(sym)["close"].transform(lambda s: (s/s.cummax() - 1.0))
df["ulcer_index_63"] = _dd.pow(2).groupby(sym).transform(lambda s: np.sqrt(s.rolling(63).mean()))

# --- H. liquidity (NEW) ---
df["__tval"] = df["close"] * df["volume"]
df["turnover_21"] = df.groupby(sym)["__tval"].transform(lambda s: s.rolling(21).mean())
df["amihud_illiq_21"] = (df["__daily_ret"].abs() / (df["__tval"] + eps)).groupby(sym).transform(lambda s: s.rolling(21).mean())
df["dollar_vol_zscore_63"] = df.groupby(sym)["__tval"].transform(lambda s: (s - s.rolling(63).mean())/(s.rolling(63).std() + eps))
```

Then EXTEND the cross-sectional ranks block (these stay last):

```python
df["xs_rank_ret_126"] = df.groupby("date")["ret_126d"].rank(pct=True)
df["xs_rank_ret_252"] = df.groupby("date")["ret_252d"].rank(pct=True)
df["xs_rank_vol_adj_mom_63"]  = df.groupby("date")["vol_adj_mom_63"].rank(pct=True)
df["xs_rank_vol_adj_mom_126"] = df.groupby("date")["vol_adj_mom_126"].rank(pct=True)
df["xs_rank_sharpe_63"] = df.groupby("date")["sharpe_63"].rank(pct=True)
```

And add ALL the new names to `MOMENTUM_FEATURE_ORDER` (grouped, keeping the existing 18 first). Drop internal helpers (`__pvt`, `__tval`, `__obv`, `__daily_ret`) in the final column select (they already do this for `__obv`/`__daily_ret`).

> **Serving-safety note (critical, learned earlier):** every per-symbol feature must use `groupby(sym).transform(...)`, NEVER `groupby().apply()` that returns a DataFrame on a single-symbol frame. The two `.apply()` uses above (ADX, ATR) MUST be verified on a single-symbol panel — add `test_single_symbol_serving_path` below.

- [ ] **Step 4: Single-symbol serving-path test** (append)

```python
def test_single_symbol_path_no_crash():
    import numpy as np, pandas as pd
    from ml.features.momentum_features import build_momentum_features, MOMENTUM_FEATURE_ORDER
    idx = pd.date_range("2022-01-01", periods=320, freq="B")
    close = 100 + np.linspace(0, 30, 320)
    panel = pd.DataFrame({"date": idx, "symbol": "ONE", "open": close, "high": close*1.01,
                          "low": close*0.99, "close": close, "volume": 500_000})
    out = build_momentum_features(panel)   # must not raise on a single symbol
    assert set(MOMENTUM_FEATURE_ORDER).issubset(out.columns)
```

- [ ] **Step 5: Run both tests, confirm PASS.** If ADX/ATR `.apply()` crashes on the single-symbol frame, replace with a `groupby(sym).transform`-compatible form or compute ATR via the existing inline true-range pattern in `momentum.py::_atr14`.

- [ ] **Step 6: Commit**

```bash
git add ml/features/momentum_features.py tests/ml/features/test_momentum_features.py
git commit -m "feat(momentum): expand feature factory to ~65 stock-only features (quality/trend/volume/vol/ranks/liquidity)"
```

---

### Task 2: RS-vs-NIFTY features (benchmark layer)

**Files:**
- Modify: `ml/features/momentum_features.py` (add `benchmark` param + RS block)
- Test: `tests/ml/features/test_momentum_rs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/features/test_momentum_rs.py
import numpy as np, pandas as pd
from ml.features.momentum_features import build_momentum_features, MOMENTUM_FEATURE_ORDER

def _series(n=400, slope=40.0):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return idx, 100 + np.linspace(0, slope, n)

def test_rs_features_with_benchmark():
    idx, close = _series(slope=40)            # stock up 40%
    _, bench = _series(slope=10)              # NIFTY up 10% → stock has positive RS
    panel = pd.DataFrame({"date": idx, "symbol": "AAA", "open": close, "high": close*1.01,
                          "low": close*0.99, "close": close, "volume": 1_000_000})
    benchmark = pd.DataFrame({"date": idx, "close": bench})
    out = build_momentum_features(panel, benchmark=benchmark)
    for col in ["rs_index_21","rs_index_63","rs_index_126","rs_index_252",
                "rs_index_slope_21","beta_index_63","corr_index_63","xs_rank_rs_index_63"]:
        assert col in out.columns and col in MOMENTUM_FEATURE_ORDER
    last = out.dropna(subset=["rs_index_63"]).iloc[-1]
    assert last["rs_index_63"] > 0          # outperforming the benchmark

def test_rs_features_failsoft_without_benchmark():
    idx, close = _series()
    panel = pd.DataFrame({"date": idx, "symbol": "AAA", "open": close, "high": close*1.01,
                          "low": close*0.99, "close": close, "volume": 1_000_000})
    out = build_momentum_features(panel, benchmark=None)   # no benchmark → RS cols are NaN, not a crash
    assert out["rs_index_63"].isna().all()
```

- [ ] **Step 2: Run, confirm FAIL** (`build_momentum_features() got unexpected keyword 'benchmark'`).

- [ ] **Step 3: Implement** — change the signature to `build_momentum_features(panel, benchmark: Optional[pd.DataFrame] = None)` and, AFTER the per-symbol return windows but BEFORE the cross-sectional ranks, add:

```python
# --- F. RS-vs-NIFTY (benchmark = DataFrame['date','close']; None → NaN cols) ---
_RS_COLS = ["rs_index_21","rs_index_63","rs_index_126","rs_index_252",
            "rs_index_slope_21","beta_index_63","corr_index_63"]
if benchmark is not None and not benchmark.empty:
    b = benchmark[["date","close"]].rename(columns={"close":"__bclose"}).sort_values("date")
    df = df.merge(b, on="date", how="left")
    df["__bret"] = df["__bclose"].pct_change()  # NOTE: df is sorted by (symbol,date); see guard below
    for w in (21, 63, 126, 252):
        bret_w = df["__bclose"]/df["__bclose"].shift(w) - 1.0
        df[f"rs_index_{w}"] = df[f"ret_{w}d"] - bret_w
    df["rs_index_slope_21"] = df.groupby(sym)["rs_index_63"].transform(lambda s: s.diff(21))
    # rolling beta/corr of stock daily ret vs benchmark daily ret (per symbol)
    def _beta(g):
        cov = g["__daily_ret"].rolling(63).cov(g["__bret_sym"]); var = g["__bret_sym"].rolling(63).var()
        return cov/(var + eps)
    def _corr(g):
        return g["__daily_ret"].rolling(63).corr(g["__bret_sym"])
    df["__bret_sym"] = df.groupby(sym)["__bclose"].transform("pct_change")
    df["beta_index_63"] = df.groupby(sym, group_keys=False).apply(_beta).reset_index(level=0, drop=True)
    df["corr_index_63"] = df.groupby(sym, group_keys=False).apply(_corr).reset_index(level=0, drop=True)
    df.drop(columns=["__bclose","__bret","__bret_sym"], inplace=True, errors="ignore")
else:
    for c in _RS_COLS:
        df[c] = np.nan
```

> **Sort-order guard:** the existing builder sorts by `["symbol","date"]`. `__bclose.shift(w)` across that ordering is WRONG (it shifts across symbol boundaries). Fix: compute `bret_w` per symbol with `df.groupby(sym)["__bclose"].transform(lambda s, _w=w: s/s.shift(_w)-1.0)`. Apply the same per-symbol guard to any benchmark diff. (Write the test FIRST; it will catch a cross-symbol leak if you have ≥2 symbols — add a 2-symbol case.)

Then add the cross-sectional RS rank (in the ranks block): `df["xs_rank_rs_index_63"] = df.groupby("date")["rs_index_63"].rank(pct=True)` and append all 8 RS names to `MOMENTUM_FEATURE_ORDER`.

- [ ] **Step 4: Run, confirm PASS** (both RS tests).

- [ ] **Step 5: Commit**

```bash
git add ml/features/momentum_features.py tests/ml/features/test_momentum_rs.py
git commit -m "feat(momentum): RS-vs-NIFTY features (fail-soft without benchmark; sector RS deferred)"
```

---

### Task 3: Forecast ensemble column (`ens_fwd_ret`)

**Files:**
- Modify: `ml/features/forecast_features.py`
- Test: `tests/ml/features/test_forecast_ensemble.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/features/test_forecast_ensemble.py
import pandas as pd
from ml.features.forecast_features import merge_forecast_features, FORECAST_FEATURES

def test_ens_fwd_ret_is_mean_of_tsfm_and_kronos():
    base = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"], "ret_21d": [0.1]})
    tsfm = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"],
                         "tsfm_fwd_ret": [0.04], "tsfm_uncert": [0.02]})
    kron = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "symbol": ["AAA"], "kronos_fwd_ret": [0.06]})
    out = merge_forecast_features(base, [tsfm, kron])
    assert "ens_fwd_ret" in out.columns and "ens_fwd_ret" in FORECAST_FEATURES
    assert abs(out["ens_fwd_ret"].iloc[0] - 0.05) < 1e-9   # mean(0.04, 0.06)
```

- [ ] **Step 2: Run, confirm FAIL** (`ImportError: FORECAST_FEATURES` / no `ens_fwd_ret`).

- [ ] **Step 3: Implement** — in `forecast_features.py`: add `FORECAST_FEATURES = TIMESFM_FEATURES + KRONOS_FEATURES + ["ens_fwd_ret"]`, and at the END of `merge_forecast_features` (after merging tsfm+kronos onto base), add:

```python
    if "tsfm_fwd_ret" in merged.columns and "kronos_fwd_ret" in merged.columns:
        merged["ens_fwd_ret"] = merged[["tsfm_fwd_ret", "kronos_fwd_ret"]].mean(axis=1)
    else:
        merged["ens_fwd_ret"] = np.nan
```

- [ ] **Step 4: Run, confirm PASS.**

- [ ] **Step 5: Commit**

```bash
git add ml/features/forecast_features.py tests/ml/features/test_forecast_ensemble.py
git commit -m "feat(momentum): ens_fwd_ret = mean(TimesFM, Kronos) forecast column"
```

---

### Task 4: Wire the trainer (NIFTY benchmark) + update the smoke assertion

**Files:**
- Modify: `ml/training/trainers/momentum_lambdarank.py`
- Modify: `scripts/runpod/smoke_momentum_local.sh`

- [ ] **Step 1: Load the benchmark in `_build_dataset`** — after `panel = load_ohlcv(symbols, ...)`, load NSEI and pass it through:

```python
    # NIFTY benchmark for RS features (NSEI cached as data/cache/NSEI_10y.csv)
    try:
        bench = load_ohlcv(["NSEI"], cfg.start, cfg.end)
        bench = bench[["date", "close"]] if not bench.empty else None
    except Exception:
        bench = None
    feats = build_momentum_features(panel, benchmark=bench)
```

(If `cfg.with_forecasts`, the existing forecast-merge block stays; ensure `feature_cols` includes the new RS/forecast columns — they come from `MOMENTUM_FEATURE_ORDER` + `FORECAST_FEATURES` automatically.)

- [ ] **Step 2: Make `feature_cols` include the ensemble forecast col** — where forecasts merge, change the forecast feature list from the 3-col set to `FORECAST_FEATURES` (import it).

- [ ] **Step 3: Update `scripts/runpod/smoke_momentum_local.sh`** — the assertion `m["n_features"] == 21` is now wrong. Change to:

```python
    assert m["n_features"] >= 70, f"expected the full feature set (>=70), got {m['n_features']}"
```

- [ ] **Step 4: Quick CPU dataset-build check** (no full training):

```bash
KMP_DUPLICATE_LIB_OK=TRUE FORECAST_DEVICE=cpu PYTHONPATH="$HOME/.kronos_repo:$(pwd)" python3 -c "
from datetime import date
from ml.training.trainers.momentum_lambdarank import _build_dataset, MomentumConfig, cached_universe
from ml.features.momentum_features import MOMENTUM_FEATURE_ORDER
cfg = MomentumConfig(with_forecasts=False, start=date(2021,1,1), end=date(2026,2,1))
X, cols = _build_dataset(cfg, cached_universe(limit=8))
print('rows', len(X), 'feature cols', len(cols), 'has RS', 'rs_index_63' in cols)
assert len(cols) >= 60 and 'rs_index_63' in cols
print('OK')
"
```
Expected: `feature cols >= 60`, `has RS True`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add ml/training/trainers/momentum_lambdarank.py scripts/runpod/smoke_momentum_local.sh
git commit -m "feat(momentum): wire NIFTY benchmark + full feature set into the trainer"
```

---

### Task 5: Debug the full DL stack on Mac CPU

**Files:** none (verification task).

- [ ] **Step 1: Run the full-stack local smoke** (TimesFM + Kronos + the ~75 features + LGBM, tiny universe):

```bash
bash scripts/runpod/smoke_momentum_local.sh
```
Expected: `✓ FULL STACK GREEN` with `n_features >= 70` and `stack` = `price+forecasts (TimesFM+Kronos)`. This proves the whole pipeline (including the GPU forecasters on CPU) runs error-free before any RunPod spend.

- [ ] **Step 2: If it fails**, debug iteratively (common: a feature producing all-NaN drops every row; an `.apply()` crashing on a single symbol; a forecast-merge key mismatch). Fix in the relevant file, re-run. Do NOT proceed to RunPod until this is green.

- [ ] **Step 3: Full backend suite still green** (features are pure additions; serving not yet changed):

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest -q
```
Expected: 882+ passed (the new feature tests add to it).

- [ ] **Step 4: Commit** any fixes from Step 2 with a clear message.

---

## PHASE 2 — post-training (Tasks 6–7): do these AFTER the RunPod run produces the new artifact

Tasks 1–5 are the **build + Mac-CPU debug** (no GPU, no retrained model required). Tasks 6–7 are the **post-training** phase — they depend on the retrained model's new `feature_order.json` and the weekly forecast-cache. Do NOT start Task 6 until Task 7's artifact exists.

### Task 7 (run first of Phase 2): RunPod full training

**Files:** none (the user launches this on RunPod with GPU).

- [ ] **Step 1:** On a fresh RunPod GPU pod: `git clone -b feat/mldl-4engine https://github.com/Ri2506/quantx.git && cd quantx && bash scripts/runpod/train_momentum_gpu.sh` (optionally `UNIVERSE_LIMIT=120` for a cheaper first run). This runs the full ~75-feature + TimesFM + Kronos training on GPU and writes `artifacts/models/momentum_lambdarank/{momentum_lambdarank.txt, feature_order.json, metrics.json}`.
- [ ] **Step 2:** Compare `metrics.json` (rank_ic_mean / rank_icir / decile_spread) against the current base (0.0843 / 2.02 / 0.0369). Read LightGBM feature importance; **prune** features with ~zero gain (drop from `MOMENTUM_FEATURE_ORDER`, re-run, re-validate) toward the spec's tighter end. Download the artifact and commit the new `feature_order.json` + `metrics.json`.
- [ ] **Step 3:** Add a weekly scheduler job that runs `forecast_features` over the universe and **persists** the 4 forecast columns (`tsfm_fwd_ret, tsfm_uncert, kronos_fwd_ret, ens_fwd_ret`) keyed by `(date, symbol)` to a `momentum_forecasts` table/parquet — so serving reads the cache instead of running GPU per request. (This is the serving-cost mechanism §Notes references.)

### Task 6 (after Task 7): update `MomentumEngine` serving to the new contract

**Files:**
- Modify: `backend/ai/signals/engines/momentum.py`
- Test: `tests/ml/serving/test_momentum_engine.py` (extend)

- [ ] **Step 1: Write the test** — extend the existing `MomentumEngine` test. The fixture loader from `test_ranks_and_levels_and_outputs` must also return an `"NSEI"` row-set when symbols include it; assert the engine builds the full feature set and `run()` still returns ranked `MomentumSignal`s with the new (≥70) feature contract, and that a missing forecast cache (None) does not crash (LightGBM scores with NaN):

```python
def test_engine_full_contract_with_benchmark(monkeypatch):
    # symbols include NSEI; fake_loader returns a ramp for each incl NSEI.
    # inject a _FakeBooster whose .feature_name() == the new MOMENTUM_FEATURE_ORDER
    # so the contract check passes; assert len(run(top_n=3)) == 3 and each is a
    # MomentumSignal with finite entry/SL/target. (Concrete fixture mirrors the
    # existing test_ranks_and_levels_and_outputs; reuse its _ramp_ohlcv + offsets,
    # add "NSEI" to the symbol list + offsets.)
    pass
```

(Write the concrete fixture by copying `test_ranks_and_levels_and_outputs` and adding `"NSEI"` to `syms`/`offsets` + a `_FakeBooster.feature_name()` returning `list(MOMENTUM_FEATURE_ORDER)`.)

> **Phase-1 status (done early):** the benchmark wiring already shipped in
> Task 2 — `MomentumEngine.run()` loads NSEI via `self._load_ohlcv(["NSEI"], …)`
> (fail-soft) and passes it to `build_momentum_features(panel[cols],
> benchmark=bench)`. The live 18-feature model ignores RS, so this is a no-op
> for it today. **Two Phase-2 fixes remain here:** (a) switch the engine's
> benchmark source from `self._load_ohlcv(["NSEI"])` (routes to yfinance
> `NSEI.NS` → 404) to `ml.data.benchmark.load_nifty_benchmark` — the same
> offline-cache loader the trainer uses; (b) the cache only runs to its last
> ingested date, so for live serving the most-recent bars get NaN RS — refresh
> NSEI in the weekly job (Task 7 Step 3) or read a current index quote, else RS
> is stale-to-NaN at the serving bar.

- [ ] **Step 2: Implement** — in `MomentumEngine.run()`: replace the benchmark
  load with `load_nifty_benchmark(start, end)` (per the note above). Read the
  most-recent persisted forecast columns (Task 7 Step 3 table) and left-merge by
  `(date, symbol)`; if the cache is absent the forecast columns are NaN and
  LightGBM scores fine. The engine already scores `self._feature_order` from the
  artifact's `feature_order.json`, so the new (≥70) contract flows through
  automatically once the new artifact lands — no hard-coded count to change.
  Keep honest-empty + per-symbol skip.

- [ ] **Step 3: Run the engine tests + the live serve smoke**

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -m pytest tests/ml/serving/test_momentum_engine.py -v
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$(pwd)" python3 -c "from backend.api.signals_routes import _compute_momentum; print(_compute_momentum(top_n=5)['status'])"
```
Expected: tests pass; serve smoke `status: ok` (the new artifact's `feature_order.json` matches the engine's contract).

- [ ] **Step 4: Commit**

```bash
git add backend/ai/signals/engines/momentum.py tests/ml/serving/test_momentum_engine.py
git commit -m "feat(momentum): serving reads NIFTY benchmark (RS) + weekly-cached forecasts (new contract)"
```

---

## Notes
- **NaN policy:** the trainer drops rows with NaN in any feature column (warmup-controlled). With 252-bar windows, the first year per symbol is warmup — fine for a multi-year history. Watch that no single feature is *all*-NaN (would drop everything) — the Task-1 finite test guards this.
- **No leakage:** every feature uses only past data (`shift`, trailing `rolling`); cross-sectional ranks are within-date. RS uses contemporaneous benchmark (a feature, not a label) — correct.
- **Serving cost:** forecasters run WEEKLY (`forecast_stride=5`) on RunPod and persist their 4 columns; serving reads the cache — it never runs TimesFM/Kronos per request.
- **NSEI is an index, not a yfinance equity ticker:** `load_ohlcv(["NSEI"])`
  routes through the production provider → yfinance `NSEI.NS` → 404. The
  benchmark MUST come from the offline cache via
  `ml.data.benchmark.load_nifty_benchmark` (Task 4), which normalizes the date
  exactly like `FreeDataProvider` so it merges onto the equity panel. Built +
  tested in Phase 1.
- **Sector RS follow-on:** add `rs_sector_*` once sector indices (NIFTY BANK/IT/AUTO/…) are ingested; the `build_momentum_features` benchmark hook is the extension point.
- **30/60d label heads — DEFERRED (spec §5.1 "30/60d added as heads"):** this plan keeps the single 20d decile-relevance ranking target (the core momentum signal). The 30/60d horizons are a multi-head/multi-target enhancement (either separate LambdaRank models or a multi-output setup) — a follow-on after the 20d full-feature model validates, to avoid coupling the feature build with a training-topology change in one pass. The output schema (`MomentumSignal`) already has room (`expected_return`) and the spec's §4.7 `expected_return_{...}` per-horizon fields can be added then.

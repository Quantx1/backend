# Quant X — Four-Engine ML/DL Trading Signal System (Master Design Spec)

**Date:** 2026-06-15
**Status:** Design of record — approved-pending-user-review
**Supersedes / reverses:** v1 scope trim (momentum cut), PR-M/PR-N removal (TimesFM removed),
momentum_chronos/chronos2_macro retirement. Those cuts are intentionally **reversed** by this spec.
**Built on:** `docs/ML_DL_DEEP_AUDIT_2026_06_15.md` — every foundation fix here closes a numbered audit finding.

---

## 0. One-paragraph summary
Build four **style-specific signal engines** — Momentum, Swing, Positional, Intraday — each a small
stack of forecaster model(s) feeding a LightGBM ranking/decision layer, on a shared foundation
(TrueData data plane + Qlib/TA-Lib/MLFinLab feature & label factory + a fixed train→serve→promote
contract + a risk engine that derives trade levels). Models emit **absolute** signals (expected
return / probability / confidence / cross-sectional rank) — **no benchmark comparison**. The build
fixes, inline, the train/serve skew and dead gates the 2026-06-15 audit found, so new models actually
serve what was trained and can't ship on noise.

## 1. Goals & non-goals
**Goals**
- Four production engines, each with real trained ML/DL models, wired into the existing app
  (`/signals/{intraday,swing,positional}` + new `/signals/momentum`) and scheduler.
- One reusable foundation: data plane, feature factory, labeling, purged CV, serving contract,
  registry, risk engine, output schema, unified training runner.
- Correctness first: no train/serve skew, no leakage, real promote gates, fail-loud, models never
  emit trade levels.

**Non-goals (v1)**
- Options RL, social/alt-data, paid news bodies, AUM/portfolio optimization.
- Per-user broker execution changes (AutoPilot consumes these signals; its executor is separate).

## 2. System architecture (layers)
```
            ┌──────────────────────── DATA PLANE ────────────────────────┐
TrueData (TD_hist/TD_live) ──┐                                            │
screener.in (fundamentals)   ├─► data_loader (uniform OHLCV/options/fund) │
nselib (FII/DII)             │   + storage (pg candles, intraday bars,    │
news pipeline (sentiment) ───┘     option snapshots, PIT fundamentals)    │
            └──────────────────────────────┬─────────────────────────────┘
                                            ▼
        FEATURE FACTORY  (Qlib factors + TA-Lib/pandas-ta + custom)  — one builder per engine,
                                            ▼                          used identically train & serve
        LABELING  (MLFinLab triple-barrier [intra-bar fixed] + sample-uniqueness + fwd-return quantile)
                                            ▼
        MODEL LAYER  per engine:  forecaster(s) ──► LightGBM ranker/classifier
                                            ▼
        VALIDATION + PROMOTE GATE  (purged+embargoed walk-forward; real ranking/financial metric)
                                            ▼
        REGISTRY + CONTRACT  (filename+format+feature-order sidecar; smoke-load-before-promote)
                                            ▼
        SERVING ENGINES  (MomentumEngine / SwingEngine / PositionalEngine / IntradayEngine)
                                            ▼
        RISK ENGINE  (expected return + ATR → entry / SL / target)
                                            ▼
        API + FRONTEND  (/signals/*)   +   SCHEDULER (per-engine cadence)
```

## 3. Data plane
### 3.0 Provider strategy — pluggable, free-data-first
The data layer is a **pluggable provider** behind one `data_loader` interface. Two providers:
- **`FreeDataProvider` (v1, available NOW):** yfinance + nselib/nsepy/jugaad bhavcopy + pg `candles`
  cache (323 cached 10-yr daily CSVs + 2,366-symbol universe) + screener.in fundamentals + nselib
  FII/DII + news pipeline. **Sufficient to train Momentum, Swing, Positional today** (all daily-OHLCV
  based). No credentials required.
- **`TrueDataProvider` (drop-in later):** enabled when `TRUEDATA_LOGIN`/`TRUEDATA_PASSWORD` arrive.
  (a) upgrades EOD fidelity for engines 1–3, (b) **unblocks Intraday (M4)** — minute bars + option
  chain/OI/Greeks/VIX, which free sources cannot provide reliably.
Switching providers is config only (`DATA_PROVIDER=free|truedata`), no engine rework.
**Consequence for build order:** M0–M3 proceed on free data now; **M4 (Intraday) is gated on TrueData
login** — and it is last in the order regardless, so nothing stalls.

### 3.1 TrueData adapter — `src/backend/data/providers/truedata_provider.py` (new, drop-in)
- Wraps `truedata` v7: `TD_hist(login, pwd)` for backfill/EOD/intraday history; `TD_live(login, pwd)`
  for realtime ticks/1-min bars/option chain/Greeks.
- Methods: `get_history(symbol, start, end, bar_size)`, `get_n_bars`, `get_option_chain(symbol, expiry,
  chain_length, greek=True)`, `get_bhavcopy`, live callbacks (`trade_callback`, `one_min_bar_callback`,
  `greek_callback`, `bidask_callback`).
- Bar sizes: `eod`/`week`/`month` (momentum/swing/positional), `1/3/5/15 min` + `tick` (intraday).
- Credentials in `.env` (`TRUEDATA_LOGIN`, `TRUEDATA_PASSWORD`) — never committed.
- **Fail-loud:** on auth/empty-response, raise — never return an empty frame (audit: `production_ohlcv`
  empty-frame masking).
### 3.2 Auxiliary sources (existing, reused)
- **Fundamentals:** screener.in (current snapshot) → `ml/data/fundamentals_pit.py`. **PIT capture forward**:
  snapshot daily into a `fundamentals_pit` table so history deepens from today (positional decision).
- **FII/DII:** nselib → `ml/data/fii_dii_history.py`.
- **Sentiment:** existing news-intelligence pipeline → Mood/materiality features.
### 3.3 Storage
- Daily candles: existing pg `candles` table (corp-action adjusted; fix BOM/symbol bug already done).
- New: `intraday_bars` (symbol, ts, ohlcv, oi), `option_chain_snapshots` (symbol, expiry, strike,
  type, ltp, oi, iv, greeks, ts), `fundamentals_pit` (symbol, asof_date, factors…).
### 3.4 Universe — `ml/data/liquid_universe.py` (fix)
Top-N by 30-day median ADV, price floor, exclude SME/illiquid/circuit. **Fix audit bug:** source from
our pg OHLCV (not yfinance), fail loud instead of silent NIFTY-200 fallback.
### 3.5 Uniform loader — `ml/data/data_loader.py` (new)
Single interface every trainer + serving engine calls: `load_ohlcv(symbols, start, end, freq)`,
`load_options(...)`, `load_fundamentals(...)`. Guarantees identical data shape train vs serve.

## 4. Shared foundation
### 4.1 Feature factory
- `ml/features/factory.py` orchestrates: **Qlib** Alpha158-style factors (returns/vol/momentum/MA/
  cross-sectional ranks), **TA-Lib/pandas-ta** indicators, **custom** (options/flows/regime/calendar).
- **One feature builder function per engine** (`momentum_features`, `swing_features`,
  `positional_features`, `intraday_features`), imported by BOTH the trainer and the serving engine —
  train/serve parity by construction (audit: kills the LGBM/Qlib/TFT skew class).
### 4.2 Labeling — `ml/labeling/` (wire the dead-but-good code)
- **MLFinLab triple-barrier**, FIXED to check intra-bar High/Low (audit finding), + **sample-uniqueness
  weights** (AFML Ch.4). For rankers: forward-return → **quantile relevance grades** (absolute, no
  benchmark). For intraday: time-boxed triple-barrier (15/30/60m).
### 4.3 Validation & promote gate — `ml/training/` + `ml/eval/`
- **Purged + embargoed walk-forward CV**; embargo sized in **trading days = label horizon** (audit:
  fixes positional-row embargo), date-grouped so a date's symbols never split the boundary.
- Promote gate uses a **real metric per model type**: rankers → NDCG@k / rank-IC OOS; classifiers →
  OOS Sharpe/PF + DSR/PBO (un-hardcode PBO — audit). **Fix runner gate plumbing** so the
  `<trainer>_quality_pass` key matches and no model promotes on `--promote` alone (audit: Qlib/HMM
  dead-gate findings).
### 4.4 Serving contract — `src/backend/ai/registry/` (fix)
- **Per-model contract:** exact artifact filenames + a `feature_order`/`dataset_params` sidecar.
- **Smoke-load-before-promote:** after training, round-trip the artifact through the *production* engine
  predictor; promotion blocked on failure or feature-order mismatch.
- **Remove `resolve_model_file` stale-disk last-resort** (compat.py:88-90) so a missing registered file
  fails loud instead of silently serving an old artifact (audit: TFT stale-checkpoint finding).
### 4.5 Registry — registry-first resolution, `model_versions` rows, one `is_prod` per model.
### 4.6 Risk engine — `src/backend/trading/risk_engine.py` (new/extend)
Models emit expected return + confidence; risk engine derives **entry / SL / target** from ATR
(e.g. SL = 1.5·ATR, TP = 3·ATR), per-style. **Models never emit levels** (audit: TFT `_derive_levels`).
### 4.7 Output schema (absolute, no benchmark) — `src/backend/ai/signals/types.py` (extend with `style`)
- Momentum: `expected_return`, `rank`, `percentile`, `top_decile_prob`, `confidence`.
- Swing: `expected_return_{5,10,20}d`, `rank`, `percentile`, `confidence`, `expected_hold_days`.
- Positional: `expected_return_{1,3,6}m`, `rank`, `percentile`, `confidence`, factor sub-scores.
- Intraday: `signal {buy/sell/no-trade}`, `prob_up`, `expected_return_{15,30,60}m`, `confidence`, `rank`.
### 4.8 Unified training runner — each engine seeds trainer modules into `ml/training/trainers/`;
GPU runs on RunPod (existing). `python -m ml.training.runner --only <engine> --promote`.

## 5. Engine specs
> Every engine: forecaster(s) generate forecast features → LightGBM produces the final signal/rank.
> Models output absolute signals; risk engine adds levels; RS-vs-index/sector kept as **features only**.

### 5.1 Momentum  (build #1 — reference)
- **Models:** TimesFM + Kronos (forecast features) → **LGBM LambdaRank**.
- **Data:** TrueData EOD OHLCV + index/sector series (RS features only).
- **Labels:** absolute forward-return over 20d → decile relevance grades (30/60d added as heads).
- **Features (~40-80):** multi-horizon momentum, momentum quality (consistency/accel/decay/vol-adj),
  trend/MA alignment, volume confirmation, volatility/risk, RS-vs-index/sector, cross-sectional
  percentile ranks by date, + TimesFM/Kronos forecast columns.
- **Rebalance:** weekly. **Outputs:** §4.7. **Frontend:** new `/signals/momentum` (mirrors swing).

### 5.2 Swing  (build #2)
- **Models:** TFT + Chronos (forecast features) → **LGBM Ranker**.
- **Data:** TrueData daily OHLCV + screener.in fundamentals (snapshot) + nselib FII/DII + news sentiment.
- **Labels:** absolute forward-return 5/10/20d (no NIFTY) → ranker relevance.
- **Features (~120-250):** price/return, trend, volatility, momentum/RS, cross-sectional, fundamentals,
  ownership/flow, sentiment/event, regime, calendar (per blueprint) + TFT/Chronos forecasts.
- **Retrain:** weekly. **Frontend:** existing `/signals/swing` (replace rule-based with ML ensemble).

### 5.3 Positional  (build #3 — price-first)
- **Models:** Kronos + TimesFM (forecast features) → **LGBM Ranker**.
- **Data:** TrueData long-horizon EOD/weekly + light screener.in fundamentals snapshot; **PIT capture
  forward** to deepen quality/value/growth factors over time.
- **Labels:** absolute forward-return 1/3/6m → ranker relevance.
- **Features:** long-horizon price/momentum, quality/value/growth (as available), ownership/liquidity,
  sector relative-strength, regime + Kronos/TimesFM forecasts.
- **Retrain:** monthly. **Frontend:** existing `/signals/positional`.

### 5.4 Intraday  (build #4 — data-unblocked by TrueData)
- **Models:** PatchTST (sequence forecaster) → **LGBM** (buy/sell/no-trade + prob).
- **Data:** TrueData 1/3/5/15-min bars + option chain/OI/Greeks + India VIX + breadth.
- **Labels:** time-boxed **triple-barrier** at 15/30/60m horizons (TP/SL/time).
- **Features (~60-150 + 20-60 sequence channels):** raw OHLCV sequence channels for PatchTST; summary
  features (RSI/MACD/ADX, short returns, vol regime, OI delta, PCR, VWAP distance, breadth, session
  flags) for LGBM.
- **Inference:** intraday loop (every 5 min, market hours) reusing existing scheduler scan slot.
  **Frontend:** existing `/signals/intraday` (replace rule-based scanner with ML).

## 6. Model framework choices
- **neuralforecast** → PatchTST (intraday) + TFT (swing) under one API (heals the audit's TFT
  framework split; one serving path).
- **chronos-forecasting** (Amazon Chronos-2) → swing auxiliary forecaster.
- **TimesFM** (Google) → momentum + positional. Install via git (Python-3.12 wheel caveat noted).
- **Kronos** (finance-specific, 12B candlesticks) → momentum + positional. Clone + PYTHONPATH.
- **LightGBM** → all final rankers/classifiers (LambdaRank for momentum).
- **Qlib** → feature factory + research backtest. **MLFinLab/mlfinpy** → labeling, sampling, purged CV.

## 7. Integration with existing app
- Generalize `signals/types.py` with a `style`/`horizon` enum; persistence + API already per-style.
- Tier gating via existing `FEATURE_MATRIX` (which styles/limits per Free/Pro/Elite).
- Scheduler: per-engine cadence (intraday 5-min; momentum/swing weekly; positional monthly) reusing
  existing job registration patterns.
- AutoPilot consumes the new signals through the same registry/contract (no executor rewrite here).

## 8. Build order & milestones
- **M0 — Foundation + data plane:** TrueData adapter + loader + storage tables, feature factory
  skeleton, MLFinLab labeling wired + triple-barrier fix, purged-CV + promote-gate plumbing fix,
  serving contract + smoke-load + stale-fallback removal, risk engine, output schema, `style` enum.
- **M1 — Momentum** (reference end-to-end).
- **M2 — Swing.**
- **M3 — Positional.**
- **M4 — Intraday.**
  *(Order follows your blueprint. Note: Intraday is data-clean on TrueData and could move ahead of
  Swing/Positional if we want a quick high-value win — flag for decision at M1 close.)*

## 9. Testing strategy (TDD)
Per engine + foundation: (1) **feature parity** — train builder == serve builder on same input;
(2) **label correctness** — intra-bar barrier touches; (3) **purged-CV no-leakage** — embargo covers
horizon; (4) **contract smoke-load** — artifact loads through prod predictor with matching feature
order; (5) **e2e** — rank/score a small universe; (6) **backtest realism** — costs/slippage/fills.

## 10. Audit fixes incorporated (traceability)
Train/serve skew (feature builders shared; contract + smoke-load) · stale artifact (remove
last-resort disk fallback) · dead promote gates (key-name fix, real metric, un-hardcode PBO,
block promote-on-flag-alone) · leakage (intra-bar barrier, day-sized embargo, date-grouped folds,
no fundamentals broadcast — PIT capture) · fail-loud (TrueData + universe) · no model-derived levels
(risk engine) · wire dead-but-good MLFinLab/Qlib machinery.

## 11. Open items / risks
- **TrueData commercial/redistribution licence** — confirm plan permits paid-SaaS display (ties to
  Data-Licensing Path A). Building proceeds; surface before public launch.
- **Positional PIT fundamentals depth** — shallow today; deepens via forward capture.
- **GPU budget/time** — five foundation models × four engines is multi-hour RunPod training; sequenced
  per milestone, not one run.
- **TimesFM Python-3.12 install**, **Kronos clone path** — environment setup tasks in M0.

## 12. Explicitly deferred
Options RL, social/alt-data, paid news bodies, AUM/portfolio optimization, multi-leg options signals.

# Quant X — ML/DL Master Plan, Universe, Observability & Production Reorg (2026-06-16)

# Quant X — MASTER PLAN: 4-Engine ML/DL System

**Date:** 2026-06-16 · **Author role:** Lead Engineer · **Status:** the single document that ties together the per-model training plan, the universe strategy, the observability/evaluation design, and the production repo reorganization. Everything below is grounded in the live tree (verified, not assumed): the bidirectional `ml ↔ src.backend` import cycle, the standalone `train_momentum()` (not yet a `Trainer` subclass), the missing `truedata_provider.py`, the GPU-gated `forecast_features.py`, the hardcoded `/ml/regime` stub, the unrouted `list_versions`, the three artifact roots, and the absent `pyproject.toml`.

---

## 0. The one-paragraph version (read this first)

We have **4 trading engines** (Momentum, Swing, Positional, Intraday). Each engine is **one trainer** that internally runs **forecaster models → a LightGBM decision model**, producing **one shippable artifact**. We train the engines **independently, sequenced milestone by milestone** (M0→M4), NOT in one giant simultaneous run, because the forecasters are GPU-bound and would serialize anyway. The **universe** is a 5-tier ADV-ranked structure (XL/L/M/B/U) so each engine sees the right liquidity band — three of four engines train **today on free data** (~230k–322k cross-sectional rows); Intraday waits for TrueData. We **see** everything through the registry we already own (`model_versions` + B2) plus four small additions — live logs, a persisted `TrainReport` (JSON + markdown + plots) per run, version-comparison endpoints, and a "Model Detail" admin page — **no MLflow**. The **repo** gets reorganized into three concentric rings (`core/` → `ml/` + `backend/`) enforced by import-linter, which makes "train==serve feature parity" structurally true instead of a convention people forget. The whole thing is gated on two external blockers: **RunPod GPU funds** (for the forecasters in M1c/M2/M3) and **TrueData** (for all of M4).

---

## 1. How we train each model — end to end, per model

### 1.1 The mental model: forecasters are FEATURE generators, the LightGBM is the SIGNAL

The crux that everything else hangs on: within each engine there is a strict **producer→consumer** chain, but it is **inside one trainer**, not across trainers.

```
ENGINE = one trainer (one trainers/*.py, one --only target, one model_versions row)
  ├─ forecaster A  (TimesFM / Kronos / TFT / Chronos-2 / PatchTST)  ─┐
  ├─ forecaster B  (zero-shot or DL-fit)                            ─┤→ forecast FEATURE COLUMNS
  └─ price + cross-sectional + RS features                          ─┴→ LightGBM ranker/classifier → SIGNAL (the artifact)
```

- The DL **forecasters emit columns** (`tsfm_fwd_ret`, `tsfm_uncert`, `kronos_fwd_ret`, TFT/Chronos quantile spreads, PatchTST sequence forecasts).
- The **LightGBM consumes those columns alongside price/cross-sectional features** and is the model that emits the user-facing signal.
- TimesFM, Kronos, Chronos-2 are **zero-shot / pretrained** — no fit loop, no standalone artifact, no promote gate. They are **inference calls inside `build_features`**, exactly as the working reference trainer `ml/training/trainers/momentum_lambdarank.py` already calls `timesfm_forecast_features` + `kronos_forecast_features` + `merge_forecast_features` when `with_forecasts=True`.
- TFT and PatchTST **are** fit — but **inside their engine trainer's own train step**, before the LightGBM step. Still one trainer, one artifact bundle.

The reference trainer already proves the whole spine end-to-end (**OOS rank-IC 0.084**): `load_ohlcv → build_momentum_features (train==serve) → forward_return_quantile_labels (absolute) → purged_walk_forward_by_date (day-sized embargo) → LGBMRanker(lambdarank) → OOS rank-IC + decile spread per fold → final fit + feature_order.json sidecar`. Every model below slots into that same spine.

### 1.2 MOMENTUM engine — `ml/training/trainers/momentum_lambdarank.py` (EXISTS, working; must become a `Trainer` subclass — see §1.7)

**TimesFM** (`google/timesfm-2.5-200m-pytorch`) — *forecaster, GPU.* Daily CA-adjusted close from `load_ohlcv`, context 512 bars, min 252 history. Produces `tsfm_fwd_ret = point[-1]/last_close − 1` and `tsfm_uncert = (q90−q10)/last_close` at 20d, one row per (rebalance-date, symbol), forward-filled to daily via `merge_forecast_features` (as-of backward). **No labels, no CV, no fitting** — zero-shot. Re-inferred each weekly retrain at `stride=5`. Evaluated only **indirectly** (the incremental OOS rank-IC the column adds to the LGBM + EDA forward-IC). Verified: `ml/features/forecast_features.py:62-63` raises `RuntimeError` if CUDA absent.

**Kronos** (`NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base`) — *forecaster, GPU.* Full OHLCV panel, context 400, min 252. Produces `kronos_fwd_ret = pred_close[horizon]/last_close − 1` (finance K-line foundation model, sampled inference `T=1.0, top_p=0.9, sample_count=1`). Zero-shot, no fitting. Live adapter is `forecast_features.kronos_forecast_features` (the old `kronos_features.py` was deleted). Cloned at a pinned commit on `KRONOS_PATH`. Verified: same CUDA raise at `forecast_features.py:125-126`.

**LGBM LambdaRank — THE Momentum artifact** — *decision model, CPU.* Data: `load_ohlcv(cached_universe(), 2020-01-01..2026-03-01)` over the liquid universe (top-N by 30d median ADV; fix to source ADV from pg `candles`, fail-loud). Features: the 13 columns in `MOMENTUM_FEATURE_ORDER` (multi-horizon returns 5/10/21/63/126/252d, mom-consistency/accel, vol-adj momentum, dist-SMA-50/200, above-63d-high, realized vol, drawdown-252, rel-volume, OBV slope, `xs_rank_ret_21/63`) + (when `with_forecasts=True`) the 3 forecast columns + RS-vs-index/sector (features only, **no benchmark in the label**). Labels: `forward_return_quantile_labels(horizon=20, n_quantiles=10)` — **absolute** forward return → per-date decile relevance. CV: `purged_walk_forward_by_date(n_folds=5, test_days=63, embargo_days=20, train_days=378)` — date-grouped, embargo ≥ the 20d horizon. Per fold: fit `LGBMRanker(objective=lambdarank, metric=ndcg, ndcg_eval_at=[10,20])` with `group=` rows-per-date → predict OOS → rank-IC + decile spread → final fit on all rows. HPO: Optuna TPE (`ml/training/optuna_search.py`) on the same OOS rank-IC (opt-in; n_trials feeds the DSR null). Eval: OOS rank-IC mean/std/ICIR, decile spread, NDCG@10/20, DSR + PBO via CPCV, impact-cost-deducted top-decile long-short Sharpe. Output: `expected_return, rank, percentile, top_decile_prob, confidence`. Artifact: `momentum_lambdarank.txt` + `feature_order.json` + `metrics.json`. Weekly retrain.

### 1.3 SWING engine — `ml/training/trainers/swing_ranker.py` (BUILD NEW; refactor the TFT piece from the existing `tft_swing.py`)

**TFT** (neuralforecast Temporal Fusion Transformer) — *forecaster, GPU, IS fit.* Daily OHLCV + screener.in fundamentals snapshot + nselib FII/DII + news sentiment (thin → honest zero-fill, flagged in EDA). Trains on a **quantile-loss forecasting objective over 5/10/20d FORWARD returns**. Critical audit fix: it must **forecast the next bars, not reconstruct the last 5 observed bars**, and serve via **ONE framework — neuralforecast end-to-end** (delete the pytorch-forecasting `.ckpt` path). CV: purged walk-forward, embargo ≥ max horizon. HPO: light/capped (hidden size, heads, dropout, lr — GPU-expensive). Must pass **serve-smoke** (§1.7) before promotion. Produces 5/10/20d point + interval-width columns.

**Chronos-2** (Amazon `chronos-forecasting >=2.0`) — *auxiliary forecaster, CPU-capable.* Daily close (univariate), zero-shot, no fold loop, lowest-risk library. Produces forecast mean + interval-width columns. Pin the revision.

**LGBM Ranker — THE Swing artifact** — *decision model, CPU.* Data: daily OHLCV + fundamentals + FII/DII + sentiment. Features (~120-250): price/return, trend, vol, momentum/RS, cross-sectional ranks, fundamentals, ownership/flow, sentiment/event, regime, calendar + TFT/Chronos columns. **Author `ml/features/swing_features.py` with a frozen `SWING_FEATURE_ORDER`, mirroring `momentum_features.py` exactly** (groupby-transform, no `groupby.apply` on the single-symbol serve path). Labels: `forward_return_quantile_labels(horizon=5/10/20)` — absolute decile relevance. CV: `purged_walk_forward_by_date`, embargo ≥ 20d. Per-fold `LGBMRanker` with date groups → OOS → final fit; AFML sample-uniqueness weights from triple-barrier `t1`. HPO: Optuna on OOS rank-IC. Eval: rank-IC/ICIR, NDCG@k, decile spread, DSR/PBO, SPA, impact-cost-deducted Sharpe. Output: `expected_return_{5,10,20}d, rank, percentile, confidence, expected_hold_days`. Artifact: `swing_ranker.txt` + sidecars. Weekly.

### 1.4 POSITIONAL engine — `ml/training/trainers/positional_ranker.py` (BUILD NEW)

**Kronos** and **TimesFM** at long-horizon context (zero-shot, GPU, monthly cadence) → `kronos_fwd_ret`, `tsfm_fwd_ret`, `tsfm_uncert` at 1/3/6m. Same adapters as Momentum, long-horizon config.

**LGBM Ranker — THE Positional artifact** — *decision model, CPU.* Data: `load_ohlcv` long-horizon EOD/weekly + **PIT fundamentals captured forward** (do NOT broadcast a single snapshot across history — the audit's leakage fix; deepens over time). Features: long-horizon price/momentum, quality/value/growth (sparse, honestly zero-filled today), ownership/liquidity, sector relative-strength, regime + forecast columns. **Author `ml/features/positional_features.py` with frozen `POSITIONAL_FEATURE_ORDER`, shared train==serve.** Labels: `forward_return_quantile_labels(horizon=21/63/126 = 1/3/6m)` — absolute decile relevance. CV: `purged_walk_forward_by_date`, embargo ≥ 126d, wide test windows, fewer folds. Per-fold `LGBMRanker` → OOS → final fit; AFML weights. Output: `expected_return_{1,3,6}m, rank, percentile, confidence, factor sub-scores`. Monthly retrain.

### 1.5 INTRADAY engine — `ml/training/trainers/intraday_patchtst.py` (BUILD NEW; **HARD-GATED on TrueData**)

**Verified blocker:** `src/backend/data/providers/truedata_provider.py` does **not exist**, and `free_provider.py:106` raises `NotImplementedError` for intraday. This engine cannot start until TrueData lands.

**PatchTST** (neuralforecast / transformers) — *sequence forecaster, GPU, IS fit.* TrueData 1/3/5/15-min bars (+ OI). Trains on a forecasting objective over the bar sequence. CV: purged walk-forward in trading **sessions**, embargo ≥ longest horizon in bars. HPO: capped (patch length, stride, model dim). Produces 15/30/60m sequence forecasts + uncertainty column.

**LGBM Classifier — THE Intraday artifact** — *decision model, CPU.* Data: TrueData minute bars + option chain/OI/Greeks + India VIX + breadth. Features (~60-150 + 20-60 sequence channels): RSI/MACD/ADX, short returns, vol regime, OI delta, PCR, VWAP distance, breadth, session flags + PatchTST columns. **Author `ml/features/intraday_features.py`, shared train==serve.** Labels: **time-boxed intra-bar triple-barrier** at 15/30/60m via `triple_barrier_events(close, atr, high=, low=)` — High/Low touch (the M0 fix), conservative tie→stop; `t1` feeds sample-uniqueness weights + embargo. `LGBMClassifier` (buy/sell/no-trade) with AFML weights. HPO: Optuna on OOS PF/Sharpe (impact-cost-deducted; intraday costs dominate). Output: `signal {buy/sell/no-trade}, prob_up, expected_return_{15,30,60}m, confidence, rank`. Weekly retrain; 5-min inference loop during market hours.

### 1.6 Per-engine summary table

| Engine | Forecaster(s) → columns | Decision model (artifact) | Label | Cadence | GPU | Status |
|---|---|---|---|---|---|---|
| **Momentum** | TimesFM + Kronos | LGBM LambdaRank | abs 20d decile | weekly | forecasters yes, LGBM no | **trainer EXISTS** |
| **Swing** | TFT (fit) + Chronos-2 | LGBM Ranker | abs 5/10/20d decile | weekly | TFT yes, LGBM no | build new |
| **Positional** | Kronos + TimesFM | LGBM Ranker | abs 1/3/6m decile | monthly | forecasters yes, LGBM no | build new |
| **Intraday** | PatchTST (fit) | LGBM Classifier | intra-bar triple-barrier 15/30/60m | weekly | both yes | **gated on TrueData** |

### 1.7 The shared 9-stage spine every model rides (do NOT re-implement per trainer)

Each engine trainer subclasses `ml.training.base.Trainer`, drops into `ml/training/trainers/`, auto-registers via `discovery.py`, and delegates to a canonical `run_pipeline(ctx)` (BUILD NEW in `ml/training/pipeline.py`). It overrides only **3+1 hooks**: `build_features` (Stage 4), `build_labels` (Stage 3), `fit_model` (Stage 6), `predict_for_serve` (Stage 8b serve-smoke). Stages 1 (EDA hard-fail), 2 (quality gate), 5 (purged WFCV/CPCV), 8 (eval: rank-IC/NDCG + DSR/PBO + SPA + Almgren-Chriss impact), 9 (registry/promote) are 100% shared.

**The #1 audit fix is Stage 8b serve-smoke**: round-trip the trained artifact through the production predictor and block promote on any feature-order/shape/normalization mismatch. This is what guarantees "what trained is what serves" and kills the train/serve skew that affected all 4 prior PROD models.

Two structural fixes confirmed against the tree:
- **`momentum_lambdarank` is currently `train_momentum()`, NOT a `Trainer` subclass** (verified). It is invisible to the runner/registry/report. Converting it is a prerequisite for everything downstream.
- The runner promote-gate already exists (verified at `runner.py` ~lines 199-260): financial gate → `<trainer>_quality_pass` quality gate → safety-net (blocks when `primary_value is None` or `n_calibrations==0`) → Kelly. Fix the `<trainer>_quality_pass` key so the forecaster quality gate isn't dead.

### 1.8 Universal invariants (apply to all 11 models)

- One shared feature builder per engine, imported by trainer AND serving engine — train/serve parity **by construction**.
- Frozen `feature_order.json` sidecar beside every artifact; serve-smoke blocks promote on mismatch.
- Absolute labels, no benchmark — RS-vs-index/sector are FEATURES only.
- Day-sized, date-grouped embargo ≥ label horizon via `purged_walk_forward_by_date` (never positional-row embargo).
- Intra-bar High/Low triple-barrier wherever triple-barrier is used; `t1` feeds AFML sample-uniqueness weights + the embargo.
- Real promote gate per model type (rankers → rank-IC/NDCG OOS; classifiers → impact-cost-deducted Sharpe/PF + DSR/PBO); never promote on `--promote` alone.
- **Models never emit trade levels** — a separate `backend/trading/risk_engine.py` derives entry/SL/target from ATR.
- Fail loud — no empty-frame masking on data load; forecasters raise if CUDA absent (verified).

---

## 2. Separate vs together — the definitive answer

**Train each ENGINE as ONE self-contained trainer, and train the four engines INDEPENDENTLY, sequenced per milestone (M0→M1→M2→M3→M4), NOT one monolithic simultaneous run. The forecaster→LGBM dependency is resolved INSIDE each engine's trainer's feature-build step, never across trainers.**

Five reasons this is correct and not a compromise:

1. **Within an engine the dependency is intra-trainer, not inter-trainer.** The forecasters emit feature columns the LGBM consumes, so the LGBM can't fit until those columns exist. But the zero-shot forecasters (TimesFM/Kronos/Chronos-2) are *inference calls inside `build_features`* — exactly as `momentum_lambdarank._build_dataset` already works — with no separate fit loop to schedule. (TFT/PatchTST are fit, but inside their own engine trainer's train step before the LGBM step — still one trainer, one artifact.)

2. **The atomic schedulable unit is the ENGINE:** one `trainers/*.py`, one `--only` target, one `model_versions` row for the shipped LGBM.

3. **Across the four engines the signal models are independent** — shared foundation (loader/feature-factory/labeling/purged-CV/eval/registry) but disjoint features, labels, horizons, artifacts. There is **no cross-engine `depends_on`** for the four rankers. (`depends_on` in the runner is reserved for genuine ordering like a regime overlay or an ensemble.)

4. **Not one giant simultaneous run:** the GPU forecasters are multi-hour on a single RunPod GPU and would serialize on the one card anyway; per-milestone runs isolate install/data failures and keep the GPU budget tractable.

5. **Not 11 fully-separate runs:** promoting a zero-shot forecaster is meaningless (no artifact, no gate), and splitting forecaster from LGBM into separate runs would break the shared train==serve feature contract — re-introducing the exact skew the audit found.

### Training order

- **M0 — Foundation (NO training).** Build `ml/training/pipeline.py` (9-stage spine) + `serve_smoke.py` (Stage 8b) + `specs.py` + `baseline_drift.py`; wire EDA/quality/purged-CV/eval/registry; fix the `<trainer>_quality_pass` key + un-hardcode PBO; requirements hygiene (add neuralforecast, commit to neuralforecast-only TFT, un-comment timesfm, bump chronos>=2.0); RunPod env (TimesFM git `--no-deps`+jax pin; Kronos clone@pinned-commit on `KRONOS_PATH`). **Also convert `train_momentum()` into a `Trainer` subclass.**
- **M1a — Momentum price-only on CPU:** `python -m ml.training.runner --only momentum_lambdarank --promote` (`with_forecasts=False`). A real, shippable cross-sectional ranker today that exercises stages 1-9 + serve-smoke.
- **M1b — Close pure-code Momentum feature gaps:** add RS-vs-index (NSEI cached), then ingest sectoral indices + full-universe sector map for RS-vs-sector (features only).
- **M1c — Momentum specced stack on RunPod GPU:** install/verify TimesFM + Kronos adapters, re-run `--with-forecasts --promote` (forecaster inference → columns → re-fit LGBM).
- **M2 — Swing:** author `swing_features.py` + `swing_ranker.py`; TFT (neuralforecast, GPU) + Chronos-2 (zero-shot) → columns → LGBMRanker; `--only swing_ranker --promote`.
- **M3 — Positional:** author `positional_features.py` + `positional_ranker.py`; Kronos+TimesFM at 1/3/6m (zero-shot, GPU) → columns → LGBMRanker; **start PIT-fundamentals forward capture now**; `--only positional_ranker --promote`.
- **M4 — Intraday (gated on TrueData):** build `truedata_provider.py` + `intraday_features.py` + `intraday_patchtst.py`; PatchTST (GPU) → columns → LGBMClassifier on intra-bar triple-barrier; `--only intraday_patchtst --promote`.

**Within every engine** the fixed order is: forecaster feature emission [zero-shot inference OR DL fit] → merge into the frozen shared `feature_order` → purged-WFCV fit of the LGBM → OOS eval + DSR/PBO/SPA/impact → final fit → Stage-8b serve-smoke → registry promote gate.

### The unified runner (UNCHANGED in orchestration)

`ml/training/runner.py` + `ml/training/discovery.py` stay as-is: discovery → topo-sort on `depends_on` → per-trainer train/evaluate/register → promote gate → Kelly → safety-net → Sentry. Invoke per-engine: `python -m ml.training.runner --only <engine> --promote`, or `--all` once all four exist. **Note:** there is a second, dead orchestration path (`scripts/train_all_models.py` → `src/backend/ai/training.all_trainers()` which now returns `[]` — verified). Collapse to the one canonical runner (see §5 cruft).

---

## 3. Stock universe for big datasets

**Ship a 5-tier ADV-ranked universe (XL=50 / L=200 / M=350 / B=~500 / U=PIT survivorship pool).** Each engine gets the liquidity band its horizon needs. Three of four engines train **today on free data with zero new data dependencies**.

| Tier | Size | ADV floor | Price floor | Source pool | Engine | PIT |
|---|---|---|---|---|---|---|
| **XL** | 50 | ₹200 cr/day | ₹50 | nifty100 ∪ F&O | **Intraday** (TrueData) | yes |
| **L** | 200 | ₹25 cr/day | ₹30 | nse_all ∪ nifty500 | **Momentum** | yes |
| **M** | 350 | ₹10 cr/day | ₹30 | nse_all ∪ nifty500 | **Swing** | yes |
| **B** | ~500 | ₹5 cr/day | ₹50 | nse_all ∪ nifty500 | **Positional** | yes |
| **U** | dynamic | membership-only | — | B + `historical_universe_extras(as_of)` | survivorship spine (all) | required |

**Rationale.** Momentum decays in illiquid names + LambdaRank needs a clean fillable cross-section → L=200 (matches the reference trainer's `cached_universe`, OOS rank-IC 0.084). Swing's 5-20d alpha needs mid-caps → M=350 at ≥₹10 cr ADV (still fillable for Pro/Elite). Positional's 1-6m holds tolerate lower liquidity → B=500 for the deepest factor cross-section. Intraday edge lives only in the most liquid names → XL=50, all TrueData minute-bar + OI ingestion can support at 5-min cadence.

**The non-negotiables (PIT + survivorship).**
1. **Rank from pg `candles`, not live yfinance** — fix `ml/data/liquid_universe.py` (spec §3.4 audit fix; today it ranks ADV via live yfinance → non-reproducible + survivorship-leaky). Keep `strict=True` for trainers.
2. **Always pass `as_of_date`** at every walk-forward fold boundary — the 2021 fold uses the 2021 universe.
3. **Survivorship rejoin** via `historical_universe_extras(as_of)` from the hand-curated `DELISTED_NSE` registry (DHFL/RCOM/JETAIRWAYS/MINDTREE/HEXAWARE…) so the model *sees the losers*.
4. **Corp-action volume adjustment** (`adjust_volume_for_actions`) before ADV ranking so a bonus doesn't fake a liquidity jump.
5. **Freeze a `universe_snapshot.json`** (tier, as_of, ranked symbols + ADV) per training run, stored next to the artifact — reproducibility + audit.
6. **Floors** (`min_price`, `min_avg_volume`) reject penny/circuit names so signals are fillable.

**Dataset sizes (today, free data).** Window ≈ 1,416 cache bars; usable = bars − 252 warmup − label horizon; cross-sectional rows = usable × symbols.

| Engine | Tier | Cached symbols now | Usable bars/sym | Total rows | Label horizon |
|---|---|---|---|---|---|
| Momentum | L=200 | 200 (all cached) | 1,144 | **~229k** | 20d |
| Swing | M=350 | ~310 cached (~40 backfill) | 1,144 | **~355k** | 5/10/20d |
| Positional | B=500 | ~310 cached (~190 backfill) | 1,038 | **~322k** | 1/3/6m (126d) |
| Intraday | XL=50 | 0 (TrueData) | — | ~0.9–1.9M 5-min rows/yr | 15/30/60m |

These are healthy LightGBM ranker sizes. The binding constraint is **breadth (symbols × dates)**, which is exactly why widening the candidate pool to `nse_all ∪ nifty500` matters.

**Note on the live tree vs the universe doc snapshot:** I verified the cache currently holds **323 CSVs** (the doc said 310 — the cache has grown), and the tier files are **nse_all=551, nifty500=580, nifty100=105, nifty250=329, nifty50=51** (the doc's counts predate the latest ingest). The strategy is unchanged; the numbers above are the founder-facing planning figures and the backfill scope shrinks slightly with the larger cache.

**Data-sourcing path.** *Now (free):* `load_ohlcv(tier_symbols, start, end)` → cache CSV → pg `candles` → yfinance. *Two backfill jobs before the GPU run:* (1) refresh the ~4-month-stale cache (ends Feb 2026) to today; (2) backfill the Tier-B mid-caps not yet cached (`scripts/backfill_ohlc_pg.py` exists for the pg path; BOM-tolerant normalizer + corp-action volume adjust). *Later (TrueData):* flip `DATA_PROVIDER=truedata` (config-only) to unblock Intraday minute bars + OI/Greeks. No engine rework.

**Concrete to-do.** Add a `tiers.py` constant (XL/L/M/B + floors); fix `liquid_universe.py` to rank from pg; default `candidate_pool = nse_all ∪ nifty500`; emit/persist `universe_snapshot.json` per run + pass `as_of_date` at every fold; run the two backfill jobs before RunPod; grow `DELISTED_NSE` + `CORPORATE_ACTIONS` as backfill surfaces gaps.

---

## 4. How you SEE training, outputs, and results

**You already own ~70% of a real MLOps observability stack** — it just isn't assembled into one place and it stops at the database row. Today every model trains, evaluates with a serious metric library (rank-IC, DSR, PBO, SPA, impact-cost, Kelly), and writes a `model_versions` JSONB row + a `training_runs` history row that an admin page renders. Verified what exists: `ml/training/verbose.py` (line-buffered `flush=True` step output), `runner.py` (`RunReport` + per-trainer try/except + Sentry), `ml/eval/*` (the full metric library — the crown jewel), the registry (`src/backend/ai/registry/{model_registry,versions,b2_client}.py` with promote/shadow/retire/rollback), and the admin pages (`frontend/app/admin/{training,ml,model-performance}/`).

### What is MISSING (verified)

1. **No live log streaming surface** — the `_worker` daemon thread exists in `admin/training.py:202` but there's no `run_log` column and no streaming endpoint; you can't watch a fold print in the browser.
2. **No persisted human report per run** — verified zero `savefig`/`matplotlib`/`report.md`/`build_report` anywhere under `ml/`. Metrics live only as a JSONB blob; no plots ever.
3. **No version-comparison view** — `versions.list_versions()` exists in the registry (verified `versions.py:35` + `model_registry.py:231`) but is **not routed** in `admin/ml.py`. You can't diff v4 vs v5.
4. **Per-fold curves captured but never visualized** — the arrays are in the JSONB; the UI dumps the whole blob as `JSON.stringify`.
5. **`/ml/regime` returns hardcoded values** — verified verbatim: `bull / 0.87 / since 2026-03-01 / days_active 11 / empty history`. The dashboard is lying about regime.

### The design: registry stays the source of truth (NO MLflow)

MLflow would (a) introduce a **second competing registry** to keep in sync with `model_versions` (the truth already wired into serving via `ModelRegistry.resolve`), (b) require running/operating a server + backend store, and (c) not understand your domain gates (DSR/PBO/SPA/impact-cost). The gap closes with **4 small additions, ~1-2 days**, no new infra.

**Layer 1 — Live logs (build-new, thin).** In the `_worker` daemon thread, attach a `logging.Handler` + tee `sys.stdout` into a per-`run_id` ring buffer; flush to a new `training_runs.run_log` TEXT column + a `run.log` file in `out_dir` (uploaded to B2). Add `GET /admin/training/runs/{run_id}/log?since=<offset>`; the `RunRow` component (already polling every 3s) long-polls the tail while `status==running`. `verbose.py`'s `flush=True` is already line-buffered for this.

**Layer 2 — Uniform metrics contract (extend).** Standardize on `PipelineContext.metrics` from the 9-stage spine so `model_versions.metrics` is uniform across engines: always `primary_metric/value`, `*_per_fold` arrays, `rank_ic_mean/std/icir`, `decile_spread`, `deflated_sharpe`, PBO, `sharpe_mean`, `n_folds`, `feature_order`, and (new) `feature_importance` (LGBM `.feature_importances_` is free). Point every trainer at `wfcv.aggregate_fold_metrics` (already emits `_mean/_std/_per_fold`).

**Layer 3 — Persisted `TrainReport` per run (build-new, the centerpiece).** New `ml/training/report.py` `build_report()` writes into the runner's existing `out_dir/<trainer>/` and uploads to B2 beside the model: **`report.json`** (full metrics superset), **`report.md`** (header + metrics table + promote-gate verdict + "is this shippable?" one-liner — what you read), and **PNGs** via matplotlib (already in `.venv`, rendered headless once on the GPU pod): `rank_ic_by_fold.png`, `decile_spread.png`, `equity_curve.png`, `feature_importance.png`. The report versions and rolls back with the model for free.

**Layer 4 — Expose the registry (build-new endpoints).** `GET /admin/ml/versions/{model}` (wraps the already-written `versions.list_versions`), `GET /admin/ml/report/{model}/{version}` (resolves the version dir via `ModelRegistry.resolve`, returns `report.json` + presigned B2 URLs for the `.md` and PNGs), `POST /admin/ml/versions/{model}/{v}/promote|rollback` (flip versions from the UI after reading the report). Add `report_uri` to the metrics JSONB.

**Layer 5 — Dashboard + CLI.** New **"Model Detail"** page at `frontend/app/admin/models/[model]/page.tsx`: (1) a **version table** (v, trained_at, prod/shadow badge, rank-IC, Sharpe, DSR, PBO, gate-pass) with two-checkbox select → side-by-side **compare** panel with a green/red delta column; (2) **per-fold charts** (rank-IC line, decile-spread bars, equity curve, feature-importance bars) from the embedded B2 PNGs OR re-charted client-side from `report.json` arrays with recharts (already in the stack); (3) the **promote-gate verdict** block (the exact reasons the runner recorded); (4) the rendered `report.md` + promote/rollback buttons. CLI parity: `python -m ml.training.report --compare <model> v4 v5` prints the same delta table to the terminal.

**Cheap high-trust fixes.** Point `/ml/regime` at the real `regime_history` table (it currently lies). Replace the `JSON.stringify` metric dump on `/admin/training` with structured per-fold cards. Wire the existing-but-unused `ml/eval/spa.py` (Hansen SPA) + `ml/eval/impact_cost.py` (Almgren-Chriss) into the eval gate — both are re-exported in `ml/eval/__init__.py` (verified) but not yet called at runtime.

---

## 5. Production repo structure

### The single root cause: there is no training↔serving boundary

Verified bidirectional `ml ↔ src.backend` cycle:
- **`ml/ → src.backend`** (6 sites): `ml/data/data_loader.py:14-15,25` (providers), `ml/data/kite_source.py:61`, `ml/training/base.py:86` (registry), `ml/training/trainers/lgbm_signal_gate.py:470`, `ml/data/sentiment_history.py:172`.
- **`src.backend → ml`** at request/serve time (~20 sites): `ai/signals/generator.py:33` (`ml.regime_detector`), `platform/scheduler.py:981,2012,3179,3390`, `ai/feature_engineering.py:304` (`ml.features.lgbm_v2`), `ai/signals/options.py:327` (`ml.strategies`), `data/screener/engine.py:22,365,379`, plus many `ml.features.indicators` / `ml.features.patterns` consumers.

So `ml/` is not a clean research package — `ml/scanner.py`, `ml/risk_manager.py`, `ml/regime_detector.py`, `ml/strategies/`, `ml/backtest/` are serving modules misfiled under the training tree, and `ml/data/` duplicates the serving data layer in `src/backend/data/providers/`. Nothing structurally prevents a trainer from computing features differently than serving — the exact skew serve-smoke is meant to catch.

### Design principle: three concentric rings, one-directional, import-linter-enforced

1. **`core/` (pure domain + contracts)** — dataclasses (`MarketDepth`, signal types, `EngineSpec`/`LabelSpec`/`FeatureSpec`/`CVSpec`), provider Protocols, registry interface, output schema, **shared feature/label builders** (train==serve identical). Depends on nothing internal.
2. **`ml/` (training & research)** — data factory, features, labeling, CV, eval, trainers, the 9-stage pipeline, RunPod entry. May import `core/`. **May NOT import the serving app.**
3. **`backend/` (serving app)** — FastAPI api/services/trading/data-serving/registry-impl/scheduler. May import `core/` + read registry artifacts. **May NOT import `ml/` trainers at request time.**

The two legitimate `backend → ml` couplings (regime features at serve, lgbm inference features at serve) are resolved by **moving the shared feature/label code into `core.features`** (pure, no-deps) so both the trainer and the live engine import identical code. The scheduler's "kick off a training run" call moves behind a thin `ml.cli` subprocess boundary, not an in-process import.

### Naming conventions (locked)

- **One import root.** Drop `src/`: app becomes `backend.api.app:app`. Add `pyproject.toml` declaring `core/`, `ml/`, `backend/` as packages + editable install → `import core/ml/backend` resolves everywhere, killing the `sys.path.insert` hacks (verified in `tests/conftest.py` + ~15 scripts). Only 4 deploy files hardcode the old path (verified: `railway.toml:5`, `nixpacks.toml:25`, `Dockerfile:24`, `scripts/dev.sh:37`).
- **Layer-suffix modules:** trainers `*_trainer.py`, serving engines `*_engine.py`, feature builders `*_features.py`, label builders `*_labels.py`, providers `*_provider.py`, API routers `*_routes.py`.
- **Per-engine trinity:** every `EngineStyle` enum member has exactly one `core/features/<style>_features.py` + one `ml/training/trainers/<style>_trainer.py` + one `backend/ai/engines/<style>_engine.py`. A test asserts the trinity exists for every member.
- **One artifact root** `artifacts/models/<model>/<version>/` (merge the verified three: `ml/models/`, top-level `models/`, `.model_cache/`). Tiny configs committed; large weights gitignored + B2-sourced.
- **Static vs runtime data:** `data/` keeps only committed reference inputs (universes, holidays, tiers, instruments); new `var/` (gitignored) holds regenerable runtime caches.
- **Scripts collapse** into `python -m ml.cli` / `python -m backend.cli` Typer apps; only orchestration shell (`runpod_*.sh`, `dev.sh`, `qa/`, `release/`) stays under `scripts/`.

### Key moves

| From | To | Why |
|---|---|---|
| `ml/regime_detector.py` (`compute_regime_features`) + `ml/features/lgbm_v2.py` (`compute_inference_features`) | `core/features/regime_features.py` + `core/features/lgbm_features.py` | imported by serving (`generator.py:33`, `feature_engineering.py:304`) yet live in research tree → forces the cycle; moving to ring-1 gives trainer + live engine identical code (spec §4.1) |
| provider Protocols (in `src/backend/data/providers/base.py`) + `ml/data/data_loader.py` | `core/contracts/providers.py` (Protocol) + `ml/data/data_loader.py` (consumes it) | breaks the cycle: `ml/` depends on the abstract contract; both research + serving providers implement it |
| `ml/scanner.py`, `ml/risk_manager.py`, `ml/strategies/`, `ml/backtest/` | `ml/research/` + re-point serve-time imports to backend engines | quarantine serving responsibilities out of the training tree |
| `src/backend/` (`src.backend.api.app:app`) | `backend/` (`backend.api.app:app`) + `pyproject.toml` editable install | stable import root, kills sys.path hacks; only 4 deploy files change |
| `ml/models/` + `models/` + `.model_cache/` | `artifacts/models/<model>/<version>/` + `artifacts/rl/` | one registry-managed root; fixes the audit's version-drift finding |
| `scripts/train_*.py`, `backfill_*.py`, `retrain_pipeline.py`, `smoke_all.py`, `seed_*.py` (~30 of 55) | `ml/cli.py` + `backend/cli.py` Typer subcommands | business logic out of argv-parsing scripts → importable + testable |
| `src/backend/services/` (70-file flat) | `backend/services/{screener,fno,intraday,autopilot,portfolio,news,options,market,strategy_runner,assistant}/` | domain ownership; completes the ai/+trading/+data/+platform peer target |
| model-derived levels (e.g. TFT `_derive_levels`) | `backend/trading/risk_engine.py` | spec §4.6: models emit only `expected_return/confidence`; risk engine derives entry/SL/target from ATR |

### Migration is phased, shim-guarded, test-gated — never big-bang

Every move uses a re-export shim at the old path (`from new.path import *  # DEPRECATED, remove after <date>`), with a CI grep that fails if a NEW import uses an old path — mirroring the repo's existing shim policy (structural target 2026-05-25) + route-alias deprecation pattern. The full suite + `uvicorn ... --check` import + the new import-linter contract run green after each phase.

- **Phase 0 — Scaffolding & guards (no moves):** `pyproject.toml` (packages + editable install + import-linter in report-only mode), `requirements/` split (base/train/dev), `tests/contracts/` with the import-linter contract + per-engine-trinity test + deprecated-import grep.
- **Phase 1 — Import root flip:** `git mv src/backend → backend`; update the 4 deploy spots; leave a `src/backend` shim; codemod `src.backend → backend`; remove sys.path hacks. Gate: uvicorn import + suite green + preview boot.
- **Phase 2 — Carve out `core/`:** move provider Protocol, signal schema + EngineSpec family, shared feature builders (`regime_features` + `lgbm_features` + per-style), labeling. Re-point `data_loader` + serve-time imports. import-linter ENFORCE for `core → {}`. Gate: feature-parity test (train builder == serve builder) green.
- **Phase 3 — Quarantine serving code out of `ml/`:** move `scanner/risk_manager/strategies/backtest → ml/research/`; replace serve-time imports with backend engines. Enforce `ml ↛ backend`. Gate: scheduler + signal-generation smoke green.
- **Phase 4 — Pipeline spine + trainer refactor (lands with the 4-engine build):** add `pipeline.py/specs.py/serve_smoke.py/baseline_drift.py` + `adapters/`; rename trainers `*_trainer.py`; refactor onto `run_pipeline` via the 3+1 hooks; wire serve-smoke as a promote precondition. Gate: serve-smoke green for every promoted model + trinity test green.
- **Phase 5 — Services domain-grouping + `api/routes/` + risk engine:** group the flat services; add `backend/trading/risk_engine.py`; strip model-derived levels. Gate: route-inventory test (no router dropped).
- **Phase 6 — Scripts → CLIs + artifact/data consolidation:** fold scripts into the two Typer CLIs; merge artifact roots → `artifacts/`; split `data/` + `var/`; rename `infrastructure/ → infra/`. Gate: a training run writes to `artifacts/` and the registry resolves it.
- **Phase 7 — Shim sweep & contract lock:** after a soak period, delete shims, flip import-linter to strict on all three ring rules, rewrite `ml/__init__.py` docstring + README. Gate: zero deprecated imports; full contract enforced.

### Cruft to clear alongside the reorg

**HEADLINE: two parallel training orchestrators — collapse to one.** Canonical = `python -m ml.training.runner` (discovery over `ml/training/trainers/`), used by every RunPod entrypoint. Legacy/dead = `scripts/train_all_models.py` + `src/backend/ai/training/__init__.py` whose `all_trainers()` now returns `[]` (verified) — trains zero engines.

- **Safe deletes (0 external refs):** `scripts/pod_bootstrap.sh`, `scripts/preflight.sh`, `scripts/train_tft.py`, `scripts/generate_demo_signals.py`, plus untracked `.pyc` cruft (`find . -name '*.pyc' -path '*__pycache__*' -delete`).
- **Investigate-then-delete (need a rewire first):** `scripts/retrain_pipeline.py` is the trickiest — the scheduler migrated off it but admin endpoint `POST /ml/retrain` (`admin/ml.py:234`) still subprocess-launches it, AND it imports a non-existent `scripts/train_quantai.py`. **Rewire the admin endpoint to `ml.training.runner`, then delete** `retrain_pipeline.py` + the legacy `train_lgbm.py`. Then `train_all_models.py` + `src/backend/ai/training/__init__.py` go together. `train_models_full.py` (sole producer of the net-negative outcome models + the no-op RL q_table) and the off-CI backtest harnesses (`backtest_strategies.py`, `backtest_options_strategies.py`, `backtest_harness.py` + `qa/backtest_smoke.sh` + the broken `release/capture_rc_baseline.sh` which references 3 non-existent modules) — confirm the founder doesn't still run them.
- **KEEP (the audit's "dead but good machinery" is the reorg's TARGET infra):** `ml/features/frac_diff.py` is actually LIVE (called by `lgbm_v2.py`); `ml/eval/{spa,impact_cost}.py` are slated to be wired (§4); schema tools (`consolidate_schema.py`, `audit_schema.py`) are deliberate maintenance entrypoints; `train_ai_stock_ranker.py` feeds a shipped screener feature (keep until Momentum replaces AI Top Picks). `exit_engine/{tick_exit,stagnation_trailing}.py` + `ai/microstructure/features.py` are genuinely unwired but gated on a paid tick/L2 feed (Intraday/M4) — investigate against the roadmap, don't delete blind.

---

## 6. Phased execution roadmap (tying it to RunPod-funds + TrueData blockers)

The two external blockers govern sequencing:
- **RunPod GPU funds** gate every step that runs a forecaster (M1c, M2, M3, M4) — `forecast_features.py:62,125` hard-raise without CUDA (verified).
- **TrueData** gates all of M4 — `truedata_provider.py` is missing and intraday raises `NotImplementedError` (verified).

Everything that is **CPU + free-data** (the repo reorg Phases 0-3, the observability Layers 1-5, the M0 spine, M1a Momentum price-only, M1b feature gaps, the two backfill jobs) can proceed **now, with no spend**. This is the critical insight: we build and ship a real Momentum ranker + the full observability surface + a clean repo before we ever turn on a GPU.

**Phase A — Foundation & cleanup (now, no spend, no blockers).** Repo Phase 0-1 (pyproject + import-root flip), the safe cruft deletes, rewire `POST /ml/retrain` to the canonical runner then delete the legacy orchestrator + `retrain_pipeline.py` + `train_lgbm.py`. Convert `train_momentum()` to a `Trainer` subclass. Build the M0 spine (`pipeline.py` + `serve_smoke.py` + `specs.py` + `baseline_drift.py`); fix the `<trainer>_quality_pass` key + un-hardcode PBO; requirements hygiene. **Gate:** full suite + uvicorn import + import-linter (report mode) green.

**Phase B — Universe + Momentum price-only (now, no spend).** Fix `liquid_universe.py` to rank from pg `candles`; add `tiers.py`; default `candidate_pool = nse_all ∪ nifty500`; emit `universe_snapshot.json` + pass `as_of_date`. Run the two backfill jobs (refresh stale cache + Tier-B mid-caps). Train M1a: `python -m ml.training.runner --only momentum_lambdarank --promote` (price-only, CPU) — a real shippable ranker on ~229k rows. Add RS-vs-index, then RS-vs-sector (M1b). **Gate:** M1a passes stages 1-9 + serve-smoke; repo Phase 2 (carve out `core/`) green with feature-parity test.

**Phase C — Observability (now, no spend, parallelizable with B).** Layers 1-5: `run_log` column + log-streaming endpoint + RunRow disclosure; `ml/training/report.py` (JSON+md+PNGs); routed `list_versions` + report endpoints; the Model Detail page; fix the `/ml/regime` stub to read `regime_history`; wire SPA + impact-cost into the gate. **Gate:** a CPU Momentum run produces a full `TrainReport` visible on Model Detail with version compare.

**Phase D — Repo Phase 3 + Phase 4 trainer refactor (now → lands with engine build).** Quarantine serving code out of `ml/`; refactor every trainer onto `run_pipeline`; wire serve-smoke as a promote precondition. **Gate:** serve-smoke green for every promoted model; per-engine trinity test green.

**Phase E — RunPod GPU forecasters (gated on RunPod funds).** Provision the pod (TimesFM git `--no-deps`+jax pin; Kronos clone@pinned-commit on `KRONOS_PATH`). M1c: re-run Momentum `--with-forecasts --promote` (upgrade degraded → specced). Then M2 Swing (author `swing_features.py` + `swing_ranker.py`; TFT fit + Chronos-2 zero-shot). Then M3 Positional (author `positional_features.py` + `positional_ranker.py`; Kronos+TimesFM long-horizon; start PIT-fundamentals forward capture). **Gate:** each engine's LGBM clears its promote gate (rank-IC/NDCG OOS + DSR/PBO + impact-cost Sharpe) + serve-smoke.

**Phase F — Repo Phase 5-6 (services grouping + CLIs + artifact consolidation; now → after E).** Group `services/` into domain subpackages; add `risk_engine.py`; fold scripts into `ml.cli`/`backend.cli`; merge artifact roots → `artifacts/`; split `data/`+`var/`; rename `infra/`. **Gate:** route-inventory test; a training run writes to `artifacts/` and the registry resolves it.

**Phase G — Intraday (gated on TrueData AND RunPod funds — last).** Build `src/backend/data/providers/truedata_provider.py` + `ml/features/intraday_features.py` + `ml/training/trainers/intraday_patchtst.py`; train PatchTST (GPU) → LGBMClassifier on intra-bar triple-barrier (XL=50). **Gate:** OOS net Sharpe/PF clears the impact-cost gate + serve-smoke; 5-min inference loop smoke-tested.

**Phase H — Contract lock (after the soak period).** Repo Phase 7: delete all shims, flip import-linter to strict, rewrite `ml/__init__.py` + README. **Gate:** zero deprecated imports; full 3-ring contract enforced in CI; clean deploy.

**Critical-path summary:** Phases A→D + the observability of C are **free and unblocked — do them first and ship a real Momentum ranker with full visibility on a clean repo.** Phase E (M1c/M2/M3) waits only on RunPod funds. Phase G (Intraday) waits on both TrueData and RunPod. Nothing in the GPU-blocked phases blocks shipping the CPU engine — that decoupling is the whole point of training engines independently per milestone.

---

## Files relevant to executing this (all absolute)

- Reference trainer (the proven spine): `/Users/rishi/Downloads/Swing_AI_Final/ml/training/trainers/momentum_lambdarank.py` (currently `train_momentum()`, must become a `Trainer` subclass)
- Runner + discovery + base: `/Users/rishi/Downloads/Swing_AI_Final/ml/training/{runner.py,discovery.py,base.py}`
- Forecaster adapters (GPU-gated): `/Users/rishi/Downloads/Swing_AI_Final/ml/features/forecast_features.py`
- Universe builder (fix to pg-ranked): `/Users/rishi/Downloads/Swing_AI_Final/ml/data/liquid_universe.py`
- Eval library (wire SPA + impact-cost): `/Users/rishi/Downloads/Swing_AI_Final/ml/eval/{spa.py,impact_cost.py,overfitting.py}`
- Registry (source of truth): `/Users/rishi/Downloads/Swing_AI_Final/src/backend/ai/registry/{model_registry.py,versions.py,b2_client.py}`
- Observability surfaces: `/Users/rishi/Downloads/Swing_AI_Final/src/backend/api/admin/{training.py,ml.py}` (fix `/ml/regime` hardcode at `ml.py` ~line 142; route `list_versions`)
- Deploy files to flip (`src.backend → backend`): `/Users/rishi/Downloads/Swing_AI_Final/{railway.toml,nixpacks.toml,Dockerfile,scripts/dev.sh}`
- Missing intraday blocker (build): `/Users/rishi/Downloads/Swing_AI_Final/src/backend/data/providers/truedata_provider.py`
- Legacy orchestrator to retire: `/Users/rishi/Downloads/Swing_AI_Final/scripts/train_all_models.py` + `/Users/rishi/Downloads/Swing_AI_Final/src/backend/ai/training/__init__.py`
- To build new: `core/` (top-level), `ml/training/{pipeline.py,serve_smoke.py,specs.py,baseline_drift.py,report.py}`, `ml/features/{swing,positional,intraday}_features.py`, `ml/training/trainers/{swing,positional,intraday}_*.py`, `pyproject.toml`, `frontend/app/admin/models/[model]/page.tsx`

---

# Appendix A — Per-model training plan (full)

# Quant X — Proper End-to-End Training Plan for the 4-Engine ML/DL System

**Date:** 2026-06-16 · **Author role:** Lead ML Engineer · **Grounded in:** the 4-engine spec (`docs/superpowers/specs/2026-06-15-quantx-4engine-mldl-design.md`), the readiness/pipeline doc (`docs/ML_TRAINING_READINESS_AND_PIPELINE_2026_06_16.md`), the deep audit (`docs/ML_DL_DEEP_AUDIT_2026_06_15.md`), and the live tree (`ml/training/runner.py`, `ml/training/{base,discovery,purged_cv}.py`, the working reference trainer `ml/training/trainers/momentum_lambdarank.py`, `ml/features/{momentum_features,forecast_features}.py`, `ml/labeling/{ranking_labels,triple_barrier}.py`, `ml/eval/*`).

The reference trainer `momentum_lambdarank.py` already proves the pattern end-to-end (OOS rank-IC **0.084**): `load_ohlcv → build_momentum_features (train==serve) → forward_return_quantile_labels (absolute) → purged_walk_forward_by_date (day-sized embargo) → LGBMRanker(lambdarank) → OOS rank-IC + decile spread per fold → final fit on all data + artifact + feature_order.json sidecar`. Every model below is described to slot into that same spine.

---

## 1. The two-tier model topology (this is the crux)

There are **11 models across 4 engines**, but they are NOT 11 peers. Within each engine there is a strict **producer→consumer dependency**: the deep-learning **forecasters emit FEATURE COLUMNS** (e.g. `tsfm_fwd_ret`, `tsfm_uncert`, `kronos_fwd_ret`) that the engine's **LightGBM ranker/classifier consumes as inputs alongside the price/cross-sectional features**. The LGBM is the model that emits the signal; the forecasters are upstream feature generators that are *zero-shot or pretrained* — they are not fit per engine in the supervised sense.

```
ENGINE = one Trainer (one trainers/*.py, one model_versions row, one --only target)
  ├─ forecaster A  (zero-shot/pretrained DL)  ─┐
  ├─ forecaster B  (zero-shot/pretrained DL)  ─┤→ forecast feature columns
  └─ price + cross-sectional + RS features    ─┴→ LGBM ranker/classifier  → SIGNAL (artifact)
```

| Engine | Forecaster(s) → feature emitters | Decision model (the artifact) | GPU |
|---|---|---|---|
| **Momentum** | TimesFM + Kronos | **LGBM LambdaRank** | forecasters yes, LGBM no |
| **Swing** | TFT + Chronos-2 | **LGBM Ranker** | forecasters yes, LGBM no |
| **Positional** | Kronos + TimesFM | **LGBM Ranker** | forecasters yes, LGBM no |
| **Intraday** | PatchTST | **LGBM (buy/sell/no-trade)** | both yes (needs TrueData) |

**Consequence for "separately vs together":** see §5. The forecasters do not have an independent training loop to schedule — they are loaded inside the engine trainer's `build_features` closure (exactly as `momentum_lambdarank._build_dataset` does with `with_forecasts=True`). So the unit of scheduling is **the engine**, and the dependency is INSIDE each engine, resolved by the feature-build step, not by `depends_on` across trainers.

---

## 2. Per-model lifecycle (all 11)

### MOMENTUM ENGINE — trainer `ml/training/trainers/momentum_lambdarank.py` (exists, working)

**M-1. TimesFM (forecaster, feature emitter)**
- **Data:** daily CA-adjusted close from `load_ohlcv` (the same panel the ranker uses). Context window 512 bars, min 252 history.
- **Features it produces:** `tsfm_fwd_ret = point[-1]/last_close - 1`, `tsfm_uncert = (q90−q10)/last_close` at the 20d horizon. One row per (rebalance-date, symbol), forward-filled to daily by `merge_forecast_features` (as-of backward merge).
- **Labels / CV / training:** NONE — `google/timesfm-2.5-200m-pytorch` is run **zero-shot**; no fitting, no fold loop. It is inference-only. (Optional later: light covariate fine-tune; out of scope for v1.)
- **HPO:** none (context/horizon/stride are config, tuned by the consuming LGBM's OOS metric, not on the forecaster itself).
- **Eval metrics:** measured indirectly via the LGBM's OOS rank-IC delta with vs. without the forecast columns, and the column's own forward-IC logged in EDA (Stage 1). No standalone promote gate.
- **Output schema:** 2 float columns appended to the feature matrix.
- **Artifact:** none of its own — it is a HuggingFace pull (pin the model id + revision). The shipped artifact is the LGBM that learned to use its columns.
- **Cadence:** re-inferred on each engine retrain (weekly) at `stride=5` (weekly rebalance to control GPU cost).
- **GPU:** YES (`forecast_features.py` raises if `torch.cuda.is_available()` is False).

**M-2. Kronos (forecaster, feature emitter)**
- **Data:** full OHLCV panel (open/high/low/close/volume), context 400 bars, min 252 history.
- **Features it produces:** `kronos_fwd_ret = pred_close[horizon]/last_close − 1` (finance K-line foundation model, `NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base`).
- **Labels / CV / training:** NONE — pretrained, sampled inference (`predict(..., T=1.0, top_p=0.9, sample_count=1)`). Zero-shot.
- **HPO:** none (sampling temp/top_p fixed).
- **Eval metrics:** same as TimesFM — incremental rank-IC contribution + EDA forward-IC of the column.
- **Output schema:** 1 float column.
- **Artifact:** none of its own (clone @ pinned commit + HF weights on `KRONOS_PATH`). **Risk: the adapter must be the live `forecast_features.kronos_forecast_features` (the old `kronos_features.py` was deleted; only a stale `.pyc` existed).**
- **Cadence:** re-inferred per engine retrain (weekly), `stride=5`.
- **GPU:** YES.

**M-3. LGBM LambdaRank (the Momentum signal — THE artifact)**
- **Data source:** `load_ohlcv(cached_universe(), 2020-01-01..2026-03-01)` over the liquid universe (top-N by 30d median ADV; fix to source ADV from pg, fail-loud).
- **Features:** the 13 columns in `MOMENTUM_FEATURE_ORDER` (multi-horizon returns 5/10/21/63/126/252d, mom consistency/accel, vol-adj momentum, dist-SMA-50/200, above-63d-high, realized vol, drawdown-252, rel-volume, OBV slope, **xs_rank_ret_21/63 cross-sectional**) + (when `with_forecasts=True`) `tsfm_fwd_ret, tsfm_uncert, kronos_fwd_ret`. RS-vs-index/sector columns are additive features (NSEI cached; sector indices to be ingested) — kept as FEATURES only, no benchmark in the label.
- **Labels:** `forward_return_quantile_labels(horizon=20, n_quantiles=10)` — **absolute** forward return → per-date decile relevance grade (no benchmark). 30/60d added as additional heads/configs.
- **CV scheme:** `purged_walk_forward_by_date(PurgedCVConfig(n_folds=5, test_days=63, embargo_days=20, train_days=378))` — date-grouped, embargo ≥ label horizon (20d).
- **Training procedure:** per fold → fit `LGBMRanker(objective=lambdarank, metric=ndcg, ndcg_eval_at=[10,20])` with LightGBM `group=` = rows-per-date; predict OOS; compute rank-IC + decile spread. Then final fit on all usable rows → shipped artifact (`booster_.save_model`).
- **HPO:** Optuna TPE (`ml/training/optuna_search.py`) on the SAME OOS rank-IC metric over leaves/lr/min_child/subsample/colsample/reg_lambda; n_trials feeds the DSR null. Opt-in.
- **Eval metrics:** OOS rank-IC mean/std/ICIR, decile spread, NDCG@10/20; DSR + PBO via CPCV; impact-cost-deducted long-short Sharpe of the top-decile portfolio.
- **Output schema (§4.7):** `expected_return, rank, percentile, top_decile_prob, confidence`.
- **Artifact:** `momentum_lambdarank.txt` (LightGBM booster) + `feature_order.json` sidecar + `metrics.json`.
- **Cadence:** weekly retrain.
- **GPU:** NO (CPU-native; runs in minutes).

---

### SWING ENGINE — trainer `ml/training/trainers/swing_ranker.py` (build new; `tft_swing.py` exists as the TFT piece to refactor)

**S-1. TFT (forecaster, feature emitter)**
- **Data:** daily OHLCV + (zero-filled-if-thin) screener.in fundamentals snapshot, nselib FII/DII, news sentiment — but the TFT itself trains on the price/return target series with known/observed covariates.
- **Features it produces:** multi-horizon point forecast (5/10/20d) + quantile spread columns → forecast feature columns for the ranker.
- **Labels:** the TFT trains on its own forecasting objective (quantile loss over 5/10/20d forward returns). **Audit fix: it must forecast the NEXT bars, not reconstruct the last 5 observed bars; and serve via ONE framework — neuralforecast end-to-end (delete the pytorch-forecasting `.ckpt` path).**
- **CV:** purged walk-forward windows (same date-grouped embargo).
- **Training:** neuralforecast TFT fit on GPU; artifact `tft_swing_nf.*`. Must round-trip through the production predictor (serve-smoke) before promotion.
- **HPO:** light (hidden size, attention heads, dropout, lr) — capped; this is GPU-expensive.
- **Eval metrics:** OOS directional accuracy + quantile calibration; gate on its incremental contribution to the swing ranker's OOS rank-IC.
- **Output schema:** forecast columns.
- **Artifact:** neuralforecast tarball + feature_order sidecar.
- **Cadence:** weekly (re-fit with the engine).
- **GPU:** YES.

**S-2. Chronos-2 (forecaster, feature emitter)**
- **Data:** daily close series (univariate zero-shot).
- **Features:** Chronos-2 forecast mean + interval-width columns (auxiliary forecaster, lowest-risk lib; bump floor to >=2.0.0).
- **Labels / CV / training:** zero-shot (pretrained Amazon Chronos-2); no fold loop.
- **HPO:** none.
- **Eval metrics:** incremental rank-IC + EDA forward-IC.
- **Output schema:** forecast columns.
- **Artifact:** none of its own (HF pull, pin revision).
- **Cadence:** re-inferred per engine retrain (weekly).
- **GPU:** optional (CPU-capable, GPU faster).

**S-3. LGBM Ranker (the Swing signal — THE artifact)**
- **Data source:** `load_ohlcv` daily + screener.in fundamentals + nselib FII/DII + news sentiment (honest zero-fill where history is thin; flagged in EDA).
- **Features (~120-250):** price/return, trend, volatility, momentum/RS, cross-sectional ranks, fundamentals, ownership/flow, sentiment/event, regime, calendar + TFT/Chronos forecast columns. **Author `ml/features/swing_features.py` with a frozen `SWING_FEATURE_ORDER`, shared train==serve (mirror `momentum_features.py` exactly — groupby-transform, no groupby.apply on single-symbol serve path).**
- **Labels:** `forward_return_quantile_labels(horizon=5/10/20)` — absolute, decile relevance.
- **CV scheme:** `purged_walk_forward_by_date` with embargo ≥ max horizon (20d).
- **Training:** per-fold `LGBMRanker` with date groups → OOS metrics → final fit. AFML sample-uniqueness weights (from triple-barrier t1) applied.
- **HPO:** Optuna on OOS rank-IC.
- **Eval metrics:** rank-IC/ICIR, NDCG@k, decile spread, DSR/PBO, SPA, impact-cost-deducted Sharpe.
- **Output schema:** `expected_return_{5,10,20}d, rank, percentile, confidence, expected_hold_days`.
- **Artifact:** `swing_ranker.txt` + `feature_order.json` + `metrics.json`.
- **Cadence:** weekly.
- **GPU:** NO.

---

### POSITIONAL ENGINE — trainer `ml/training/trainers/positional_ranker.py` (build new)

**P-1. Kronos (forecaster)** — same model as M-2 but long-horizon context; **features:** `kronos_fwd_ret` at 1/3/6m horizon. Zero-shot, no fitting. Cadence monthly. GPU YES.

**P-2. TimesFM (forecaster)** — same as M-1 but long-horizon; **features:** `tsfm_fwd_ret, tsfm_uncert` at 1/3/6m. Zero-shot. Cadence monthly. GPU YES.

**P-3. LGBM Ranker (the Positional signal — THE artifact)**
- **Data source:** `load_ohlcv` long-horizon EOD/weekly bars + light screener.in fundamentals snapshot (**PIT capture forward** — deepens over time; do NOT broadcast a single snapshot across history, the audit's leakage finding).
- **Features:** long-horizon price/momentum, quality/value/growth (sparse, zero-filled honestly today), ownership/liquidity, sector relative-strength, regime + Kronos/TimesFM forecast columns. **Author `ml/features/positional_features.py` with frozen `POSITIONAL_FEATURE_ORDER`, shared train==serve.**
- **Labels:** `forward_return_quantile_labels(horizon=21/63/126)` (1/3/6m) — absolute, decile relevance.
- **CV scheme:** `purged_walk_forward_by_date` with embargo ≥ longest horizon (126d) — wide test windows, fewer folds.
- **Training:** per-fold `LGBMRanker` → OOS → final fit. AFML weights.
- **HPO:** Optuna on OOS rank-IC.
- **Eval metrics:** rank-IC/ICIR, NDCG@k, decile spread, DSR/PBO, impact-cost-deducted Sharpe.
- **Output schema:** `expected_return_{1,3,6}m, rank, percentile, confidence, factor sub-scores`.
- **Artifact:** `positional_ranker.txt` + `feature_order.json` + `metrics.json`.
- **Cadence:** monthly.
- **GPU:** NO.

---

### INTRADAY ENGINE — trainer `ml/training/trainers/intraday_patchtst.py` (build new; **GATED on TrueData**)

**I-1. PatchTST (sequence forecaster, feature emitter)**
- **Data source:** TrueData 1/3/5/15-min bars (+ OI). `FreeDataProvider.get_ohlcv` raises NotImplementedError for intraday — this is the hard data stop; needs `truedata_provider.py`.
- **Features it produces:** sequence-channel forecast over 15/30/60m + a forecast-return/uncertainty column for the LGBM.
- **Labels:** trains on its forecasting objective over the bar sequence.
- **CV:** purged walk-forward in trading SESSIONS (embargo ≥ longest horizon in bars).
- **Training:** neuralforecast / transformers PatchTST on GPU; artifact + serve-smoke.
- **HPO:** patch length, stride, model dim — capped.
- **Eval metrics:** OOS directional accuracy at 15/30/60m + incremental contribution to the LGBM.
- **Output schema:** forecast columns.
- **Artifact:** PatchTST checkpoint + sidecar.
- **Cadence:** weekly retrain; inference every 5 min during market hours.
- **GPU:** YES.

**I-2. LGBM (the Intraday signal — THE artifact)**
- **Data source:** TrueData minute bars + option chain/OI/Greeks + India VIX + breadth.
- **Features (~60-150 + 20-60 sequence channels):** RSI/MACD/ADX, short returns, vol regime, OI delta, PCR, VWAP distance, breadth, session flags + PatchTST forecast columns. **Author `ml/features/intraday_features.py`, shared train==serve.**
- **Labels:** **time-boxed triple-barrier** at 15/30/60m via `triple_barrier_events(close, atr, high=, low=)` — intra-bar High/Low touch (the M0 fix), conservative tie-resolution to stop. `t1` feeds sample-uniqueness weights + embargo.
- **CV scheme:** purged walk-forward by session; embargo ≥ longest horizon in bars.
- **Training:** `LGBMClassifier` (buy/sell/no-trade) with sample weights → OOS → final fit.
- **HPO:** Optuna on OOS PF/Sharpe (impact-cost-deducted; intraday costs dominate).
- **Eval metrics:** OOS Sharpe/PF, hit-rate by class, DSR/PBO, impact-cost-deducted net Sharpe.
- **Output schema:** `signal {buy/sell/no-trade}, prob_up, expected_return_{15,30,60}m, confidence, rank`.
- **Artifact:** `intraday_lgbm.txt` + `feature_order.json` + `metrics.json`.
- **Cadence:** weekly retrain; 5-min inference loop.
- **GPU:** NO (LGBM); the engine as a whole needs GPU for PatchTST.

---

## 3. Shared spine every model rides (do NOT re-implement per trainer)

Each engine trainer subclasses `ml.training.base.Trainer`, drops into `ml/training/trainers/`, auto-registers via `discovery.py`, and delegates `train()` to the canonical 9-stage `run_pipeline(ctx)` (to be built in `ml/training/pipeline.py`). It overrides only the 3+1 hooks: `build_features` (Stage 4), `build_labels` (Stage 3), `fit_model` (Stage 6), `predict_for_serve` (Stage 8b serve-smoke). Stages 1 (EDA hard-fail), 2 (quality gate), 5 (purged WFCV/CPCV), 8 (eval: rank-IC/NDCG + DSR/PBO + SPA + Almgren-Chriss impact), 9 (registry/promote) are 100% shared. The **#1 audit fix is Stage 8b serve-smoke**: round-trip the artifact through the production predictor and block promote on any feature-order/shape/normalization mismatch — this is what guarantees "what trained is what serves" and kills the train/serve skew that affected all 4 prior PROD models.

The runner (`ml/training/runner.py`) is unchanged in orchestration: discovery → topo-sort on `depends_on` → per-trainer train/evaluate/register → promote gate (financial + `<trainer>_quality_pass` key — fix the Qlib-style key typo so the gate isn't dead) → Kelly → safety-net (blocks promote when `primary_value is None` or `n_calibrations==0`) → Sentry. Run via `python -m ml.training.runner --only <engine> --promote` (per engine) or `--all`.

---

## 4. SEPARATELY or ALL TOGETHER — the definitive answer

**Train each ENGINE as ONE self-contained trainer, and train the engines INDEPENDENTLY (sequenced per milestone, not one monolithic run). WITHIN an engine, the forecaster→LGBM dependency is resolved INSIDE the trainer's feature-build step, not across trainers.**

Why this is correct and not a compromise:

1. **Within an engine there IS a hard dependency order, but it is intra-trainer, not inter-trainer.** The forecasters (TimesFM/Kronos/TFT/Chronos/PatchTST) produce FEATURE COLUMNS that the LGBM consumes. So the LGBM cannot be fit until the forecast columns exist for the training window. But the forecasters are **zero-shot/pretrained** — they are *inference calls inside `build_features`*, exactly as `momentum_lambdarank._build_dataset` calls `timesfm_forecast_features` + `kronos_forecast_features` + `merge_forecast_features` when `with_forecasts=True`. There is no separate fit loop to schedule for them. (TFT/PatchTST are the exception: they ARE fit, but they are fit *inside the engine trainer's own train step* before the LGBM step, still one trainer, one artifact bundle.) So "the engine" is the atomic, schedulable unit — one `trainers/*.py`, one `--only` target, one `model_versions` row for the shipped LGBM.

2. **Across engines the models are independent** — Momentum/Swing/Positional/Intraday share the foundation (loader, feature factory, labeling, purged CV, eval, registry) but have disjoint feature sets, labels, horizons, and artifacts. There is **no cross-engine `depends_on`** for these four signal models. (`depends_on` in the runner is reserved for genuine ordering like a regime overlay or an ensemble that re-reads a just-trained version; the four engine rankers don't need it.)

3. **Why NOT one giant simultaneous run:** the GPU forecasters (TimesFM, Kronos, TFT, PatchTST) are multi-hour on a single RunPod RTX 4090; running all engines at once would serialize on the one GPU anyway and make failures hard to isolate. The audit + readiness docs both mandate **sequencing per milestone (M0→M1→M2→M3→M4)**, so a broken forecaster install can't abort a whole batch. The runner already keeps going across trainer failures, but per-milestone runs keep the GPU budget and debugging tractable.

4. **Why NOT fully separate per model (11 runs):** the forecasters have no standalone artifact or promote gate — promoting a zero-shot TimesFM "model" is meaningless. The LGBM IS the engine's shippable model and the forecasters only matter through their incremental rank-IC contribution to it. Splitting them into separate runs would break the train==serve feature contract (the whole point of `build_features` being shared) and re-introduce the skew the audit found.

**Net:** unified RUNNER + discovery model, **one trainer per engine** (4 trainers, each internally orchestrating its forecasters→LGBM), run **per-engine on `--only`** in milestone order. The forecasters are dependencies *inside* the engine, satisfied by the shared feature builder, never as peer trainers.

---

## 5. Concrete training order

**Build the spine first (M0), then train one engine per milestone. Within each engine run, the order is forecaster-features → LGBM.**

1. **M0 — Foundation (no training):** build `pipeline.py` (9-stage spine) + `serve_smoke.py` (Stage 8b) + `specs.py` + `baseline_drift.py`; wire EDA/quality/purged-CV/eval/registry; fix the promote-gate key typo + un-hardcode PBO; requirements hygiene (add neuralforecast, re-enable pytorch-forecasting OR commit to neuralforecast-only for TFT, un-comment timesfm, bump chronos floor); RunPod env (TimesFM git `--no-deps`+jax pin, Kronos clone@pinned-commit on `KRONOS_PATH`).
2. **M1 — Momentum (CPU first, then GPU upgrade):**
   a. `python -m ml.training.runner --only momentum_lambdarank --promote` price-only (`with_forecasts=False`) — CPU, fast, exercises stages 1-9 + serve-smoke. This is a real, shippable cross-sectional momentum ranker today.
   b. Close pure-code feature gaps: RS-vs-index (NSEI cached), then RS-vs-sector (ingest sectoral indices + full-universe sector map).
   c. On RunPod GPU: install/verify TimesFM + Kronos adapters, re-run with `--with-forecasts` (forecaster inference → columns → re-fit LGBM). This upgrades M1 from degraded → specced stack.
3. **M2 — Swing:** author `swing_features.py` + `swing_ranker.py`; train TFT (neuralforecast, GPU) + Chronos-2 (zero-shot) → forecast columns → `LGBMRanker`. `--only swing_ranker --promote`.
4. **M3 — Positional:** author `positional_features.py` + `positional_ranker.py`; reuse Kronos+TimesFM at long horizons (zero-shot, GPU) → columns → `LGBMRanker`. Start PIT-fundamentals forward capture now so quality/value/growth deepen. `--only positional_ranker --promote`.
5. **M4 — Intraday (gated on TrueData):** build `truedata_provider.py` + `intraday_features.py` + `intraday_patchtst.py`; train PatchTST (GPU) → columns → `LGBMClassifier` on time-boxed triple-barrier labels. `--only intraday_patchtst --promote`.

Within every engine the strict order is: **(1) forecaster feature emission [zero-shot inference or DL fit] → (2) merge into the shared, frozen feature_order → (3) purged-WFCV fit of the LGBM → (4) OOS eval + DSR/PBO/SPA/impact → (5) final fit → (6) serve-smoke → (7) registry promote.**

---

## 6. Universal invariants (apply to all 11, from the audit)

- **One shared feature builder per engine**, imported by trainer AND serving engine — train/serve parity by construction.
- **Frozen `feature_order.json` sidecar** beside every artifact; serve-smoke blocks promote on mismatch.
- **Absolute labels, no benchmark** — RS-vs-index/sector are FEATURES only.
- **Day-sized, date-grouped embargo ≥ label horizon** via `purged_walk_forward_by_date` (never positional-row embargo).
- **Intra-bar High/Low triple-barrier** (M0 fix) wherever triple-barrier is used; `t1` feeds AFML sample-uniqueness weights + the embargo.
- **Real promote gate** per model type (rankers → rank-IC/NDCG OOS; classifiers → impact-cost-deducted Sharpe/PF + DSR/PBO); fix the `<trainer>_quality_pass` key; never promote on `--promote` alone.
- **Models never emit trade levels** — a separate risk engine derives entry/SL/target from ATR.
- **Fail loud** — no empty-frame masking on data load; forecasters raise if CUDA absent.


---

# Appendix B — Stock universe design (full)

## Stock-Universe Strategy — 4-Engine ML/DL Training

### 1. Ground truth (verified against the live tree, not docs)

| Asset | Reality |
|---|---|
| **Cache** (`data/cache/*_NS_10y.csv`) | **310 EQ symbols** + `NSEI_10y.csv` (benchmark) + `INDIAVIX_10y.csv`. ~1,416 daily rows each, **2020-05-27 → 2026-02-06** (~5.7 yr, **not 10y** — name is a misnomer). 274/310 are full-length; **stale by ~4 months** (ends Feb 2026). |
| **Tier files** (`data/nse_tiers/`) | nifty50=50, nifty100=102, nifty250=321, **nifty500=517**, **nse_all=506**. nse_all ∪ nifty500 ≈ **650 distinct** (251 cache symbols ∈ nifty500; 59 cache symbols are mid-caps outside it). |
| **Full NSE main board** | ~2,385 EQ (Supabase universe, memory). We do **not** train on all of it — illiquid tail is untradeable noise. |
| **Builder** (`ml/data/liquid_universe.py`) | Top-N by 30-day median ADV, `min_price=10`, `min_avg_volume=100k`, `strict` fail-loud, **PIT `as_of_date`** + survivorship rejoin already implemented. **One bug to fix (spec §3.4):** it ranks ADV via **live yfinance**, not pg `candles` — non-reproducible + survivorship-leaky. |
| **Survivorship** (`delisted_registry.py`) | Hand-curated `DELISTED_NSE` (11 events: DHFL, RCOM, JETAIRWAYS, MINDTREE, HEXAWARE…). `was_listed_at()` + `historical_universe_extras()` wired into the builder. Thin but correct mechanism. |
| **Corp actions** (`corporate_actions.py`) | Hand-curated splits/bonuses; `adjust_volume_for_actions` fixes phantom volume spikes (yfinance auto-adjusts price, not volume). Needed for honest ADV ranking. |
| **Data path** | `load_ohlcv` → `FreeDataProvider._default_loader`: **cache CSV → pg `candles` → yfinance** (daily/weekly/monthly only; intraday raises `NotImplementedError`). |

### 2. Real ADV distribution (from the 310 cached symbols, ₹ crore/day, last 60d median)

| Cut | ADV floor | Marginal name |
|---|---|---|
| Top 20 | ~₹537 cr | MCX |
| Top 50 | ~₹237 cr | POLICYBZR |
| Top 100 | ~₹144 cr | TATAPOWER |
| Top 150 | ~₹72 cr | IRCTC |
| Top 200 | ~₹29 cr | ACC |
| Top 250 | ~₹13 cr | MASTEK |
| Top 300 | ~₹3.5 cr | RAJESHEXPO |

Liquidity falls off a cliff after ~250. This is what sets the tier floors below — **price floor ₹50, ADV floor ₹5 cr/day for the broadest equity tier** keeps the cross-section tradeable.

### 3. Tiered universes (membership = top-N by pg-sourced 30d median ADV, then floors)

| Tier | Size | ADV floor | Price floor | Source pool | Engine | PIT |
|---|---|---|---|---|---|---|
| **XL** | 50 | ₹200 cr/day | ₹50 | nifty100 ∪ F&O | **Intraday** (TrueData) | yes |
| **L** | 200 | ₹25 cr/day | ₹30 | nse_all ∪ nifty500 | **Momentum** | yes |
| **M** | 350 | ₹10 cr/day | ₹30 | nse_all ∪ nifty500 | **Swing** | yes |
| **B** | ~500 | ₹5 cr/day | ₹50 | nse_all ∪ nifty500 | **Positional** | yes |
| **U** | dynamic | (membership-only) | — | B + `historical_universe_extras(as_of)` | survivorship spine for all | required |

**Design rationale**
- **Momentum (L=200):** momentum decays in illiquid names + LambdaRank needs a clean, fillable cross-section. Top-200 = the proven sweet spot (matches the reference trainer's `cached_universe`, OOS rank-IC 0.084). Weekly rebalance ⇒ executable.
- **Swing (M=350):** wider net for 5–20d alpha; mid-caps add signal the bluechips lack. Still ≥₹10 cr ADV so a Pro/Elite user can fill.
- **Positional (B=500):** 1–6m holds tolerate lower liquidity ⇒ broadest equity cross-section for the deepest factor spread; ₹5 cr floor screens penny noise.
- **Intraday (XL=50):** intraday edge lives only in the most liquid names (tight spreads, depth, option chains). 50 is all TrueData minute-bar + OI ingestion can realistically support at 5-min cadence. **Gated on TrueData — last.**

All tiers share one ranking function and one PIT spine (U); they differ only by `top_n` + floors, so there's no second universe codebase.

### 4. Membership rule + point-in-time + survivorship (the non-negotiables)

1. **Rank from pg `candles`, not live yfinance** (fix `liquid_universe.py` per spec §3.4). Reproducible + leak-free.
2. **Always pass `as_of_date`** at every walk-forward fold boundary. Membership = top-N ADV *as of that date*, so the 2021 fold uses the 2021 universe, not today's winners.
3. **Survivorship rejoin:** `historical_universe_extras(as_of)` adds back names tradeable then but delisted since (DHFL pre-2021-09, RCOM pre-2023-08, JETAIRWAYS pre-2024-10, MINDTREE pre-2022-11…). They're admitted to the candidate pool; the ADV filter drops the ones with no data — the point is the model *sees the losers*.
4. **Corp-action volume adjustment** before ADV ranking (`adjust_volume_for_actions`) so a 1:1 bonus doesn't fake a liquidity jump.
5. **Freeze a membership snapshot** (`universe_snapshot.json`: tier, as_of, ranked symbols + ADV) per training run, stored next to the artifact — reproducibility + audit.
6. **Floors:** `min_price` and `min_avg_volume` (ADV) reject penny/circuit names so signals are fillable.

### 5. Dataset size per engine (today, free data)

Window = 1,416 cache bars; usable = bars − 252 warmup − label horizon; cross-sectional rows = usable × symbols.

| Engine | Tier | Symbols (cached now) | Usable bars/sym | **Total rows** | Label horizon |
|---|---|---|---|---|---|
| **Momentum** | L=200 | 200 (all cached) | 1,144 | **~229k** | 20d |
| **Swing** | M=350 | 310 cached (40 need backfill) | 1,144 | **~355k** | 5/10/20d |
| **Positional** | B=500 | 310 cached (190 need backfill) | 1,038 | **~322k** | 1/3/6m (126d) |
| **Intraday** | XL=50 | 0 (TrueData) | — | **~0.9–1.9M** 5-min rows/yr | 15/30/60m |

**Symbols × years × bars (specced, post-backfill):** Momentum 200×5.7yr×~250 ≈ 285k · Swing 350×5.7×250 ≈ 500k · Positional 500×5.7×250 ≈ 712k · Intraday 50×1yr×~75 bars/day ≈ 0.94M.

These are healthy LightGBM ranker sizes (hundreds of k cross-sectional rows). The binding constraint is **breadth (symbols × dates)**, not depth — which is exactly why widening the candidate pool to nse_all ∪ nifty500 matters.

### 6. Data-sourcing path

**Now (free):** `load_ohlcv(tier_symbols, start, end)` → cache CSV → pg `candles` → yfinance. Trains Momentum/Swing/Positional with **zero new data dependencies**.

**Two backfill jobs before the GPU run (free, daily/EOD):**
1. **Refresh stale cache** — 310 symbols are ~4 months stale (end Feb 2026); re-pull to today via yfinance/pg.
2. **Backfill Tier B mid-caps** — ~190 names in nse_all ∪ nifty500 lack cache CSVs. Pull 5.7yr daily via yfinance, run BOM-tolerant normalizer + corp-action volume adjust, upsert to pg. (`scripts/backfill_ohlc_pg.py` exists for the 3yr pg path.)

**Later (TrueData):** flip `DATA_PROVIDER=truedata` (config-only) to unblock Intraday minute bars + OI/Greeks and upgrade EOD fidelity for engines 1–3. No engine rework.

### 7. Concrete to-do to make this real
1. Add `LiquidUniverseConfig` presets / a `tiers.py` constant for XL/L/M/B with the floors above.
2. Fix `liquid_universe.py` to rank from pg `candles` (spec §3.4 audit fix); keep `strict=True` for trainers.
3. Default `candidate_pool = nse_all ∪ nifty500` (≈650) so mid-caps enter the ranking.
4. Emit + persist `universe_snapshot.json` per run; pass `as_of_date` at every fold.
5. Run the two backfill jobs (refresh + Tier-B mid-caps) before the RunPod GPU run.
6. Grow `DELISTED_NSE` + `CORPORATE_ACTIONS` as backfill surfaces gaps (both are hand-curated by design).


---

# Appendix C — Training observability & evaluation (full)

# ML Observability Design — How the Founder SEES Training, Outputs & Results

## TL;DR for the founder

You already own ~70% of a real MLOps observability stack — it just isn't **assembled into one place** and it **stops at the database row**. Today every model trains, evaluates with a serious metric library (rank-IC, DSR, PBO, SPA, impact-cost, Kelly), and writes a `model_versions` JSONB row + a `training_runs` history row that an admin page renders. What's missing is exactly the part you asked for: **(a) live streaming logs you can watch**, **(b) a persisted human report (JSON + markdown + plots) per run**, **(c) a version-comparison view** (so you can ask "is v5 better than v4?" at a glance instead of reading raw JSON blobs), and **(d) the per-fold curves** (rank-IC by fold, decile-spread, equity, feature importance) surfaced visually.

The right move is **NOT MLflow/W&B**. You have a registry that IS the source of truth, a reports-dir convention (`scripts/preflight.sh` already writes `reports/<ts>/`), a discovery/runner spine, and a working admin dashboard. Adding MLflow would duplicate `model_versions`, fight your B2 layout, and add a service to operate. The whole gap closes with **one new `TrainReport` artifact (JSON + .md + 3-4 PNGs) written into the run's `out_dir` and uploaded to B2 beside the model**, a **`run_log` capture in the runner**, **2 new read endpoints** (`/admin/ml/versions/{model}` and `/admin/ml/report/{model}/{version}`), and **one new "Model Detail" admin page** that ties it together. No new infra, no new service, no new datastore.

---

## What EXISTS today (verified against the live tree)

| Capability | Where | State |
|---|---|---|
| Uniform verbose log helpers (banner/step/fold_header/fold_result/epoch_progress, `flush=True`) | `ml/training/verbose.py` | **Solid.** Every trainer prints readable, streamable step output. |
| Unified runner with `RunReport` (name/status/duration/metrics/version/promoted/error) + per-trainer try/except + Sentry tagging + human + `--json` output | `ml/training/runner.py` | **Solid.** This is the orchestration backbone. |
| Per-fold OOS metric capture | trainers themselves: `momentum_lambdarank.py` (`rank_ic_per_fold`, `decile_spread_per_fold`), `lgbm_signal_gate.py` (`fold_metrics`, `dsr_pbo_from_fold_returns`), `regime_hmm.py` (`log_likelihood_per_obs_per_fold`) + `wfcv.aggregate_fold_metrics` → `_mean`/`_std`/`_per_fold` | **Exists but per-trainer & inconsistent.** Momentum trainer doesn't even subclass `Trainer` yet (standalone `train_momentum()`), so the runner can't see it. |
| Full metric library | `ml/eval/` — `backtest_eval` (Sharpe/Calmar/PF/`promote_gate_passes`), `overfitting` (DSR/PBO/CSCV), `spa` (Hansen SPA), `impact_cost` (Almgren-Chriss), `kelly`, `lambdarank_ic` (rank-IC obj+metric), `drift`/`drift_monitor` | **Solid + deep.** This is the crown jewel; better than most startups have. |
| Registry as source of truth: B2 bytes + Postgres `model_versions` (metrics JSONB, is_prod/shadow/retired, git_sha, trained_by, partial-unique prod index) + promote/shadow/retire/rollback | `src/backend/ai/registry/{model_registry,versions,b2_client}.py` + `migrations/2026_04_19_pr2_v1_ai_stack.sql` | **Solid.** `_json_safe` even sanitizes NaN/inf so degenerate folds still write a row. |
| Persisted run history | `training_runs` table (`migrations/2026_04_29_pr154_training_runs.sql`) — params + full `reports` JSONB + status, mirrored from in-memory dict | **Solid.** Survives restarts. |
| Promote gate enforced in runner | `runner.py` lines 199-260: financial gate + quality gate + safety-net + Kelly compute, reasons recorded | **Solid.** Already blocks bad models from `is_prod`. |
| Admin training UI: trainers list, run config (dry-run/skip-gpu/promote), trigger run, recent-runs with expandable per-trainer reports, 3s polling | `frontend/app/admin/training/page.tsx` ↔ `admin/training.py` (`/training/trainers`, `/training/runs`, `/training/run`, `/launch-readiness`) | **Solid** for *triggering & status*. Metrics shown as raw `JSON.stringify` though. |
| Admin ML dashboard: model cards (type/accuracy/features/path), regime panel, drift panel, retrain buttons | `frontend/app/admin/ml/page.tsx` + `admin/ml.py` (`/ml/performance`, `/ml/regime`, `/ml/drift`, `/ml/retrain`) | **Mostly real** (`/ml/performance` reads `model_versions` + `model_rolling_performance`). BUT `/ml/regime` returns **hardcoded** values (bull/0.87/2026-03-01) — a known stub. |
| Per-model PROD performance: live-IC-vs-backtest-IC, drift ratio, rolling windows table | `frontend/app/admin/model-performance/page.tsx` + `/ml/performance` + `drift_monitor.assess_one_model` | **Solid** for live/post-deploy monitoring. |
| Rolling realized stats (win-rate/Sharpe/dir-acc/DD per 7/30/90/365d) | `model_rolling_performance` table + weekly aggregation job | **Solid** (live drift, not training). |

## What is MISSING (the actual gap you're asking me to close)

1. **No live log streaming surface.** Confirmed: the only `stream` matches in `admin/training.py` are the word "running"; the runner uses `logging` to stdout (captured by `tee`/RunPod console only). The admin UI shows *status*, not *logs*. You cannot watch a fold print in the browser.
2. **No persisted human-readable report per run.** Confirmed: zero `.md`/`savefig`/`matplotlib`/report-generation anywhere under `ml/` or `scripts/` (except `preflight.sh` which only tees raw logs). Metrics live only as a JSONB blob. No plots ever.
3. **No version-comparison view.** `versions.list_versions()` exists in the Python registry but is **not exposed by any admin endpoint** (confirmed: no `/ml/versions` route). You can't diff v4 vs v5 metrics side-by-side.
4. **Per-fold curves are captured but never visualized.** The arrays (`rank_ic_per_fold`, `decile_spread_per_fold`) are in the JSONB but the UI dumps the whole blob as a string — no rank-IC-by-fold chart, no decile-spread bars, no equity curve, no feature-importance bars.
5. **Inconsistent per-trainer report shape** (each trainer rolls its own metric keys), and **momentum trainer isn't wired into the spine** (it's `train_momentum()`, not a `Trainer` subclass).

---

## The design: 5 layers, mostly REUSE/EXTEND, 2 BUILD-NEW

### Layer 1 — Live training logs (BUILD-NEW, thin)
**Capture, don't add infra.** In the runner's background worker (`admin/training.py::_worker` already runs the run on a daemon thread), attach a `logging.Handler` + tee `sys.stdout` into an in-memory ring buffer keyed by `run_id`, and flush periodically to `training_runs.run_log` (new TEXT column) + a `run.log` file in `out_dir` (uploaded to B2 with the artifacts). Add `GET /admin/training/runs/{run_id}/log?since=<offset>` returning the tail. Frontend `RunRow` (already polling every 3s) gets a "Logs" disclosure that long-polls the tail while `status==running`. The `verbose.py` `flush=True` output is *already* line-buffered for exactly this. This gives you the "watch it train" surface without a websocket or a log service.

### Layer 2 — Per-fold + final metrics capture (EXTEND, standardize)
Standardize what every trainer emits via the **9-stage pipeline spine** the readiness doc already specifies (`ml/training/pipeline.py`, BUILD-NEW per that doc — this design rides on it, doesn't re-spec it). The spine's `PipelineContext.metrics` becomes the single namespaced contract so `model_versions.metrics` is uniform across engines: always carry `primary_metric`/`primary_value`, `*_per_fold` arrays, `rank_ic_mean/std/icir`, `decile_spread_mean`, `deflated_sharpe`, `probability_backtest_overfitting`, `sharpe_mean`, `n_folds`, `n_rows/symbols/dates`, `feature_order`, and (new) `feature_importance` (LGBM `.feature_importances_` is free). `wfcv.aggregate_fold_metrics` already produces `_mean/_std/_per_fold` — point every trainer at it. **Wire `momentum_lambdarank` into a `Trainer` subclass** so the runner/registry/report sees it (today it's invisible to the spine).

### Layer 3 — Persisted TrainReport per model (BUILD-NEW, the centerpiece)
New module `ml/training/report.py` with `build_report(run_report, train_result, eval_metrics, out_dir)` that writes, into the run's `out_dir/<trainer>/` (which the runner already creates) and uploads to B2 beside the model:
- **`report.json`** — the full structured metrics (superset of the JSONB row; the row stays the queryable summary, the JSON is the complete artifact).
- **`report.md`** — human narrative: header (model/version/git_sha/trained_by/stack/data window), a metrics table, the promote-gate verdict + reasons, and "is this shippable?" one-liner. This is what you read.
- **PNG plots** via matplotlib (already in `.venv`; render headless on the pod, no new dep): `rank_ic_by_fold.png`, `decile_spread.png`, `equity_curve.png` (from `fold_return_arrays` the lgbm trainer already computes), `feature_importance.png`. Plots are generated **once on the GPU box at train time** and uploaded as bytes — the web app never needs matplotlib.
This is the missing "I can see the results" object. It's an artifact, so it versions and rolls back with the model for free.

### Layer 4 — Registry as source of truth (REUSE, expose more)
Keep `model_versions` as the truth (no MLflow). Add the **read endpoints that don't exist yet**: `GET /admin/ml/versions/{model_name}` (wraps `versions.list_versions` — already written, just not routed) and `GET /admin/ml/report/{model_name}/{version}` (resolves the version dir via `ModelRegistry.resolve`, returns `report.json` + presigned B2 URLs for the `.md` and PNGs). Add `report_uri` to the metrics JSONB (or a new column) so the dashboard can deep-link. Promote/shadow/retire/rollback already exist on the registry — expose `POST /admin/ml/versions/{model}/{v}/promote|rollback` so you can flip versions from the UI after reading the report.

### Layer 5 — Dashboard + CLI to compare versions & see curves (EXTEND + small BUILD-NEW)
- **CLI (REUSE):** `python -m ml.training.runner --json` already emits machine-readable reports; add `--report` to also write the Layer-3 artifacts locally, and a tiny `python -m ml.training.report --compare <model> v4 v5` that prints a side-by-side metrics table to the terminal (founder can diff without the UI).
- **UI (BUILD-NEW page):** `frontend/app/admin/models/[model]/page.tsx` "Model Detail": (1) a **version table** (v, trained_at, prod/shadow badge, rank-IC, Sharpe, DSR, PBO, gate-pass) with row-select to **compare two versions** (delta column, green/red); (2) the **per-fold charts** (rank-IC-by-fold line, decile-spread bars, equity curve, feature-importance bars) rendered from the embedded B2 PNGs (or re-charted client-side from `report.json` arrays — recharts is already in the stack); (3) the **rendered `report.md`**; (4) promote/rollback buttons. The existing `/admin/training` page links each `model_versions` row to this detail page. This is the single screen that answers "show me how this model trained and whether it's better than the last one."

---

## Concrete fixes also worth flagging (cheap, high-trust)
- `/admin/ml/regime` returns **hardcoded** bull/0.87 (lines 142-160 of `admin/ml.py`) — point it at the real `regime_history` table (data exists) so the dashboard stops lying.
- Admin training page renders metrics via `JSON.stringify` — replace with the structured per-fold cards once Layer 3 ships.
- Momentum trainer must become a `Trainer` subclass or it never appears in any of the above.

---

## Why NOT MLflow (justification, since you asked)
MLflow's three pillars are tracking (params/metrics/artifacts), a model registry, and a UI. You already have: the registry (`model_versions` + B2, with promote/shadow/retire/rollback semantics MLflow's basic registry lacks), artifact storage (B2 with a clean `<model>/v<n>/` layout), tracking (`training_runs` + per-fold metrics), and a UI (admin pages). MLflow would (a) introduce a second, competing registry you'd have to keep in sync with `model_versions` (the existing source of truth wired into serving via `ModelRegistry.resolve`), (b) require running/operating the MLflow server + its backend store, and (c) not understand your domain gates (DSR/PBO/SPA/impact-cost/regime-stratified Sharpe) — you'd still build custom plots. The 4 new modules/endpoints above are ~1-2 days of work versus standing up and migrating to a service. Reserve MLflow only if you later need multi-engineer concurrent experiment tracking at scale — not the case for a solo founder.


---

# Appendix D — Production repo reorganization (full)

## Quant X — Production Repository Reorganization (full-app)

### Why this matters now
The 4-engine ML/DL build (Momentum/Swing/Positional/Intraday) is about to add ~10 new shared-foundation modules (`pipeline.py`, `serve_smoke.py`, `specs.py`, `baseline_drift.py`, `factory.py`, `risk_engine.py`, per-engine feature builders + trainers) plus a TrueData provider. Those modules sit exactly on the fault line the 2026-06-15 audit blamed for **train/serve skew on every PROD model**. The current tree has no enforced boundary between *what is trained* and *what is served*, so the same code drifts into two copies. Fixing the structure first is the cheapest way to make "feature builder shared train==serve" (spec §4.1) and "serve-smoke before promote" (spec §4.4 / pipeline Stage 8b) structurally true instead of a convention people forget.

### The single root cause: there is no training↔serving boundary
The repo has two top-level Python trees, `ml/` (research/training) and `src/backend/` (serving/app), but they import each other in **both directions** and one of them does both jobs:

- `ml/` imports the serving app: `ml/data/data_loader.py` → `src.backend.data.providers.{base,free_provider,truedata_provider}`; `ml/data/kite_source.py` → `src.backend.services.kite_data_provider`; `ml/training/base.py` + `ml/training/trainers/lgbm_signal_gate.py` → `src.backend.ai.registry`; `ml/data/sentiment_history.py` → `src.backend.ai.sentiment.finbert_india`.
- `src/backend/` imports the research tree **at request/serve time**: `ai/signals/generator.py` → `ml.regime_detector` + `ml.scanner`; `platform/scheduler.py` → `ml.regime_detector` + `ml.training.runner` + `ml.eval.drift_monitor` + `ml.data.fii_dii_history`; `ai/feature_engineering.py` → `ml.features.lgbm_v2`; `ai/signals/options.py` → `ml.strategies`.

So `ml/` is **not** a clean research package — `ml/scanner.py`, `ml/risk_manager.py`, `ml/regime_detector.py`, `ml/strategies/`, `ml/backtest/` are serving/business modules misfiled under the training tree, and `ml/data/` duplicates the serving data layer in `src/backend/data/providers/`. The canonical `data_loader` even lives in `ml/` but wraps providers in `src/backend/`.

### Design principle for the target
Make **three concentric rings** with a one-directional dependency rule, enforced by an import-linter contract in CI:

1. **`core/` (pure domain + contracts)** — dataclasses (`MarketDepth`, signal `types`, `EngineSpec`/`LabelSpec`), provider Protocols, registry interface, output schema. Depends on nothing internal. Both rings 2 & 3 may import it.
2. **`ml/` (training & research)** — data factory, features, labeling, CV, eval, trainers, the 9-stage pipeline, RunPod entry. May import `core/`. **May NOT import the serving app.**
3. **`backend/` (serving app)** — FastAPI api/services/trading/data-serving/registry-impl/scheduler. May import `core/` and the **serialized model contract** in `core/`, plus call `ml/` only through the registry artifact path. **May NOT import `ml/` trainers at request time.**

The two legitimate `backend → ml` couplings today (regime features at serve, lgbm inference features at serve) are resolved by moving the *shared* feature/label code into `core.features` (pure, no-deps, train/serve-identical — which is exactly what spec §4.1 mandates anyway) and leaving only fit/HPO/eval in `ml/`. The scheduler's "kick off a training run" call moves behind a thin `ml.cli` subprocess/job boundary, not an in-process import.

### Naming conventions (locked)
- One import root. Drop the `src/` wrapper: app becomes `backend.api.app:app` (was `src.backend.api.app:app`). Add a `pyproject.toml` with `[tool.setuptools] packages = ["core","ml","backend"]` and editable install so `import backend` / `import ml` / `import core` work without sys.path hacks (kills the `sys.path.insert` lines in conftest + every backfill script).
- Layer-suffix modules consistently: trainers `*_trainer.py`, serving engines `*_engine.py`, feature builders `*_features.py`, label builders `*_labels.py`, providers `*_provider.py`, API routers `*_routes.py` (already the norm).
- Per-engine symmetry: for each style `X` there is exactly one `core/features/X_features.py` (shared), one `ml/training/trainers/X_trainer.py`, one `backend/ai/engines/X_engine.py`. A test asserts the trinity exists for every `EngineStyle` enum member.
- Artifacts: ONE local cache root `artifacts/models/<model>/<version>/` (merge `ml/models/`, top-level `models/`, `.model_cache/`); committed tiny configs allowed, large weights gitignored + sourced from B2. `data/` stays for static reference inputs only (universes, holidays, tiers), `var/` (gitignored) for runtime caches (bhavcopy, parquet caches).
- Scripts collapse into a thin `python -m ml.cli` / `python -m backend.cli` Typer app; only orchestration shell (`runpod_*.sh`, `dev.sh`, `qa/`, `release/`) stays under `scripts/`.

### Migration is phased, shim-guarded, test-gated — never big-bang
Every move uses a re-export shim at the old path (`from new.path import *  # DEPRECATED, remove after <date>`) so existing imports keep working; the full test suite (130 tests) + `uvicorn ... --check` import + a new import-linter contract run green after each phase before the next starts. Shims carry a removal date and a CI grep that fails if a NEW import uses an old path. This mirrors the codebase's existing shim policy (structural target 2026-05-25) and route-alias deprecation pattern.

### Files relevant to executing this
- `pyproject.toml` (new, repo root) — declares the 3 packages + editable install.
- `core/` (new top-level) — receives shared contracts/features/labels.
- `nixpacks.toml`, `Dockerfile`, `railway.toml`, `scripts/dev.sh` — the 4 places that hardcode `src.backend.api.app:app` and must flip to `backend.api.app:app` in Phase 1.
- `tests/conftest.py` + `scripts/*.py` — drop sys.path hacks once editable install lands.
- `ml/__init__.py` — currently documents a stale rule-based structure; rewrite to the new layout.
- Import-linter config (new) in `pyproject.toml` — encodes the 3-ring contract; wired into `.github/workflows/backend-ci.yml`.


---

# Appendix E — Dead/duplicate code to clean up

HEADLINE: there are TWO parallel ML training orchestration systems, and the cleanup hinges on collapsing to one. CANONICAL/current = `python -m ml.training.runner` (ml/training/discovery.py auto-discovers Trainer subclasses in ml/training/trainers/{lgbm_signal_gate,qlib_alpha158,regime_hmm,tft_swing,momentum_lambdarank}.py). Every current GPU entrypoint (runpod_full_pipeline.sh, runpod_smoke_pipeline.sh, runpod_train.sh, run_training_detached.sh, runpod_launch_run.sh, train_momentum_gpu.sh) and every helper (eda_report, smoke_all, validate_trainer, audit_trainer_data) rides this path. LEGACY/dead = scripts/train_all_models.py + src/backend/ai/training/__init__.py whose all_trainers() now returns an EMPTY list — it trains zero engines and is only launched by the equally-superseded pod_bootstrap.sh + preflight.sh.\n\nSafe DELETES (0 external refs, superseded): pod_bootstrap.sh, preflight.sh, train_tft.py, generate_demo_signals.py, plus the local-only untracked .pyc cruft (momentum_zero_shot.pyc etc. confirm deleted trainers).\n\nINVESTIGATE-then-delete (need a rewire first): retrain_pipeline.py is the trickiest — the scheduler already migrated off it but admin endpoint POST /ml/retrain STILL subprocess-launches it, AND it imports a NON-EXISTENT scripts/train_quantai.py (broken 'quantai' path) + the legacy train_lgbm.py + BreakoutMetaLabeler. Rewire the admin endpoint to ml.training.runner, then delete retrain_pipeline.py + train_lgbm.py. train_all_models.py + src/backend/ai/training/__init__.py go together once retrain wiring is settled. train_models_full.py (sole producer of the net-negative outcome models + the enabled-but-no-op RL q_table) and the backtest research harnesses (backtest_strategies.py, backtest_options_strategies.py, backtest_harness.py + its qa/backtest_smoke.sh + the BROKEN release/capture_rc_baseline.sh which references 3 non-existent modules/test files) are all off-CI legacy — confirm the founder doesn't still run them.\n\nKEEP (audit's 'dead but good machinery' is the reorg's TARGET infra, not cruft): ml/features/frac_diff.py is actually LIVE (called by lgbm_v2.py — audit claim stale); ml/eval/{spa,impact_cost}.py are slated to be wired per the readiness doc; schema tools (consolidate_schema, audit_schema) are deliberate maintenance entrypoints; train_ai_stock_ranker.py feeds a SHIPPED screener feature. exit_engine/{tick_exit,stagnation_trailing}.py + ai/microstructure/features.py are genuinely unwired (only __init__ re-export + test_pr_depth.py) but gated on a paid tick/L2 feed (Intraday/M4 deferred) — investigate against the roadmap rather than delete blind, since rl_exit_scaffold (the live one) lives beside them.</summary>
</invoke>


- **INVESTIGATE** `scripts/train_all_models.py` — LEGACY orchestrator. Walks src/backend/ai/training.all_trainers(), which now returns an EMPTY list (both stubs deleted 2026-05-31) — so this trains zero engines. The canonical orchestrator is `python -m ml.training.runner` (discovery over ml/training/trainers/), used by every current runpod pipeline + helper. Only callers are the equally-legacy pod_bootstrap.sh + preflight.sh; current runpod_full/smoke/train pipelines do NOT reference it. Recently touched (Jun 15) but only cosmetically.
- **INVESTIGATE** `src/backend/ai/training/__init__.py` — Backing registry for the legacy scripts/train_all_models.py orchestrator. all_trainers() returns [] (all trainer stubs deleted). EngineTrainer/TrainReport/TrainingContext protocol is a parallel, unused contract — the live system uses ml/training/base.py Trainer + ml/training/discovery.py. Only importer is scripts/train_all_models.py. Dead once that script goes.
- **DELETE** `scripts/pod_bootstrap.sh` — GPU pod launcher that calls the legacy `scripts/train_all_models.py --promote` (empty registry → trains nothing). Superseded by scripts/runpod_full_pipeline.sh / runpod_smoke_pipeline.sh which use `python -m ml.training.runner`. Zero external references (not invoked by any current pipeline/CI/doc).
- **DELETE** `scripts/preflight.sh` — Pre-training preflight that shells `scripts/train_all_models.py --dry-run` and `--only` (legacy empty-registry orchestrator). Superseded by validate_trainer.py / smoke_all.py on the ml.training path. Zero external references.
- **INVESTIGATE** `scripts/retrain_pipeline.py` — Legacy retrain dispatcher. Imports scripts/train_lgbm.py (audit's dead 15-feature raw-price model), ml/features/patterns.BreakoutMetaLabeler, ml/regime_detector, and scripts/train_quantai (which DOES NOT EXIST — the 'quantai' path is broken). Scheduler migrated OFF it (PR-T 2026-05-28, now uses ml.training.runner). BUT admin endpoint POST /ml/retrain (src/backend/api/admin/ml.py:234) STILL subprocess-launches it — so deleting breaks a live (super-admin) endpoint. Should be rewired to ml.training.runner, then deleted.
- **INVESTIGATE** `scripts/train_lgbm.py` — Audit-confirmed legacy LGBM trainer (raw absolute-rupee price levels, pooled cross-stock, no walk-forward/embargo/OOS, 3-class on accuracy). Superseded by ml/training/trainers/lgbm_signal_gate.py (v2, 30-feature, triple-barrier, AFML CV). Only live importer is the legacy scripts/retrain_pipeline.py (retrain_lgbm). src/backend/ai/model_registry.py only references it in comments. Delete after retrain_pipeline is rewired/removed.
- **DELETE** `scripts/train_tft.py` — Legacy standalone TFT trainer. Superseded by ml/training/trainers/tft_swing.py (the discovered trainer on the canonical runner). The only grep hit is its own docstring — ZERO external references (not in scheduler, retrain_pipeline, or any runpod pipeline).
- **KEEP** `scripts/train_ai_stock_ranker.py` — Trains src/backend/ai/ai_stock_ranker.py, which IS live: screener_routes.py 'AI Top Picks' endpoint + scheduler reference it and instruct running this script. NOT part of the new 4-engine spec (overlaps with the future Momentum ranker) but currently powers a shipped feature. Keep until Momentum engine replaces AI Top Picks; do not delete blind.
- **INVESTIGATE** `scripts/train_models_full.py` — Trains the 6 outcome XGBoost models (models/outcome/*) + RL exit Q-table (models/rl/q_table.json). Audit: outcome models at/below random (ema-crossover inverted, AUC 0.384, synthetic labels) and RL exit is a live no-op. ZERO references — not wired into any scheduler/pipeline/CI. Produces artifacts for the net-negative Gate-4 + RL-exit. RL exit is ENABLED per memory, so investigate before deleting (it's the only producer of q_table.json).
- **INVESTIGATE** `scripts/backtest_strategies.py` — Legacy research harness backtesting the 6 outcome-model strategies (the synthetic-label cluster the audit flags as net-negative). Imports ml.scanner + ml/features/patterns + ml/strategies/consolidation_breakout. ZERO external references (not in CI/pipelines). Standalone __main__ research tool.
- **INVESTIGATE** `scripts/backtest_options_strategies.py` — Legacy standalone options-strategy backtest research harness (29KB, __main__, 'strategy marketplace' backtests). ZERO external references in CI/pipelines/docs. Not on the 4-engine path (Intraday/F&O deferred, needs TrueData).
- **DELETE** `scripts/generate_demo_signals.py` — Self-labeled DEMO signal generator ('⚠ This is a DEMO signal generator. Production SignalGenerator...'). The audit's 'demo signal theater'. ZERO external references. scripts/run_real_signals.py is the real-SignalGenerator counterpart.
- **INVESTIGATE** `scripts/test_e2e.py` — Standalone E2E test script (18KB, May-25). NOT a pytest module under tests/; ZERO external references (not in CI, conftest, or docs). Superseded by the real pytest suite. Stale ad-hoc script.
- **INVESTIGATE** `scripts/runpod_launch_run.sh` — One-shot RunPod launcher using the CANONICAL `python -m ml.training.runner --all --promote` (+ requirements-train.txt). Current path but heavily overlaps with scripts/runpod_full_pipeline.sh. ZERO external references. Keep if the founder uses it as a quick one-shot; otherwise it duplicates runpod_full_pipeline.
- **KEEP** `scripts/run_training_detached.sh` — Detached background launcher using canonical `python -m ml.training.runner --promote --json`. Current path; ZERO external references. Small convenience wrapper overlapping the runpod pipelines. Harmless; keep or fold into runpod_full_pipeline.
- **INVESTIGATE** `scripts/release/capture_rc_baseline.sh` — BROKEN release-hardening baseline. References modules/files that DO NOT EXIST: ml.backtest.backtest_engine.BacktestEngine, ml.backtest.comprehensive_backtest.ComprehensiveBacktestEngine (ml/backtest/ only has engine.py), and backend/tests/test_long_strategies.py + test_signal_save_contract.py. NOT invoked by CI (release-hardening-gates.yml calls qa/*.sh directly). Stale and non-functional.
- **INVESTIGATE** `scripts/release/create_rc_ref.sh` — Companion RC-ref tagging script for the release-hardening flow. ZERO external references; not in CI. Pairs with the broken capture_rc_baseline.sh. Investigate the whole release/ dir's relevance to the current launch.
- **INVESTIGATE** `scripts/qa/backtest_smoke.sh` — QA wrapper around scripts/backtest_harness.py (--symbols/--period). NOT called by CI (release-hardening-gates.yml only runs backend_hard_gates/frontend_hard_gates/drift_gate). Only other reference is the broken capture_rc_baseline.sh. Manual smoke tool tied to the legacy backtest harness.
- **INVESTIGATE** `scripts/qa/frontend_execute_only_gate.sh` — QA gate NOT wired into release-hardening-gates.yml (which only runs frontend_hard_gates.sh). Single stale reference. Confirm whether superseded by frontend_hard_gates before removing.
- **INVESTIGATE** `scripts/backtest_harness.py` — 39KB backtest harness (yfinance + ml.scanner.get_all_strategies + ml.features.indicators). Imports resolve (get_all_strategies/compute_all_indicators exist). Only referenced by qa/backtest_smoke.sh + the broken release/capture_rc_baseline.sh — both non-CI. Investigate whether still the founder's go-to backtest tool or superseded by ml/backtest/engine.py.
- **INVESTIGATE** `src/backend/ai/exit_engine/tick_exit.py` — Per-tick walking exit engine (TickExitEngine/walk_ticks_for_exit/TickExitConfig). Audit-flagged dead: requires tick_data feed that isn't available; ZERO live (non-test) callers — only the __init__ re-export and tests/services/test_pr_depth.py import it. The live exit code (risk.py, options_backtest.py) imports exit_engine.rl_exit_scaffold, NOT this. Good machinery, unwired; keep only if intraday/tick feed is on the roadmap.
- **INVESTIGATE** `src/backend/ai/exit_engine/stagnation_trailing.py` — StagnationTrailingState/update_stagnation_trailing. Audit-flagged dead: ZERO live callers — only __init__ re-export + test_pr_depth.py. options_backtest.py implements its OWN inline 'stagnation-aware trailing' and does NOT import this module. Unwired good machinery.
- **INVESTIGATE** `src/backend/ai/microstructure/features.py` — Microstructure feature module (audit: dead, needs paid tick/L2 feed). ZERO live callers — only its own __init__ and tests/services/test_pr_depth.py. Not on the free-data 4-engine path (Intraday deferred, needs TrueData).
- **KEEP** `ml/features/frac_diff.py` — NOT dead — audit's 'unwired frac_diff' claim is STALE vs current code. frac_diff_ffd IS called by ml/features/lgbm_v2.py:158 (log_close_ffd_04), and lgbm_v2 is consumed by the live momentum_lambdarank + lgbm_signal_gate trainers and feature_engineering.py. Keep.
- **KEEP** `ml/eval/spa.py` — SPA (Superior Predictive Ability / multiple-testing) test. Audit: good machinery, not wired (only re-exported in ml/eval/__init__.py). The readiness doc explicitly plans to WIRE it (pipeline step 9 / DSR null for HPO). Keep — it is target infrastructure, not cruft.
- **KEEP** `ml/eval/impact_cost.py` — Almgren-Chriss impact cost model. Audit: good machinery, only re-exported in __init__, no runtime caller. Readiness doc plans to wire it into the gate. Keep as target infrastructure.
- **KEEP** `scripts/consolidate_schema.py` — Schema-consolidation tool (rebuilds complete_schema.sql cumulatively). Per memory 'Schema consolidation done 2026-05-08' this is the maintenance tooling behind Part A/Part B. ZERO code references but it's a deliberate maintenance entrypoint. Keep.
- **KEEP** `scripts/audit_schema.py` — Schema-consolidation audit (per-PR migration vs complete_schema.sql). Maintenance tool paired with consolidate_schema.py + the schema pytest guard. Keep.
- **DELETE** `scripts` — Local-only stale .pyc cruft (NOT git-tracked — .pyc is gitignored, 0 tracked). __pycache__ orphans for deleted modules confirm dead code: ml/training/trainers/__pycache__/momentum_zero_shot.cpython-312.pyc + intraday_lstm.cpython-312.pyc (deleted trainers), ml/__pycache__/position_manager.pyc, ml/ensemble/__pycache__/* (whole deleted package), ml/eval/__pycache__/{nse_costs,walk_forward}.pyc, ml/data/__pycache__/{kronos_features,fii_dii_moneycontrol}.pyc, src/backend/ai/__pycache__/intraday_lstm_predictor.pyc, src/backend/api/__pycache__/{ai_portfolio_routes,doctor_routes}.pyc, src/backend/ai/earnings/__pycache__/{strategy,predictor}.pyc, src/backend/ai/sentiment/__pycache__/gemini_classifier.pyc. Safe blanket cleanup: `find . -name '*.pyc' -path '*__pycache__*' -delete` (regenerated on next run; no repo impact).


---

# Appendix F — Completeness critic (gaps/risks/follow-ups)

**Gaps:**
- INTRADAY/M4 is under-specified for the data-licensing constraint that governs this whole product. The training+universe plans gate Intraday on a missing src/backend/data/providers/truedata_provider.py (confirmed absent; FreeDataProvider raises NotImplementedError for intraday freq at free_provider.py:106), but neither plan reconciles TrueData minute-bar/OI ingestion with the locked 'Data licensing Path A' memory (displaying NSE data to paying users needs per-user OAuth or an NSE licence). No answer on whether TrueData is licensed for centralized intraday signal generation/display, who pays, or the cost line. M4 is effectively un-plannable until that is resolved.
- Cross-engine model interaction / portfolio-level output is never addressed. The plans treat Momentum/Swing/Positional/Intraday as 4 fully independent rankers, but the founder's ask ('the 4-engine system') implies a user sees ONE coherent picture. There is no spec for: how a symbol ranked high by Momentum but low by Positional is reconciled in the UI, whether a fusion/ensemble layer exists (the memory mentions a shipped fusion_verdict.py weighted verdict — not referenced anywhere in these 4 plans), or how the existing 4 PROD models (regime_hmm/qlib_alpha158/tft_swing/finbert) relate to or are superseded by these 4 new engines. The reorg lists regime_hmm_trainer/qlib_alpha158_trainer/lgbm_signal_gate_trainer as surviving trainers but the training plan only covers the 4 new engines — the relationship between the old 4 PROD and new 4 engines is an unowned gap.
- The risk_engine (entry/SL/target from ATR) is named as spec §4.6 and given a home (backend/trading/risk_engine.py) in the reorg, but NO plan specifies its data contract, where ATR comes from at serve time, or how 'expected_return → levels' is computed. The training plan repeatedly says 'models never emit levels' but the consumer that DOES emit levels is hand-waved. This is the actual user-facing output (a tradeable signal needs entry/SL/target), so it is a load-bearing gap.
- Observability plan omits a serving-side report for the forecasters. It standardizes TrainReport for the LGBM artifacts (rank-IC/decile/equity/feature-importance) but the forecasters (TimesFM/Kronos/TFT/Chronos/PatchTST) have 'no standalone promote gate' — so the founder gets NO visibility into whether a forecaster column is actually contributing. The training plan says the contribution is measured as 'incremental rank-IC with vs without forecast columns' but the observability plan never surfaces this delta anywhere. The single most expensive part of the stack (GPU forecasters) has no dashboard answer to 'is it worth it?'.
- Universe plan's backfill jobs (refresh 310 stale Feb-2026 symbols + backfill ~190 Tier-B mid-caps) are listed as prerequisites but never scheduled, owned, or estimated. 'Two backfill jobs before the GPU run' is asserted with no runtime, no rate-limit handling for yfinance bulk pulls of 190 new symbols over 5.7yr, and no validation that the BOM-tolerant normalizer + corp-action adjust actually produce clean ADV. Given the OHLC-backfill history in memory (BOM/None-symbol PK-collision bug), this is a concrete unestimated risk treated as a footnote.
- No plan covers HOW training actually runs end-to-end on RunPod for the founder: the observability plan assumes 'GPU pod runs python -m ml.training.runner --report' and the training plan lists M0-M4 commands, but nobody owns the RunPod orchestration delta (the cruft report shows ~6 overlapping runpod_*.sh scripts and recommends folding into ml.cli — but the reorg defers CLI consolidation to Phase 6, AFTER the 4-engine build in Phase 4). So the founder will train the engines using the very scripts the reorg wants to delete, with no transition plan for that overlap.
- Migration plan has no rollback/abort procedure per phase. Each phase has a 'Gate' (tests green) but no 'if the gate fails, how do we revert' — critical because Phase 1 (src/backend→backend rename) and Phase 2 (carve out core/) touch import roots that, if broken, take down deploy. The shim strategy assumes forward-only success.

**Risks:**
- REPO REORG BLAST RADIUS IS UNDERSTATED. The reorg report enumerates the backend→ml couplings as 4 (generator.py, scheduler.py, feature_engineering.py, options.py) but verified reality is wider: src/backend/data/screener/engine.py (ml.features.indicators + ml.features.patterns.BreakoutMetaLabeler + ml.regime_detector), data/screener/market.py (ml.regime_detector.compute_regime_features + ml.features.indicators), services/probability_engine.py, services/chart_patterns/scanner.py + explain.py, services/indicator_interpreter.py, api/screener_routes.py, data/providers/free_provider.py (ml.data.production_ohlcv), api/admin/training.py. That is ~13 backend files importing ml across 10 distinct ml submodules (ml.features.indicators/patterns/lgbm_v2, ml.regime_detector, ml.strategies.options_base, ml.data.production_ohlcv/fii_dii_history, ml.eval.drift_monitor, ml.training.runner/discovery). Phase-2 'move regime_features + lgbm_features to core/' does NOT cover ml.features.indicators/patterns which are imported by 6+ live serving files — moving those is a much bigger surface than the plan budgets, and missing them leaves the cycle intact.
- tests/conftest.py uses sys.path.insert (verified). Phase 1 codemods `src.backend`→`backend` and relies on an editable install to drop the hack — but if the editable install isn't wired BEFORE the conftest hack is removed, test collection breaks for the entire 130-file/779-test suite, and the '130 tests green' gate becomes uncheckable. The phase ordering (flip import root, then later drop hacks) risks a window where neither the old path nor the new package resolves in CI.
- Trainer-rename convention (*_trainer.py) is NOT enforced by discovery and renaming carries hidden risk. discovery.py imports EVERY module under ml/training/trainers/ regardless of name (verified: it walks the package, no name filter), so renaming momentum_lambdarank.py→momentum_trainer.py is safe for discovery BUT: (a) every runpod script, doc, and the --only CLI target references the OLD module/trainer name; renaming changes the Trainer.name used by --only and topo-sort depends_on keys — a silent break of `--only momentum_lambdarank`. (b) The cruft report's whole 'two orchestrators' analysis keys off these exact filenames. Rename + reorg + cruft-deletion happening near each other multiplies the chance of a dangling reference.
- chronos-forecasting version contradiction will cause a silent install regression. Installed venv has chronos-forecasting==2.2.2 (verified) and the training plan correctly wants the Chronos-2 API (>=2.0), BUT requirements-train.txt AND requirements.txt both pin chronos-forecasting>=1.4.0 (verified). A fresh RunPod build from requirements-train.txt could resolve to a 1.x wheel that does NOT expose Chronos-2, breaking the Swing forecaster at install time on the pod while passing locally. The M0 'bump chronos floor' task is real and must land BEFORE any pod build, not as cleanup.
- timesfm + pytorch-forecasting are BOTH commented out in requirements-train.txt (verified lines 21 + 30) and timesfm's adapter history is fraught (memory: timesfm_adapter.py was deleted in the PR-M/PR-N removal). The training plan's M0 'un-comment timesfm (git --no-deps + jax pin)' and 'commit to neuralforecast-only TFT' are the long pole — neuralforecast is not even present in requirements-train.txt (verified: no neuralforecast line), so the Swing(TFT) and Intraday(PatchTST) engines have ZERO installed framework today. The plan asserts 'add neuralforecast' but treats a from-scratch GPU-stack install (timesfm git build + jax pin + neuralforecast + Kronos clone) as M0 housekeeping when it is historically the highest-failure-rate step.
- liquid_universe.py has a SILENT fail-soft fallback (verified: returns static NIFTY_200_FALLBACK[:top_n] when yfinance batch download fails) that directly violates the locked 'no fallbacks' / 'fail loud' memory rule AND the universe plan's own 'strict fail-loud' claim. The plan's fix ('rank from pg candles') must ALSO rip out this static-list fallback path, or training will silently build on a 200-name hardcoded list when pg/yfinance hiccups — producing a model trained on the wrong universe with no error. The plan mentions strict=True but does not call out removing the fallback branch.
- data/cache/ (310 _NS_10y.csv, verified present) is the entire basis of the universe plan's dataset-size math AND is simultaneously targeted by the reorg to MOVE to var/ (gitignored runtime). If the reorg's data/→var/ split (Phase 6) runs before or during the training milestones, load_ohlcv's cache-first path breaks and every engine silently falls through to yfinance (slow, rate-limited, non-reproducible). The two plans share data/cache/ as a hard dependency with NO coordinated ordering — reorg Phase 6 must come AFTER the GPU training run, or load_ohlcv's cache root must be updated atomically with the move.
- Observability run_log streaming tees sys.stdout into a ring buffer on a daemon thread (the plan's Layer 1). On a RunPod GPU box where training is the heavy workload, tee-ing all stdout (including multi-hour forecaster inference progress) into an in-memory buffer flushed to a Postgres TEXT column risks unbounded memory growth and DB write amplification during the longest runs — exactly when you most want logs. No cap/rotation on the ring buffer or the run_log column is specified.
- The admin POST /ml/retrain endpoint subprocess-launches scripts/retrain_pipeline.py (verified at admin/ml.py:234-238), which the cruft report confirms imports a NON-EXISTENT scripts/train_quantai.py + the dead train_lgbm.py. This is a LIVE super-admin endpoint that is already broken on the quantai path. The cruft plan's 'rewire to ml.training.runner then delete' is correct, but until done, any founder click of Retrain in admin launches a broken script — and the reorg's Phase 6 script consolidation could delete retrain_pipeline.py out from under this live endpoint if the rewire is missed.
- Momentum trainer is standalone train_momentum() not a Trainer subclass (verified — only MomentumConfig class + def train_momentum, no Trainer base). Three separate plans depend on fixing this (training M1, observability Layer 2 'wire into spine', reorg Phase 4) but NONE owns it as the FIRST blocking task. Until it subclasses Trainer, discovery.py cannot see it, the runner cannot --only it, the registry never gets a row, and the observability dashboard shows nothing — so M1a ('python -m ml.training.runner --only momentum_lambdarank --promote', the very first training command) cannot run as written today.

**Follow-ups:**
- Resolve the ordering contract between the reorg and the training+universe milestones explicitly: pin reorg Phase 6 (data/→var/ split, ml/models+models/+.model_cache→artifacts/, scripts→ml.cli) to run AFTER the full M0-M4 GPU training run completes, OR update load_ohlcv's cache root + registry artifact paths + every runpod_*.sh atomically in the same commit as the move. Document this dependency in both plan docs.
- Make 'convert momentum_lambdarank to a Trainer subclass' the explicit FIRST task of M0, before any other M0 work — it is the shared prerequisite of all three other plans and the first training command (M1a) cannot run without it. Verify --only momentum_lambdarank still resolves after the eventual *_trainer.py rename by adding an alias or keeping Trainer.name stable.
- Land the requirements hygiene as a standalone pre-M0 PR with a clean RunPod build verification: bump chronos-forecasting>=2.0 in BOTH requirements.txt and requirements-train.txt, add neuralforecast (currently absent), un-comment timesfm (git --no-deps + jax pin) and pytorch-forecasting OR commit to neuralforecast-only, and prove a fresh pod `pip install -r requirements-train.txt` succeeds and imports all 5 forecaster frameworks BEFORE training any engine. This is the highest install-failure-risk step and is currently buried as M0 housekeeping.
- Rip out the silent NIFTY_200_FALLBACK fallback in ml/data/liquid_universe.py (lines ~72, ~258-286) as part of the spec §3.4 pg-ADV fix — it violates the locked no-fallback/fail-loud rule and would train on a hardcoded 200-name list on any yfinance hiccup. Add a test asserting the builder RAISES (not falls back) when the pg ADV source is empty.
- Rewire admin POST /ml/retrain (src/backend/api/admin/ml.py:234) to call `python -m ml.training.runner` BEFORE the reorg/cruft phases touch scripts/, then delete scripts/retrain_pipeline.py + scripts/train_lgbm.py. Add a smoke test that the endpoint launches a resolvable command so the broken quantai path is caught.
- Write the missing risk_engine spec: define backend/trading/risk_engine.py's input contract (expected_return + confidence + symbol), its ATR source at serve time (which provider/lookback), and the entry/SL/target formula per engine horizon. This is the actual user-facing output and is currently only named, not specified, in any of the 4 plans.
- Add a forecaster-contribution surface to the observability plan: persist and dashboard the incremental-rank-IC delta (with vs without each forecast column) per engine retrain, so the founder can see whether the multi-hour GPU forecasters earn their cost. Today the contribution is computed transiently and shown nowhere.
- Re-scope reorg Phase 2 to cover the FULL shared-feature surface that serving imports: ml.features.indicators, ml.features.patterns (BreakoutMetaLabeler), ml.features.lgbm_v2, ml.regime_detector — all verified imported by 6+ live backend files (screener engine/market, probability_engine, chart_patterns scanner/explain, indicator_interpreter). Moving only regime_features+lgbm_features leaves the cycle intact. Produce the exact import-site inventory (the 13 backend files / 10 ml submodules verified here) as the Phase-2 worklist.
- Add a per-phase rollback procedure to the migration plan (git revert point + 'deploy preview must boot' gate per phase, especially Phase 1 import-root flip and Phase 2 core/ carve-out), and run import-linter in report-only mode through Phase 1-3 before flipping to enforce, so a missed import site surfaces as a warning rather than a CI hard-fail mid-migration.
- Cap and rotate the observability run_log ring buffer + Postgres TEXT column (size limit, tail-only retention) before enabling stdout-tee on multi-hour GPU runs, to avoid memory growth / DB write amplification during the longest forecaster inference loops.
- Resolve the TrueData-vs-Path-A licensing question as a written decision before any M4 work: is TrueData licensed for centralized intraday signal display to paying users, or does Intraday remain per-user-OAuth only? Until answered, mark M4 as blocked-on-legal in the training plan rather than merely 'gated on TrueData'.
- Reconcile the old 4 PROD models (regime_hmm, qlib_alpha158, tft_swing, finbert) and the shipped fusion_verdict.py against the new 4 engines in a single ownership doc: which are superseded, which coexist, and how the user sees one coherent verdict. The reorg keeps regime_hmm_trainer/qlib_alpha158_trainer/lgbm_signal_gate_trainer alive but the training plan ignores them — this relationship is currently unowned.

## ML Training Readiness + Canonical E2E Pipeline — Lead ML Engineer Synthesis

Two founder questions, answered bluntly:
1. **"Do we get ALL data for each model training?"** — **No.** We get *enough free data to start training 3 of the 4 engines today* (price-first), but **not all** the data each engine's spec promises. The shortfalls split cleanly into (a) **BUILD gaps** (data exists, code doesn't — TimesFM/Kronos adapters, RS-vs-index, the swing/positional feature builders) and (b) **DATA gaps** (data genuinely missing/thin — sectoral indices, fundamentals, FII/DII history, news-sentiment history, all intraday inputs). Engine 4 (Intraday) is a hard **needs-TrueData** stop.
2. **"Design a PROPER E2E pipeline."** — The repo already has a *mature* lifecycle (unified runner, Trainer protocol + drop-a-file discovery, M0-fixed labeling/CV, a full eval+overfit+drift suite, B2/model_versions registry). The gap is **structural, not missing primitives**: each engine trainer re-implements the 9 stages inline and they've drifted apart (the audit's root cause: train/serve skew on every PROD model). The fix is **one shared template-method spine** + 3 per-engine closures + **3 small new modules** (serve-smoke, specs, drift-baseline). All claims below were verified against the live tree, not the docs.

---

### (A) Data Availability — per-engine matrix + blunt verdict

The only **raw** input any of the daily engines actually consumes is **daily OHLCV** (+ a broad index + VIX). Everything else is either derived from that, or an enrichment family that is partial/missing. Verified: `data/cache/` holds **323** `*_NS_10y.csv` files, but the `_10y` name is a misnomer — they span **2020-05-27 → 2026-02-06 (~5.7yr / 1,415 rows)** and are stale (end Feb 2026).

| Field family | Used for | Momentum | Swing | Positional | Intraday | How to fill the gap |
|---|---|---|---|---|---|---|
| Daily OHLCV (CA-adjusted) | forecaster-input | have-free | have-free | have-free | n/a (needs intraday) | Present + wired (load_ohlcv → free_provider → production_ohlcv). Backfill pre-2020 / refresh stale via yfinance/pg. |
| Weekly/monthly bars | forecaster-input | n/a | n/a | have-free | n/a | yfinance interval (free_provider `_DAILY_FREQS` has week/month). |
| Price/return/vol/volume features | ranker-feature | have-free (built) | **build-new** (no swing_features.py) | **build-new** (no positional_features.py) | build-new | Only `momentum_features.py` exists. Author swing/positional builders; reuse momentum building blocks. |
| Cross-sectional rank features | ranker-feature | have-free | have-free | have-free | partial | `groupby(date).rank(pct=True)` pattern proven in momentum_features. |
| RS-vs-INDEX (Nifty) | ranker-feature | **partial (DATA yes, CODE no)** | have-free | partial | n/a | NSEI cached; `MOMENTUM_FEATURE_ORDER` has ZERO index cols. Pure feature code. |
| RS-vs-SECTOR | ranker-feature | **missing (data+code)** | partial | partial | n/a | No sectoral-index series cached; sector map only 104 symbols. Ingest ~11-18 free sectoral indices + wire full-universe map (Supabase 2,385 EQ). |
| TimesFM / Kronos forecast columns | forecaster-input | **partial (input free, NO code)** | n/a | **partial (NO code)** | n/a | Verified: timesfm/kronos appear only as string constants/enums; `kronos_features.py` is a stale `.pyc`. Install + write adapters. |
| TFT / Chronos-2 forecast columns | forecaster-input | n/a | have-free (TFT trainer exists; Chronos adapter unwired) | n/a | n/a | TFT `tft_swing.py` present; add Chronos-2 column emitter. |
| India VIX / regime context | context | have-free (unused) | have-free | have-free | partial (daily only) | INDIAVIX cached; regime_hmm trained. |
| Absolute fwd-return quantile labels | label | have-free | have-free | have-free | have-free | `ranking_labels.forward_return_quantile_labels` built + absolute. |
| Triple-barrier (intra-bar) labels | label | have-free | have-free | have-free | **have-free (ready, no bars to label)** | M0 intra-bar fix verified in `triple_barrier.py`. |
| Liquid universe (top-N ADV) | universe | have-free | have-free | have-free | partial | Built, strict fail-loud. CAVEAT: ranks ADV via LIVE yfinance, not pg (spec wants pg). |
| Fundamentals (quality/value/growth) | ranker-feature | not needed | **partial (cache empty)** | **missing (cache empty)** | n/a | `fundamentals_pit.parquet` = **2 rows / `_TEST_` symbol**. Backfill → only ~8 restated quarters. |
| Ownership/flow per-symbol (promoter/FII/DII delta) | ranker-feature | n/a | **missing (NaN by design)** | **missing** | n/a | No free historical NSE shareholding feed. Needs scraper or paid. |
| Market-wide FII/DII history | ranker-feature/context | n/a | partial (7 rows) | partial (7 rows) | n/a | `fii_dii_history.parquet` = **7 rows**. Today-only source; accumulate forward via 17:00 cron or buy. |
| News sentiment history | ranker-feature | n/a | **partial (parquet absent)** | missing (omitted in spec) | n/a | `sentiment_history.parquet` ABSENT; Google News today-only. Accumulate forward. |
| Earnings PIT (days-to-earnings) | ranker-feature | n/a | partial (upcoming only) | n/a | n/a | No historical earnings-date series free. Live-gate only. |
| Intraday 1/3/5/15-min bars + OI + option IV/Greeks + intraday VIX/breadth | all | n/a | n/a | n/a | **needs-truedata** | `FreeDataProvider.get_ohlcv` RAISES NotImplementedError for intraday (verified line 80); `DataProvider` Protocol has ONLY `get_ohlcv`; no `truedata_provider.py`. |

**Blunt verdicts:**
- **Momentum — partial-degraded.** Train the LGBM LambdaRank NOW on price-only features. NOT the specced stack until TimesFM/Kronos adapters are written, RS-vs-index is coded (data is present), and RS-vs-sector data is sourced.
- **Swing — partial-degraded.** A real price+forecast+regime ranker trains today; the fundamentals/FII-DII/sentiment/earnings columns will be ~zero-filled on any historical window. `swing_features.py` must be authored.
- **Positional — partial-degraded.** Train price-first once Kronos/TimesFM are installed and `positional_features.py` is written; quality/value/growth + ownership factors are sparse zero-fills.
- **Intraday — no, needs-TrueData.** The deliberate M4 hold. Essentially none of the intraday-specific inputs are free; the provider, feature builder, and PatchTST trainer are all unbuilt. Only the triple-barrier labeler is ready (for bars that don't exist).

---

### (B) Model / library readiness + GPU

| Forecaster | Library | In requirements? | GPU | Status |
|---|---|---|---|---|
| Chronos-2 (Swing aux) | chronos-forecasting | yes (>=1.4.0 both files) | optional | Ready; local 2.2.2 exposes Chronos-2 classes. **Bump floor to >=2.0.0**. Lowest risk. |
| LightGBM ranker (all 4) | lightgbm | yes (>=4.0/4.1) | no (CPU-native) | Ready; `LGBMRanker` present. Lowest risk. |
| TFT (Swing) | neuralforecast (primary) / pytorch-forecasting (fallback) | **NO neuralforecast; pytorch-forecasting COMMENTED OUT** | yes | **High urgency** — fresh deploy ships dark (audit). Add neuralforecast; re-enable pytorch-forecasting. Train/serve framework split to unify. |
| PatchTST (Intraday) | neuralforecast / transformers PatchTST | not in req (transformers PatchTST present) | yes | Reachable via transformers with zero installs. Gated on TrueData, not the lib. |
| TimesFM (Momentum+Positional) | git/PyPI + transformers | commented out (PR-M cut) | yes | Needs git `--no-deps` + jax[cpu]<0.5 pinning. Medium-high install friction. |
| Kronos (Momentum+Positional) | clone+PYTHONPATH (not pip) | external | yes | **Highest risk**: unpinned HEAD clone (branch `master`), HF weights, **adapter deleted (stale `.pyc` only)** — must rewrite. |

**GPU:** Training on RunPod (RTX 4090 ~24GB, ~$0.69/hr). GPU-required: TimesFM/Kronos/TFT/PatchTST. GPU-optional: Chronos-2, LightGBM. Pipelines hard-abort if `torch.cuda.is_available()` is False. Local box is CPU-only (fine for M0 + import smoke, not forecaster training). Sequence per milestone M0→M1→M2→M3→M4, not one run.

**Net:** the RunPod scripts already encode working install recipes for all 5, but `requirements*.txt` are stale vs the new 4-engine spec — that hygiene fix is the highest-urgency library task (TFT deploy-dark).

---

### (C) Canonical 9-stage E2E training pipeline (REUSE vs BUILD-NEW)

A **template-method spine** (`run_pipeline(ctx)`) runs 9 stages in fixed order with fail-loud gating and a single namespaced `PipelineContext.metrics` dict (so `model_versions.metrics` is uniform across engines). Stages **1, 2, 5, 8, 9** are 100% shared; **3, 4, 6, 7** call per-engine closures.

| Stage | Purpose | Disposition | Where |
|---|---|---|---|
| **0. Skeleton** | One canonical lifecycle | **BUILD-NEW** | `ml/training/pipeline.py` (Stage enum + PipelineContext + run_pipeline) + extend `ml/training/base.py` |
| **1. EDA / data analysis** | NaN/skew/balance/IC/leakage pre-train audit, HARD-FAIL | **REUSE+EXTEND** | `ml/preprocessing/eda.py` (5 funcs verbatim); fold `scripts/eda_report.py` INTO the stage so EDA runs on the trainer's real builders (kills drift) |
| **2. Cleaning / quality gate** | Stale/dup/gap/outlier/dead-column detection; drop or fatal-fail | **REUSE** | `ml/data/quality_check.py` (run_quality_checks + QualityCheckConfig + audit_feature_matrix) — make mandatory for qlib + tft too |
| **3. Labeling** | Triple-barrier / fwd-return quantile / TFT 5d / none(HMM) | **REUSE (M0)** | `ml/labeling/{triple_barrier(intra-bar fix), ranking_labels, sample_weights}.py`; engine `build_labels` closure. t1 → Stage-4 weights AND Stage-5 embargo |
| **4. Feature engineering** | ONE builder shared train==serve; persist frozen feature_order + fitted transformer sidecars | **REUSE+EXTEND** | `ml/features/{momentum_features(M0), lgbm_v2, indicators, frac_diff}.py`; engine `build_features` closure |
| **5. Split / purged WFCV + CPCV** | Train strictly before test, day-sized embargo ≥ horizon, purge overlaps, CPCV for PBO | **REUSE (M0)** | `ml/training/{purged_cv(by_date, day embargo), cpcv, wfcv}.py` |
| **6. Training** | Per-fold fit (OOS metrics) + final fit on all data (shipped artifact) | **REUSE+EXTEND** | engine `fit_model` closure (existing lgbm/qlib/tft/hmm fit code); skeleton owns the loop; AFML weights to every supervised engine |
| **7. HPO (opt-in)** | Optuna TPE on the SAME OOS metric; n_trials → DSR null | **REUSE** | `ml/training/optuna_search.py`; engine `search_space` hook (no trainer wires it today) |
| **8. Evaluation** | WF financial metrics + rank-IC/NDCG + DSR/PBO + SPA + Almgren-Chriss impact (deducted before Sharpe) + drift baseline | **REUSE** | `ml/eval/{backtest_eval, lambdarank_ic, overfitting, spa, impact_cost, drift}.py` (spa + impact wired in per audit) |
| **8b. Serve smoke-load** | Round-trip artifact THROUGH the real production predictor; block promote on feature-order/normalization/shape mismatch | **BUILD-NEW** | `ml/training/serve_smoke.py` + `predict_for_serve` hook — **the #1 audit fix** |
| **9. Results / registry / promote gate** | B2 upload, model_versions row, flip is_prod only if financial+overfit+quality+serve-smoke ALL pass | **REUSE+EXTEND** | `ml/training/runner.py` (extend gate to require serve_smoke) + `backend/ai/registry/model_registry.py` |

**New modules (all audit-driven):** `ml/training/pipeline.py`, `ml/training/serve_smoke.py`, `ml/training/specs.py` (EngineSpec/LabelSpec/FeatureSpec/CVSpec), `ml/training/baseline_drift.py` (writes the PSI/KS train-window baseline so `ml/eval/drift_monitor.py` can actually fire — verified inert today). Plus 3 test files. Everything else is reuse/extend. Verified absent: none of the 4 new modules exist yet.

---

### (D) How a per-engine Trainer plugs into the spine + unified runner

```
runner.py (UNCHANGED orchestration: discovery → topo-sort on depends_on →
           per-engine train/eval/register → promote-gate → Kelly → safety-net → Sentry)
   │  (one level above the spine; ONLY change = serve-smoke precondition added to its gate)
   ▼
base.Trainer.train()  ──delegates──▶  pipeline.run_pipeline(ctx)
   │                                      runs 9 stages, fail-loud, returns TrainResult
   │  engine overrides ONLY 3+1 hooks:
   ├── build_features(panel) -> X, feature_order, transformer   (Stage 4)
   ├── build_labels(ohlcv)   -> y, t1, label_dist               (Stage 3)
   ├── fit_model(X,y,w)      -> fitted model                    (Stage 6)
   └── predict_for_serve(artifact_dir, symbols) -> scores       (Stage 8b serve-smoke)
```

Each engine = a small declarative `EngineSpec` (`ml/training/specs.py`) + those 3 closures, instead of a ~700-line bespoke trainer. The trainer file still drops into `ml/training/trainers/` and auto-registers via `discovery.py`; the runner still resolves `depends_on` topologically (e.g. swing/positional depend on regime + forecasters). The whole thing runs unchanged via `python -m ml.training.runner --all` / `--promote`. The existing `lgbm_signal_gate / qlib_alpha158 / tft_swing / regime_hmm` get refactored so their inline stages become these hooks — collapsing the copy-paste drift that caused train/serve skew on every PROD model.

---

### (E) What this means for M1 (Momentum), concretely

**You can start the LGBM LambdaRank training run today — degraded but real — and it is the right first move because it forces the whole spine into existence.** Order of operations:

1. **Build the spine first** (specs.py → base hooks → pipeline.py → wire stages 1/2/5/8/9 from existing modules → serve_smoke.py → extend runner gate). This is mostly WIRING; the primitives all exist and are M0-verified.
2. **Train Momentum price-only NOW**: `build_features` = current `momentum_features` set (13 price/volume/RS-cross-sectional cols, all have-free), `build_labels` = `forward_return_quantile_labels(horizon=20)` with 30/60d extra heads, `fit_model` = `LGBMRanker` LambdaRank. CPU-trainable, fast. This produces a genuine cross-sectional momentum ranker and exercises stages 1-9 + serve-smoke end-to-end.
3. **Close the two pure-code feature gaps**: add RS-vs-index columns (NSEI is cached — append to `MOMENTUM_FEATURE_ORDER`, keep train/serve-shared), then source + build RS-vs-sector (ingest ~11-18 free sectoral indices + wire the full-universe Supabase sector map).
4. **Then add the forecaster columns** (the spec's TimesFM + Kronos): install on RunPod GPU (TimesFM git `--no-deps`+jax pinning; Kronos clone@master + **rewrite the deleted `kronos_features.py` adapter**), emit forecast feature columns into the shared builder, retrain. This is what upgrades M1 from degraded → specced.
5. **Requirements hygiene in parallel** (add neuralforecast, re-enable pytorch-forecasting, un-comment timesfm, bump chronos floor) — needed before M2 (Swing/TFT) anyway and fixes the audit deploy-dark finding.

**Honest M1 caveat for the founder:** the first Momentum model that promotes will be a **price + cross-sectional + RS-vs-index ranker** — a legitimate, ship-able momentum engine — but it is **not** the "TimesFM + Kronos foundation-forecaster" stack the spec headlines until step 4 lands. Serve-smoke (Stage 8b) is the guarantee that whatever promotes is exactly what serves — no more 15-vs-30 feature skew. No TrueData, no fundamentals, no flow/sentiment needed for Momentum (spec §5.1: "No fundamentals"), so there is no external data blocker — only the sectoral-index ingest and the two adapter writes.

**Key files to read:** `ml/training/{pipeline.py NEW, runner.py, base.py, purged_cv.py, optuna_search.py, smoke.py, discovery.py}`, `ml/preprocessing/eda.py`, `ml/data/{quality_check.py, production_ohlcv.py, liquid_universe.py, data_loader.py}`, `ml/labeling/{triple_barrier.py, ranking_labels.py, sample_weights.py}`, `ml/features/{momentum_features.py, lgbm_v2.py, indicators.py, frac_diff.py}`, `ml/eval/{backtest_eval.py, overfitting.py, lambdarank_ic.py, spa.py, impact_cost.py, drift.py, drift_monitor.py}`, `backend/ai/registry/model_registry.py`, `backend/data/providers/{base.py, free_provider.py}`, and `docs/ML_DL_DEEP_AUDIT_2026_06_15.md` (§1 + §7 sequence the fixes the pipeline must enforce).
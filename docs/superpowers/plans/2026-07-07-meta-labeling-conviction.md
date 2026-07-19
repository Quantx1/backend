# Meta-Labeling Conviction Score — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax. Spec: `docs/superpowers/specs/2026-07-07-meta-labeling-conviction-design.md`.

**Goal:** Calibrated P(win) classifier per style engine (momentum, swing) trained on walk-forward OOS fold predictions; ships as a conviction field only if pre-registered gates pass.

**Architecture:** Fold-prediction persistence in the backtest harness → engine-agnostic meta feature builder (preds cross-section + regime series + selected engine feature columns + market context) → LGBM binary classifier through the 9-stage spine (new classification evaluation branch) → isotonic calibration on pooled OOS folds → gate verdict. Serving/UI only after gates pass (separate tasks, no-fallbacks rule).

**Tech stack:** pandas, LightGBM, scikit-learn isotonic, existing spine (`ml/training/pipeline.py`), purged CV, regime series parquet.

**Build state note:** executed inline in-session 2026-07-07; tasks below record the contract each change must satisfy.

---

### Task 1: Persist per-name OOS fold predictions

**Files:** Modify `ml/eval/walkforward_backtest.py`, `scripts/eval/backtest_engines.py`; test `tests/ml/eval/test_walkforward_backtest.py`.

- [ ] `walkforward_portfolio_backtest(..., dump_preds_path: Optional[Path] = None)`: inside the fold loop collect `te[["date","symbol","pred", fwd_return_col]]` + `fold`; after the loop write one parquet (columns `date, symbol, fold, pred, fwd_return`) to the path; add `"fold_preds_path"` to the result dict when set. Predictions are already OOS by construction (purged folds).
- [ ] CLI: `--dump-preds` flag in `backtest_engines.py` → `artifacts/eval/fold_preds/{engine}_preds.parquet`.
- [ ] Test: fake trainer (reuse the module's existing fixture pattern) → parquet exists, row count == n_test_dates × names, preds match the return-dict fold count, no NaT dates.

### Task 2: Classification evaluation branch in the spine

**Files:** Modify `ml/training/specs.py`, `ml/training/pipeline.py`; test `tests/ml/training/test_classification_eval.py`.

- [ ] `EvalSpec` gains: `min_auc: float = 0.0`, `min_fold_auc: float = 0.5`, `min_tercile_lift: float = 0.0` (top-tercile win rate − base rate), `require_brier_beats_climatology: bool = False`.
- [ ] `_cv_and_fit`: when `spec.eval.task == "classification"`, fold frames carry `label` (from `spec.label_col`) and `pred` = `predict_proba[:,1]` when available.
- [ ] `_stage_evaluation` classification branch: per-fold AUC, Brier, base rate, tercile win rates (`pred` terciles per fold); metrics `auc_mean/auc_per_fold/brier_mean/base_rate/tercile_win_rates/tercile_lift`; `primary_metric="auc_mean"`. Gate = all four spec thresholds (§6 of the spec). Ranking path byte-identical when task != classification.
- [ ] `_oos_rank_ic_for_params` (HPO objective): classification → mean OOS AUC.
- [ ] Tests: synthetic separable dataset passes gates; label-shuffled dataset fails; ranking engines' metrics unchanged (regression guard).

### Task 3: Meta feature builder

**Files:** Create `ml/features/meta_features.py`; test `tests/ml/features/test_meta_features.py`.

- [ ] `META_ENGINE_MAP`: per-engine name-feature columns (momentum: `realized_vol_63/beta_index_252/amihud_illiq_63/dist_high_252`; swing: `realized_vol_21/beta_index_63/rel_volume_21/pullback_from_high_21`) + forecast cols (+ ens); validated against the engine frame — missing column = raise.
- [ ] `build_meta_features(preds, engine_feats, engine, regime, benchmark) -> (df, cols)`:
  signal context (`score_pct` per-date percentile, `score_z`, `score_dispersion`, `score_gap` to next rank / dispersion); regime one-hot + `regime_confidence` + `days_since_switch` (state-change cumcount, causal); market from benchmark closes (`mkt_rv21`, `mkt_ret_21`, `mkt_dist_high_63`) merged per date; name + forecast cols joined from the engine frame on (date, symbol); `tsfm_kronos_spread` = |tsfm−kronos|. All inputs at date t; no future joins (regime/benchmark merged with `how="left"` on exact date only).
- [ ] Test: PIT check (shifting regime forward changes only post-shift rows), per-date percentile correctness, engine-map validation raises on missing column.

### Task 4: Meta trainer + calibration

**Files:** Create `ml/training/trainers/meta_conviction.py`; test `tests/ml/training/test_meta_conviction.py`.

- [ ] `MetaConvictionConfig(engine, cost_bps_side=30.0, hpo_trials=0, cv=…)`; horizon inherited from the engine trainer's spec. Label `meta_win = (fwd_return − 2×cost) > 0`; `net_fwd_return` kept as `fwd_return_col`.
- [ ] `MetaConvictionTrainer(PipelineTrainer)`: `load_panel` reads the fold-preds parquet (raise with instructions if absent) + engine feature frame via the engine trainer's own hooks + regime + benchmark; `build_features` = Task 3 builder; `build_labels` from preds; `make_model` = `LGBMClassifier` (≤200 trees, depth ≤ 4); `engine_spec` → `task="classification"`, gates: `min_auc=0.55, min_fold_auc=0.5, min_tercile_lift=0.05, require_brier_beats_climatology=True`; CV folds aligned to primary via same date-based purged splitter, embargo = engine horizon.
- [ ] Post-spine calibration: pool OOS fold preds → `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`; persist `calibration.json` (x/y knots) + `conviction_bands.json` (calibrated tercile cut points). Loud failure if fold preds are degenerate.
- [ ] CLI `__main__`: `--engine momentum|swing --hpo-trials N --out-dir …`.
- [ ] Tests: mocked engine hooks → end-to-end artifact save; calibration knots monotone in [0,1]; gate fail on shuffled labels ⇒ `quality_pass=False` and NO calibration artifact written.

### Task 5: Generate data + train + verdict (execution)

- [ ] Re-run expanded backtests with `--dump-preds` for momentum + swing (CPU, reuses tuned params).
- [ ] Train both meta models; record `metrics.json`; verdict per engine against §6 gates.
- [ ] Gate FAIL ⇒ stop; report honestly; no serving/UI work. Gate PASS ⇒ Tasks 6–7.

### Task 6 (gated): Serving

**Files:** Modify `backend/platform/scheduler.py` (style-signals cron), engine serving payloads; tests.
- [ ] Registry-first meta model load; conviction computed from live engine frame + regime + benchmark; attach `conviction {score, band, pct}` to snapshot + API; absent model/inputs ⇒ field omitted (degraded mode, never fabricated).

### Task 7 (gated): UI conviction chip

**Files:** Signal card + `/signals/[id]` detail; duotone tokens; copy "Conviction"; Free = hidden, Pro = band, Elite = band + percentile.

---

**Self-review:** spec §3 label ✓ (Task 4), §4 features ✓ (Task 3, per-engine map), §5 model/CV ✓ (Task 4), §6 gates ✓ (Tasks 2+4+5), §7 serving ✓ (Task 6, gated), §8 sizing explicitly OUT (waits paper window), §9 evidence via Task 5 backtest reuse. Type consistency: fold-preds parquet columns fixed in Task 1 and consumed verbatim in Task 4.

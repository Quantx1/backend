# Quant X — Deep ML/DL/AI Audit (2026-06-15)

**Scope:** Every ML/DL/RL/LLM model and the training + serving + gating pipelines behind them.
**Method:** 18-subsystem multi-agent audit (map → adversarial audit → verify). 183 deduplicated
findings (28 critical / 55 high / 67 medium / 33 low). 27 critical/high findings were independently
re-verified against code (0 refuted) before a session-limit cut the verify tail short — those are
marked **✅ verified** below. The rest are single-pass findings (high precision, but re-confirm before
acting on the medium/low tail).

---

## 1. Executive summary — read this first

**The headline: the four models we call "PROD" (tft_swing v3, qlib_alpha158 v4, regime_hmm, lgbm_signal_gate)
are not the models actually scoring live money. Every one of them has a train/serve break, and the
quality gates that are supposed to stop a bad model going live are no-ops.** The training pipeline is
real and mostly well-engineered; the *serving* pipeline silently consumes stale/legacy artifacts on a
different feature contract. So promotions are theater — retraining a model often has **zero effect on
inference**, and in two cases would **break signal generation entirely**.

Eight systemic themes:

1. **Train/serve skew on every supervised voter.** TFT serves a stale March checkpoint (not v3);
   Qlib serves raw features to a model trained on normalized features; LGBM serves a legacy
   15-feature price-level model while the "real" 30-feature trainer is dead code.
2. **The promote/quality gates are dead or bypassable.** Qlib's IC gate is skipped by a dict-key typo;
   HMM auto-promotes nightly with no threshold; all 4 PROD models bypass the financial gate; the
   strategy gate's "out-of-sample" number is computed from the in-sample backtest.
3. **The drift auto-rollback safety net never fires** — the one automated defense against silent model
   decay in a live-money product is permanently inert.
4. **Outcome models (Gate 4) are at/below random and one is inverted** (ema-crossover AUC 0.384 — it
   blocks winners and passes losers). They actively degrade P&L.
5. **The RL early-exit agent is a guaranteed no-op live** (feature skew means it never reaches an EXIT
   state), and its reward function never even taught early exit. Shorts are scored with inverted sign.
6. **Strategy discovery is largely theater** — engine signals never load (40-50% of candidates scored
   crippled), F&O search produces zero trades, and there is zero multiple-testing correction over
   thousands of nightly trials.
7. **The good ML machinery is dead code.** The whole López de Prado stack (fractional differentiation,
   triple-barrier, sample-uniqueness weights), the SPA test, Almgren-Chriss impact cost, and PSI/KS
   drift detection have **no runtime callers**.
8. **Data leakage in training.** Fundamentals broadcast a single as-of-today snapshot across years of
   history; triple-barrier labels ignore intra-bar High/Low; embargoes are mis-sized.

**What this means for "we're going to work on ML/DL models":** before training anything new, the
serving contract and the gates must be fixed, or new models will keep landing in a pipeline that
ignores or breaks on them. The work plan in §7 sequences this.

---

## 2. Model inventory matrix (what's actually live)

### Supervised / statistical "PROD" models — the swing-signal ensemble
The ensemble is a weighted vote: **TFT 0.30 + LGBM 0.30 + Qlib 0.20 + HMM/regime overlay**, with a
`min_agreement` gate. Every voter below has a serving defect.

| Model | Type / framework | Declared | **Reality** |
|---|---|---|---|
| **tft_swing** | DL · TFT | v3 neuralforecast, "68% dir-acc", weight 0.30 + source of entry/SL/TP | ✅ Serves a **stale Mar-13 pytorch-forecasting `.ckpt`** with different features (6 vs 12), smaller net (hidden 32 vs 128), and it **"forecasts" the last 5 *observed* bars, not the next 5**. v3 artifact is never loaded. `pytorch-forecasting` is **commented out of `requirements.txt`** → fresh deploy = signal pipeline silently dark. |
| **qlib_alpha158** | ML · Qlib Alpha158 + LightGBM | v4, weight 0.20 + AutoPilot top-N ranker | ✅ Serving feeds **raw, un-normalized features** to a model trained on RobustZScoreNorm'd features. OOS gate measures the *same* broken pipeline. Ranks **yesterday's** data and mixes **different dates** into one cross-section. `direction_agrees` uses universe **rank, not predicted sign**. |
| **regime_hmm** | statistical · hmmlearn 3-state Gaussian HMM | drives AutoPilot gross exposure (bull 80% / bear ~24%) | ✅ **Auto-promotes nightly with no quality threshold** and no incumbent compare → can flip live regime labels overnight. Falls back to **"bull" (most aggressive)** on error/thin data. VIX-missing path **fabricates** a feature the model never trained on. |
| **lgbm_signal_gate** | ML · LightGBM 3-class | v2 (30 feat, triple-barrier, AFML), weight 0.30 | ✅ Serves a **legacy 15-feature model trained on raw absolute rupee price levels**, pooled cross-stock, **no walk-forward / no embargo / no OOS Sharpe** (shipped on accuracy). Its registry row is **inactive**. The v2 trainer is **dead code**; promoting it would feed 15 keys to a 30-feature model → `KeyError` per symbol → **zero signals**. |
| **finbert_india** | DL · HF transformers (Vansh180/FinBERT-India-v1) | standalone "Mood" engine | Pulled live from HuggingFace (no local artifact, no version pin). Removed from the signal ensemble (2026-06-06) — Mood is on-demand only. Launch gate hard-requires a `finbert_india` PROD registry row **that no code ever writes**. |

### Outcome models — strategy_runner "Gate 4" (P(win) filter before live/paper entry)
| Model | Type | Status |
|---|---|---|
| 6× **outcome:** bollinger-bounce, ema-crossover-swing, macd-bullish-with-adx, macd-momentum, sma-cross-uptrend-swing, supertrend-momentum-stack | XGBoost (200 trees, binary:logistic, 50 feat) | **At/below random OOS; ema-crossover is inverted (AUC 0.384).** Trained on 100% synthetic backtest labels (no real closed-trade writer exists). Inverted 30/70 train/test split. Regime+VIX passed at serve but never trained. |

### RL
| Model | Type | Status |
|---|---|---|
| **rl_exit** (HOLD/EXIT/TIGHTEN) | tabular Q-learning (`models/rl/q_table.json`) | **No-op live** — train/serve feature skew means no reachable equity state ever returns EXIT/TIGHTEN. Reward never trained early exit (EXIT only on final bar). Shorts scored with long-only sign. No OOS gate; unversioned file. |
| FinRL-X (PPO/A2C/DDPG), intraday_lstm | RL / DL | Correctly **retired** (RL removed from v1). |

### LLM agents — generally healthy (the strong part of the stack)
copilot graph, tradingagents Bull/Bear debate, FinRobot Portfolio Doctor, Strategy Studio NL→DSL,
chart vision, `grounded_reason` (~20 services), sentiment classifier, news enrichment, F&O advisor —
all over a single **OpenRouter adapter** with per-role routing, a budget kill-switch, and a numeric
grounding self-check. Gemini/Anthropic/Together SDKs fully removed. Defects are narrow (see §6).

### Dead code worth knowing about (good machinery, not wired)
López de Prado stack (`frac_diff`, `triple_barrier`, `sample_weights`) · SPA test (`ml/eval/spa.py`) ·
Almgren-Chriss impact cost (`ml/eval/impact_cost.py`) · PSI/KS drift (`ml/eval/drift.py`) ·
v2 30-feature LGBM trainer · microstructure & options-chain feature modules · `breakout_meta_labeler` ·
`tick_exit`, `stagnation_trailing` · DSL **Mood** engine signal (never populated).

---

## 3. CRITICAL issues (28) — grouped by theme, with file refs

### A. Train/serve skew & stale artifacts (the "PROD models aren't real" cluster)
- ✅ **TFT v3 is never served.** Trainer writes `tft_swing_nf.tar.gz` (neuralforecast); serving loads
  `ml/models/tft_model.ckpt` via pytorch-forecasting. → `model_registry.py:182-216`, `generator.py:156-159`,
  `tft_swing.py:336-368`.
- ✅ **TFT "5-bar forecast" reconstructs the past**, not the future (misaligned ~5-6 bars). →
  `model_registry.py:261-334`.
- ✅ **TFT serving dep commented out of `requirements.txt:105-106`** → clean build = whole signal
  pipeline dark, swallowed into one log line (deploy looks healthy).
- ✅ **Qlib serves raw features** to a RobustZScoreNorm-trained booster. → `qlib/engine.py:167-188`.
- ✅ **Qlib OOS gate evaluates on the same raw-feature pipeline** → reported rank-IC is meaningless. →
  `trainers/qlib_alpha158.py:317-324,367`.
- ✅ **LGBM 15-vs-30 feature skew** — promoting v2 → `KeyError` per symbol → zero signals. →
  `generator.py:378-380`, `feature_engineering.py:241-269`, `lgbm_v2.py FEATURE_ORDER`.
- **TFT trainer↔serve format skew**: promoting the trainer's tarball makes `_require()` raise at
  construction → engine won't start (fail-closed but invisible). → `tft_swing.py:217-368`.
- **AutoPilot Qlib normalization skew** — the live ranker that picks every user's buys is fed
  mis-scaled inputs (ZScore-90d vs RobustZScore-2018-2024). → `qlib/engine.py:167-175`.
- **`strategy.segment` attribute doesn't exist** → OPTIONS strategies silently route to the equity
  path live; what's backtested ≠ what executes. → `strategy_runner/runner.py:347-348`.
- ✅ **Regime VIX-missing proxy fabricates a feature** the model never saw. → `regime_detector.py:308-318`.
- ✅ **Served TFT/LGBM/regime on-disk artifacts match no current trainer** (version drift). → `ml/models/`.

### B. Dead / bypassable quality gates (real money has no guard)
- ✅ **Qlib quality gate is dead code** — runner reads `qlib_alpha158_quality_pass`, trainer emits
  `qlib_quality_pass` → promotes on `--promote` alone, even with rank_ic=0/NaN/negative. →
  `runner.py:221`, `qlib_alpha158.py:397`.
- ✅ **Regime HMM auto-promotes with no quality flag / no incumbent compare**, every weeknight at 22:00. →
  `regime_hmm.py:74,192-200`, `scheduler.py:2177-2181`.
- **Strategy gate "OOS" = the in-sample backtest** — the only barrier for LLM-generated (Studio) live
  strategies isn't actually out-of-sample. → `strategy/backtest.py:628-651,554-625`.
- **Strategy gate bypassable with a stale backtest** after a DSL edit (no `last_backtest`
  invalidation) → losing strategy B inherits strategy A's passing gate. → `strategy/registry.py:144-177`.
- **Drift auto-rollback permanently inert** — the headline "catches decay before users lose capital"
  never fires. → `scheduler.py:3142`, `drift_monitor.py:144-145,197-212`.

### C. Outcome models actively harmful
- **Gate 4 gates entries with at/below-random models; ema-crossover inverted (AUC 0.384)** → blocks
  winners, passes losers. → `models/outcome/ema-crossover-swing/metadata.json`, `strategy_runner/runner.py:479-487`.

### D. Data leakage
- **Fundamentals: one as-of-today PIT snapshot broadcast across all history.** Defeats the entire PIT
  machinery; currently masked only by a 2-row `_TEST_` stub cache. → `lgbm_signal_gate.py:269-279,324-334`.
- ✅ **LGBM legacy model trained on raw absolute price levels**, pooled cross-stock (tree splits like
  `close<259.5` are price-tier classifiers, not direction). → `scripts/train_lgbm.py`.
- ✅ **LGBM legacy CV has no walk-forward/embargo/financial metric** (shipped on accuracy of an
  imbalanced 3-class problem). → `scripts/train_lgbm.py:302-322`.
- **v2 WFCV embargo is 12 *positional* rows over a 200-symbol date-pooled array** → purges a fraction
  of one day, not 12 days; massive label overlap leaks into the test fold. → `lgbm_signal_gate.py:90,738-762`.

### E. RL no-op / inverted
- **Live equity RL exit can never reach EXIT/TIGHTEN** (state skew) → trained 6,494-episode Q-table has
  zero behavioral effect. → `risk.py:938-943`, `rl_exit_scaffold.py:139-148`.

### F. Discovery theater
- **Engine signals never load in discovery** → ~40-50% of equity candidates (and 100% of F&O) scored on
  a crippled version of themselves. → `strategy_discovery/runner.py:296-298`.
- **All F&O discovery candidates produce zero trades, tie on score** → F&O search is dead; RNG tie-break
  picks the "best". → `search_space.py:484-497`, `cron.py:47-56`.

---

## 4. HIGH issues (55) — condensed by area

**Qlib/AutoPilot:** booster fed bare positional numpy array, no feature-name persistence ·
`train_qlib_alpha158.py` docstring claims walk-forward but does a single fixed split · HMM fabricates
"bull" as error fallback · parametric VaR cap dormant (only applies when `cov_matrix` passed) ·
nightly rank runs 15:40 IST on yesterday's data · circuit-breaker bars forward-filled into fake flat
returns.

**LGBM/labeling/eval:** entire LdP stack (frac_diff+triple_barrier+sample_weights+v2) is dead ·
`lgbm_signal_gate` can never pass its own gate (**PBO hardcoded 0.5** vs gate ≤0.4) · gate backtests a
5-day strategy but trains 10-day triple-barrier labels (objective≠eval) · cross-sectional embargo too
small, single-date rows split across train/test boundary · **all 4 v1 PROD models bypass the
financial promote gate**.

**ML data:** triple-barrier checks only **close**, never intra-bar High/Low · `production_ohlcv`
returns an **empty frame instead of raising** on yfinance failure (masks total data loss) · FII/DII +
sentiment + fundamentals caches are empty/stub (features ship dead, masked by env flag).

**Registry:** `lgbm_signal_gate` registry row inactive but load-bearing · qlib `direction_agrees` uses
rank not return sign (top half of NSE always "votes BUY") · regime row has `skip_promote_gate=True`
with no quality flag.

**Outcome:** regime+VIX passed at inference but never trained (silently dropped) · VWAP/OBV computed
over different windows train vs serve · trained on synthetic labels only · inverted 30/70 split.

**RL:** reward only trains EXIT on the final bar · no OOS/walk-forward gate · unversioned Q-table, no
registry row · shorts scored with long-only reward sign · options backtest consults equity-trained
Q-table with premium/margin-scaled features.

**Discovery/DSL:** zero multiple-testing correction over thousands of nightly trials · walk-forward off
by default and blends holdout into selection · DB CHECK constraint **rejects** intraday_5m/15m inserts ·
universe gate averages holdout return across symbols (winners mask losers) · discovery output not in
the shape the live gate consumes (promoted strategies carry no OOS block).

**Sentiment/agents:** DSL "Mood" engine permanently dead (never populated) · launch gate requires a
finbert PROD row nothing writes · **sync LLM budget kill-switch never refreshes its meter** → cap
bypassed after restart · sentiment LLM classifier assumes array order and pads by position → silent
mislabeling.

**Vision/micro:** premium-confirmation gate queries `tick_data` by equity symbol but the table is keyed
by Kite `instrument_token` → gate permanently inert.

---

## 5. MEDIUM/LOW (100) — themes
Mostly: NaN/edge handling, magic-number thresholds, missing logging on silent skips, minor metric
miscomputations, untested fallback branches, and documentation/claim drift (docstrings describing
behavior the code doesn't implement). Full list in `/tmp/findings_clean.json` (regenerate from the
workflow transcript if needed). Re-confirm before acting — these are single-pass.

---

## 6. AI agents / LLM layer — verdict
**Architecturally the healthiest subsystem.** Single OpenRouter adapter, per-role model router,
free→free→paid fallback, SSE streaming, vision, a micro-USD budget kill-switch reconciled from
`llm_usage_events`, conversation-memory summarizer, and a regex numeric grounding self-check on
responder output. The do-LLMs-gate-trades invariant holds *structurally*: LLMs only reach live trading
through Studio NL→DSL strategies, which must pass the strategy gate — **but that gate is broken (§3.B)**,
so today an LLM-authored strategy can go live on in-sample numbers. Narrow real bugs: the sync budget
meter doesn't refresh after restart (cap bypass), and the sentiment classifier's positional
array-alignment can silently mislabel news.

---

## 7. Prioritized ML/DL work plan

### NOW — stop shipping into a broken pipeline (correctness & safety; mostly not retraining)
1. **Make serving and training share one contract, per model.** Add a post-train **smoke-load** that
   round-trips each artifact through the *production* predictor before promotion is allowed. Pick ONE
   TFT framework end-to-end (neuralforecast everywhere, or pytorch-forecasting everywhere) and delete
   the other path. Replace the generator's 15-key `split_feature_sets` with `compute_lgbm_v2_features`
   so live features match whatever LGBM is promoted.
2. **Fix the gates.** Correct the Qlib quality-flag key typo; give regime_hmm a real quality flag +
   incumbent comparison; route all 4 PROD models through the financial promote gate; un-hardcode PBO.
3. **Make the strategy gate truly out-of-sample** and invalidate `last_backtest` on any DSL edit. This
   is the only thing standing between LLM-authored strategies and live money.
4. **Fix Qlib serving normalization** — persist the fitted handler/processors (train fit window) and
   apply them at serve + in OOS eval; add a serve-time assert.
5. **Disable or fix Gate 4 outcome models now** — they are net-negative (one inverted). Gate them off
   until retrained on real data with a proper split, or remove from the entry path.
6. **Fail loud, not dark.** `production_ohlcv` and the TFT-import failure must raise/alert, not return
   empty / log-and-continue. Fix the regime "bull" fail-open to fail-closed.
7. **Activate the drift auto-rollback** (or remove the false safety claim).

### NEXT — make retraining trustworthy (then retrain)
8. **Fix labeling/CV leakage** before any retrain: triple-barrier must check intra-bar High/Low; size
   embargoes in trading days (not positional rows) with per-date grouping; stop broadcasting the
   fundamentals snapshot across history (wire real PIT or drop the features).
9. **Wire the dead-but-good machinery**: frac_diff + triple_barrier + sample-uniqueness into the live
   feature/label path; SPA test + Almgren-Chriss impact cost into the gate; PSI/KS drift into the
   monitor.
10. **Retrain the 4 PROD models** on the fixed pipeline with the financial gate enforced, and verify
    each loads through the production predictor (smoke-load from step 1).
11. **RL exit:** fix the reward (teach early exit across the journey, not just final bar), fix the
    short sign, add an OOS gate and a registry row — or shelve it until then.

### LATER — discovery & breadth
12. Load engine signals in discovery; fix the intraday DB CHECK constraint; add multiple-testing
    correction (the SPA test from step 9) over nightly trials; turn walk-forward on and stop blending
    the holdout into selection; fix the F&O search or remove it from the digest.
13. Wire microstructure/options-chain features (or mark honest-empty); fix the premium gate's
    symbol↔instrument_token mismatch.

---

## 8. Gaps / caveats of this audit
- The **verify phase was cut short** by a session limit: 27 of 81 critical/high findings were
  independently re-verified (0 refuted — good precision signal); the rest are single-pass. Re-verify
  any medium/low before acting.
- **Live Supabase `model_versions` was not queried** (no MCP auth in the run). The registry-vs-disk
  reconciliation is inferred from code + on-disk artifacts; confirm against the live table.
- The **synthesis + completeness-critic agents did not run** (same limit). A follow-up pass should
  grep for model-loading code *outside* the 18 audited paths (schedulers, crons, forecast/) to catch
  anything missed.
- Findings dataset: `/tmp/findings_clean.json` (183 deduped) and `/tmp/audit_full.json` (raw + inventories).

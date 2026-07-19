# Meta-Labeling Conviction Score — Design Spec

**Date:** 2026-07-07 · **Status:** DRAFT (founder review pending) · **Owner:** ML program follow-on

## 1. What this is (and is not)

A small classifier **on top of** the deployed style engines that answers one question per surfaced signal: **"given everything known right now, what is the probability this specific signal wins?"** The calibrated probability becomes (a) a user-visible **conviction score** on signal cards and (b) a **position-sizing input** for AutoPilot's Kelly fraction.

This is meta-labeling (López de Prado): the primary models decide *what* to trade; the meta-model decides *how much to trust each decision*. It claims **no new alpha** — it improves the precision and sizing of alpha already validated by the walk-forward program. That is deliberately the opposite of the positional experiments: we are not hunting a new signal, we are learning the error structure of proven ones.

**Not in scope:** new trade generation, any change to the engines' rankings, positional (shelved), marketing claims of improved returns before the gate passes.

## 2. Scope

- **Engines covered (v1):** momentum_lambdarank v2 + swing_lambdarank v2 (their walk-forward OOS fold predictions exist today — 12 folds / 756 test dates / ~300 symbols each ≈ 200k+ training rows per engine).
- **One model per engine** (horizons and error structures differ: H=20 vs H=10). Shared feature builder, shared trainer code, two artifacts.
- **Alpha ensemble signals:** v2 candidate, after the style-engine version proves out (its serving path and voter structure differ).

## 3. Label

`y = 1` if the signal's **net forward return beats costs** over the engine's horizon: `fwd_return_H − 2×30bps > 0`, else 0. Raw (absolute) win, not excess-vs-benchmark:

- Conviction shown to a user must mean "this trade makes money", not "beats an index they don't hold".
- Regime-dependence is then a *feature*: the model learning "bear regime ⇒ lower P(win)" is honest conviction, and the regime inputs let it express exactly that. (The beta-vs-alpha decomposition already lives in the regime-gated evaluation; it does not belong in this label.)
- Cost constant revisited after the paper window's fill-cost validation.

**Critical protocol rule:** training rows come **only from OOS fold predictions** of the primary models (the walk-forward harness's per-fold test frames). Never score in-sample primary predictions — an in-sample primary is optimistically wrong in ways the live model never is, and the meta-model would learn to trust phantom precision.

## 4. Features (all point-in-time at signal date, ~20 max)

| Group | Features |
|---|---|
| Signal context | engine score percentile (cross-sectional), score dispersion that date, gap to next-ranked name |
| Regime | ensemble state (bull/sideways/bear), HMM filtered probabilities, days since last switch |
| Market | NIFTY realized vol 21d, breadth (pct above 200-SMA), universe mean 21d return |
| Name | realized vol 63d, amihud illiquidity 63d, beta_index_252, distance from 52w high |
| Forecast agreement | ens_fwd_ret sign agreement count, tsfm-vs-kronos spread (momentum) / chronos_uncert (swing) |

No raw prices, no fundamentals, nothing that isn't already computed by the existing feature builders or `ml/regime/`. Feature count kept small on purpose: the training set is ~10² smaller-dimensional than the engines' and must not overfit fold idiosyncrasies.

## 5. Model & training

- **LightGBM binary classifier** (small: ≤200 trees, depth-limited), then **isotonic calibration** on held-out folds — the sizing math consumes probabilities, so calibration is not optional.
- **Purged walk-forward CV** through the existing 9-stage spine (`task="classification"`, `check_class_balance=True`), embargo = engine horizon, folds aligned to the primary engine's folds so no meta-fold tests on a period its primary fold trained on.
- Retrain cadence: monthly, or on drift-monitor breach of the primary engines. CPU-only, minutes, $0.
- HPO: ≤20 trials over OOS AUC; `n_trials` feeds the DSR null as usual.

## 6. Pre-registered gates (per engine; fail ⇒ conviction does not ship for that engine)

1. **OOS AUC ≥ 0.55** across folds (mean), no fold < 0.50.
2. **Top-tercile precision lift:** signals in the top conviction tercile must have a win rate ≥ base rate + 5pp OOS.
3. **Calibration:** Brier score ≤ climatology baseline (constant base-rate predictor); reliability slope in [0.8, 1.2].
4. **Usefulness floor:** conviction ordering must be non-degenerate — top-vs-bottom tercile win-rate spread ≥ 8pp (else it's a constant and the UI chip is noise).

Per the no-fallbacks rule: if gates fail there is **no rules-based stand-in** — the feature simply doesn't ship, signals render exactly as today.

## 7. Serving & product surface

- Scored inside the existing 15:55 IST style-signals cron; conviction persisted with the daily snapshot (registry-first model load, same degraded-mode semantics: if the meta-model or a feature input is unavailable, signals ship **without** the conviction field — never a fabricated score).
- API: one added field per signal `conviction: {score: 0-1, band: "high"|"medium"|"low", pct: int}` (bands = calibrated terciles).
- UI: a conviction chip on signal cards + `/signals/[id]` detail. Public copy names it **conviction** — no model/technique names in UI (house rule). Duotone tokens only.
- **Tier placement (recommendation, founder to confirm):** Free sees signals without conviction; **Pro sees the band; Elite sees band + percentile**. Fits the existing ladder without touching locked pricing.

## 8. Sizing integration (phase 2 of this feature — gated separately)

Kelly fraction scaled by calibrated P(win) (`f = f_kelly × (2p − 1)/(2p_base − 1)`, floored at 0, 5% per-name cap **unchanged**). Enters AutoPilot only after (a) the conviction gate passes AND (b) the engine paper window validates backtest expectations. Display ships first; sizing follows evidence.

## 9. Evidence plan

- Backtest: re-run the walk-forward harness with conviction-weighted books vs equal-weight — report iid deltas (this is a *diagnostic*, not a marketing number, since weights derive from the same OOS folds).
- Paper window: log conviction alongside every live signal from day one, so 2–4 weeks of live calibration data exists before any sizing decision.

## 10. Open decisions for founder

1. Tier placement (§7 recommendation).
2. Public copy for the chip ("Conviction" vs "AI Confidence" — recommend "Conviction": honest, no accuracy implication).
3. Whether the sizing phase (§8) waits for one or two paper cycles.

## 11. Effort & cost

Feature builder + trainer + gates ≈ 2–3 days (spine reuse is near-total); serving + UI chip ≈ 1 day. $0 compute (CPU). No new data dependencies, no licensing surface.

---

## 12. OUTCOME (2026-07-07) — GATES FAILED, FEATURE SHELVED

Both experiments run same-day, pre-registered gates unchanged:

- **E1 (raw net-win label, §3 as written):** momentum AUC 0.575 / lift +2.2pp (fail), swing AUC 0.537 with two folds < 0.5 (fail). Diagnosis: ~80% of feature importance on date-level market features — the model was a weak market-timer, not a signal grader.
- **E2 (excess-win label — beat the equal-weight cross-section; cross-fitted calibrated Brier gate):** momentum AUC 0.508 / lift +0.6pp, swing AUC 0.504 / lift +0.4pp. Coin-flips.

Conclusion: the engines' OOS errors carry no residual structure predictable from
their own context — evidence the primary rankers are properly fit. Conviction
ships nothing (no-fallbacks). Revisit ONLY with genuinely new per-name inputs
(news sentiment, order flow, fundamentals). Infrastructure retained: fold-pred
dump, classification spine branch, meta feature builder, trainer + calibration.

# Quant X — AI Systems: A Deep, Honest Audit

*Internal engineering audit for the founder. Plain English, not marketing. Where
something is **claimed but not real**, this document says so. Generated 2026-06-05
by a four-way parallel code audit, grounded in the live codebase + the Supabase
`model_versions` table.*

---

## How to read this

People conflate three different layers. Keep them separate:

1. **The ML models** — trained statistical / deep-learning artifacts that produce
   the actual *trade signals* (numbers in → a signal out). These live in
   `ml/models/` + Supabase `model_versions`.
2. **The LLM agents** — open-source *language* models that explain, narrate,
   classify, and (behind a gate) help build strategies. They are **not** the
   signal models and — except behind the backtest gate — never place or gate a trade.
3. **The features** — what the user actually clicks. Each is backed by some mix of
   the above, or by plain deterministic rules.

This doc covers each layer, then how they connect + the money/tier model, then a
consolidated **honesty scorecard**.

---

## Executive summary (the 60-second version)

**The signal brain (ML).** Live swing signals are produced by **five real trained
models** fused into one ensemble: a **Qlib Alpha158** cross-sectional ranker, a
**Temporal Fusion Transformer** price/level forecaster, a **LightGBM** BUY/HOLD/SELL
gate, a **3-state Gaussian HMM** market-regime detector, and a **sentiment voter**.
These are genuine artifacts on disk with real *no-fallback discipline* (if any model
is missing the signal pipeline refuses to run rather than fake it). That engineering
is real and rare. **But** their *measured* edge is small — Qlib's own metadata flags
`realmoney_pass: false` (IC ≈ 0.03), the LGBM gate's cross-validated Sharpe is ≈ 0.1
(≈ no edge), and the HMM's "confidence" is structurally ≈ 1.0. These are
**display/ranking-grade signals, not proven alpha.**

**The talking layer (LLM).** A custom **GraphRunner** runtime (not LangChain) runs
**three agent graphs** — Copilot chat (4 nodes), Portfolio Doctor (5), Bull/Bear
"Counterpoint" debate (7) — plus single-shot roles (strategy NL→DSL, scanner thesis,
chart vision). **All ~10 roles run on FREE OpenRouter models** under a hard **$20/mo
kill-switch**. Agents are advisory only — the **one** path to a live trade is the
Strategy Studio, and only after an **enforced out-of-sample backtest gate**.

**The product (features).** ~15 features across **Free / Pro ₹999 / Elite ₹1,999**.
The crown jewels: real ML swing signals + the enforced backtest gate. The honest
weak spots: a few marketing engine names map to **no code**, some features are
rule-based or **not actually runnable** (earnings predictor has no model on disk;
"momentum picks" has no model), and the on-page "embedded agents" were mostly
templated text (now fixed to hand off to real chat).

**Bottom line:** a genuinely well-engineered ML/LLM *platform* with real discipline
(no-fallback models, enforced gate, budget kill-switch, removed RL) wrapped around an
*edge that is not yet proven live* and a few overstated labels. The engineering is
the moat; the proof-of-profit is the gap.

---

## Table of contents
1. [The AI/ML Models — the trained "brains"](#the-aiml-models--the-trained-brains)
2. [The LLM Agents — the reasoning / conversation layer](#the-llm-agents)
3. [The Features — what the user actually gets](#the-features)
4. [How it all connects + tiers + money](#how-it-all-connects)
5. [Honesty scorecard — real vs claimed](#honesty-scorecard)
6. [Appendix — every caveat, by domain](#appendix--every-caveat-by-domain)

---

## The AI/ML Models — the trained "brains"

This section documents every trained model in the Quant X stack: what it really is, how it works, what it does, what it cannot do, and the honest caveats. Every claim is grounded in the actual code and the live `model_versions` table.

### TL;DR for a founder

- There are **four "PROD" models** behind the live swing-signal engine: a **Qlib Alpha158 ranker**, a **Temporal Fusion Transformer (TFT) price forecaster**, an **HMM market-regime detector**, and a **"FinBERT-India" sentiment voter**. A fifth required input — the **LightGBM signal gate** — is loaded too but is *not* actually flagged PROD in the registry.
- The signal engine is **strict about "no fallbacks"**: if any required model artifact is missing, `SignalGenerator.__init__` **raises and the live signal pipeline stays dark** (the rest of the app keeps serving). That discipline is real and enforced (`backend/ai/signals/generator.py:123-188`, `backend/api/app.py:401-427`).
- **The single biggest honest caveat:** the "FinBERT-India" sentiment "model" is, **by default at runtime, not FinBERT at all** — it is an **LLM zero-shot classifier** routed through OpenRouter. The real FinBERT neural net only runs if you set `USE_FINBERT_FALLBACK=1` (`backend/ai/sentiment/engine.py:52-75`, `backend/ai/sentiment/llm_classifier.py`).
- **Second-biggest caveat:** the **on-disk model artifacts diverge from what the registry says is PROD.** On disk the LGBM is the **legacy 15-feature model** (not the 30-feature v2 the DB metrics describe) and the TFT checkpoint is a **small pytorch-forecasting net (hidden=32)**, while the registry's PROD `tft_swing v3` is a **different, larger neuralforecast model (hidden=128)**. Which one runs depends on whether B2 registry credentials are configured.
- **Honest performance reality:** the models' own out-of-sample metrics are **weak-to-marginal**. Qlib's own metadata literally says `qlib_realmoney_pass: false` ("not safe for autonomous trading"). The LGBM gate's cross-validated Sharpe is ~**0.097** (essentially zero). These are display/ranking-grade signals, not proven alpha.

---

### Model inventory (verified against the live registry)

PROD = `is_prod=true` in `public.model_versions`. "On-disk" = the artifact actually sitting in `ml/models/` that loads when B2 isn't wired.

| Internal name | Public brand | Type | Predicts / outputs | Registry status | Consumed by | Ensemble weight |
|---|---|---|---|---|---|---|
| `qlib_alpha158` | **Alpha** | Microsoft Qlib **Alpha158** features + LightGBM (LambdaRank) | Cross-sectional rank of every NSE name | **PROD v8** | Swing signals, strategies, AutoPilot | 0.20 |
| `tft_swing` | *(no public brand)* | **Temporal Fusion Transformer** (PyTorch-Lightning) | 5-day p10/p50/p90 price quantiles → direction + entry/SL/target levels | **PROD v3** (registry) — but on-disk artifact differs | Swing signals (level derivation) | 0.30 |
| `lgbm_signal_gate` | *(part of signal gate)* | **LightGBM** 3-class classifier | BUY / HOLD / SELL probabilities | **NOT marked PROD** (v1 exists); loaded via disk fallback | Swing signals | 0.30 |
| `regime_hmm` | **Regime** | 3-state **Gaussian HMM** (`hmmlearn`) | bull / sideways / bear + confidence | **PROD v24** | Swing signals (bear gate + sizing), AutoPilot, dashboards | 0.10 |
| `finbert_india` | **Mood** | **Nominally** FinBERT (`Vansh180/FinBERT-India-v1`); **actually an LLM zero-shot classifier by default** | Per-headline positive/neutral/negative → score in [-1,1] | **PROD v1** ("loaded_at_runtime") | Swing signals, Doctor, Digest | 0.10 |
| `breakout_meta_labeler` | *(none — internal)* | **RandomForest** (500 trees, depth 3, 8 features) | P(breakout follows through), 0–1 | On disk only | **Scanner Lab UI only — NOT in alpha path** | n/a |
| — (AutoPilot) | **AutoPilot** | **Not a trained model** — orchestration: Qlib ranker + HMM sizing + Kelly + VIX overlay | Position sizes / rebalance | n/a (executor) | Elite portfolio automation | n/a |
| `intraday_lstm`, `momentum_timesfm`, `momentum_chronos`, `vix_tft`, `chronos2_macro`, `finrl_x_*`, `earnings_xgb` | — | Various (Bi-LSTM, TimesFM, Chronos, RL) | — | **All retired or never-prod** | Nothing live | n/a |

---

### 1. Qlib Alpha158 ranker — public name **"Alpha"**

**What it is.** The one genuinely "industrial-grade" piece: real Microsoft **Qlib 0.9.7** running the **Alpha158** feature handler (158 engineered price/volume factors) feeding a **LightGBM** gradient-boosted ranker (`qlib.contrib.model.gbdt.LGBModel`, trained with a LambdaRank/IC objective). Code: `backend/ai/qlib/engine.py`. PROD = `qlib_alpha158 v8`.

**How it works.** Once a day the engine asks Qlib's `Alpha158` handler to compute all 158 factors for the whole NSE universe over a rolling 90-day window, takes the latest bar per stock, runs the booster to get a raw score, then **sorts every stock by score and assigns a cross-sectional rank** (`rank_universe`, engine.py:145-204). Rank 1 = best expected forward return; that rank is normalised to a 0–1 voter score in the signal loop (`generator.py:329-334`).

**What it does / what it's for.** It is the cross-sectional "who's best right now" engine — it feeds Signals, the strategy library, and AutoPilot's stock selection.

**What it CANNOT do.** It does not predict price levels, timing, or magnitude — only *relative* ordering. It needs a populated Qlib provider directory; if that's missing it returns `[]` and the whole batch is refused.

**Honest caveats.** The model's **own metadata flags it as not real-money-grade**: `rank_ic_mean ≈ 0.031`, `rank_icir ≈ 0.36`, and a literal field `qlib_realmoney_pass: false` with the reason *"signal not consistent enough for real-money use. Safe for shadow/display; not safe for autonomous trading."* An IC of 0.03 is a real-but-tiny edge typical of quant factor models — it is **not** a money printer, and the team's own gate says so. Also note Qlib is initialised with `region=REG_CN` (China calendar config) even though it's NSE data (engine.py:84) — a known Qlib-on-India quirk, not necessarily a bug, but worth knowing.

### 2. Temporal Fusion Transformer forecaster — `tft_swing` (no public brand)

**What it is.** A **Temporal Fusion Transformer**, an attention-based deep-learning time-series model, used to forecast 5 trading days ahead with uncertainty bands (10th / 50th / 90th percentile). Loader: `TFTPredictor` in `backend/ai/model_registry.py:182-339`.

**How it works.** For each stock it takes the last ~120 daily bars of OHLCV + indicators, runs the TFT, and gets quantile forecasts for the next 5 days. The signal engine then **derives the actual trade levels** from these: stop-loss = the wider of TFT's pessimistic p10 and a 2×ATR floor; target = the more conservative of TFT's optimistic p90 and a 4×ATR cap (`generator.py:534-594`). Direction (bullish/bearish/neutral) comes from whether the median forecast is above/below current price.

**What it does / what it's for.** It is the **entry/stop/target brain** — the only model that produces the concrete price levels a user trades, and a 0.30-weight directional voter.

**What it CANNOT do.** It is single-stock and needs ≥125 clean bars or it returns `None` and the symbol is skipped. The direction thresholds are crude (±0.5%) and the level math is heuristic ATR clamping, not learned risk.

**Honest caveats — artifact mismatch.** This is the clearest "claimed vs real" gap. The **registry PROD record (`tft_swing v3`)** describes a `neuralforecast` model, `hidden_size=128`, `max_encoder_length=60`, 100 symbols, directional accuracy 0.68. But the **artifact on disk** (`ml/models/tft_model.ckpt` + `tft_config.json`) is a **different, smaller `pytorch_forecasting` model**: hidden_size=**32**, heads=2, encoder=**120**, trained on ~93 symbols (verified by reading the checkpoint hyperparameters and the booster embedding shape). The loader (`TFTPredictor`) uses `pytorch_forecasting.load_from_checkpoint`, which is **only compatible with the on-disk small model, not the neuralforecast PROD one.** So in any environment without B2 wired, the live TFT is the small 32-hidden model — not the one the registry metrics (0.68 accuracy) refer to. Treat "68% directional accuracy" as **unverified for the model that actually runs locally.**

### 3. LightGBM signal gate — `lgbm_signal_gate`

**What it is.** A **LightGBM** 3-class classifier (BUY/HOLD/SELL). Loader: `LGBMGate` in `backend/ai/model_registry.py:37-173`. It's a 0.30-weight voter and arguably the most heavily-weighted single classifier.

**How it works.** Per stock it builds a feature row, the model outputs class probabilities, softmaxed into buy/hold/sell, and the BUY probability becomes the voter score. The loader has a genuinely good safety property: it **refuses to load** if the feature count in its sidecar metadata doesn't match the booster (`generator`-level guard, model_registry.py:80-91) — closing an earlier silent-zero-fill bug.

**Honest caveats — legacy model is what runs, and it isn't PROD.**
- In the **live signal loop**, the gate is fed by `split_feature_sets()`, which returns the **15 legacy features** (HOLD=0/BUY=1/SELL=2) — `feature_engineering.py:241-269`. The **on-disk booster is exactly that legacy 15-feature model** (verified: `num_feature()=15`, 3 classes, 2400 trees, generic `Column_0…14` names).
- A **v2 path exists** (`compute_lgbm_v2_features`, 30 features, triple-barrier labels) but is **not wired into the live `SignalGenerator` loop.** So the better model the DB metrics describe is dormant.
- In `public.model_versions`, **`lgbm_signal_gate` is NOT marked `is_prod`** — yet `SignalGenerator` *requires* it. It only works because the disk-fallback resolver finds the file regardless of the PROD flag. That's a registry-vs-runtime inconsistency worth tightening.
- Performance: the v1 metrics show **cross-validated Sharpe ≈ 0.097** and **accuracy ≈ 0.40** across 5 folds with wildly varying per-fold results (Sharpe ranged −7.9 to +5.6). This is **statistically indistinguishable from no edge.**

### 4. HMM market-regime detector — public name **"Regime"**

**What it is.** A 3-state **Gaussian Hidden Markov Model** (`hmmlearn`) that labels the market bull / sideways / bear from Nifty returns, realized vol, and India VIX. Code: `ml/regime_detector.py`. PROD = `regime_hmm v24`.

**How it works.** Five features (5d/20d returns, 10d realized vol, VIX level, VIX 5d change) are scaled and fed to the trained HMM; the Viterbi-decoded last state is the current regime, and states are sorted so 0=bull/2=bear by mean return. When bear is active, every signal's confidence is multiplied by 0.6 and AutoPilot sizes down (`generator.py:78-79, 456-457`).

**What it does / what it's for.** A market-wide risk dial — it's a low-weight (0.10) voter but a high-impact *gate*: bear regime throttles the whole book.

**Honest caveats — degenerate confidence.** The reported "confidence" is **near-meaningless as a probability.** Inspecting the actual PROD pickle, the HMM's transition matrix has **very high self-transition probabilities** (diagonal ≈ 0.987 / 0.931 / 0.966) and a startprob of `[0,0,1]`. With such sticky states, the posterior for the decoded state is **driven to ~1.0 almost every day**, so the "confidence" number shown to users is structurally close to 100% and doesn't reflect genuine uncertainty. Separately, the class has a `_default_regime()` that returns "bull, confidence 0.0" — but the **SignalGenerator deliberately bypasses it** and *raises* if regime can't be computed (generator.py:295-309), consistent with the no-fallback rule. The model's own quality metric is just average log-likelihood (−5.1/obs); there is **no validated "regime call accuracy"** — calling regimes correctly is not measured.

### 5. "FinBERT-India" sentiment — public name **"Mood"** *(the most important caveat in this section)*

**What it CLAIMS to be.** A fine-tuned **FinBERT** transformer for Indian financial news (`Vansh180/FinBERT-India-v1`), 3-label positive/neutral/negative. Real loader exists: `backend/ai/sentiment/finbert_india.py`. PROD = `finbert_india v1`.

**What actually runs at runtime.** By default it is **NOT the FinBERT neural network.** The `SentimentEngine` selector (`engine.py:52-75`) picks the **`LLMFinanceClassifier`** first — a **prompt sent to an open LLM via the OpenRouter gateway** that asks the model to label headlines and return JSON (`llm_classifier.py`). The real FinBERT weights only load if you explicitly set **`USE_FINBERT_FALLBACK=1`**, or if the LLM key is missing. So the "Mood / FinBERT-India" voter that users see is, in production, **an LLM doing zero-shot classification at inference time**, with a synthetic probability distribution constructed from the LLM's self-reported confidence (llm_classifier.py:216-237). The DB even tags it `"loaded_at_runtime": true`.

**Why they did this (their stated reasoning).** The code comments cite the FinDPO paper (FinBERT Sharpe collapses at 5bps costs) and that `Vansh180/FinBERT-India-v1` is a 7,451-sample hobby model with self-reported F1=76.4% and no external benchmark. So they demoted FinBERT to a fallback. That's a defensible call — but it means **the public name "FinBERT-India" overstates what's running.** "Mood" is really "an LLM reads the news headlines."

**What it CANNOT do.** It only sees headlines that the news fetcher returns; with no recent news a stock gets a **neutral 0.5** (the model's own empty-input output, not a fabricated stand-in). Weight is only 0.10, so it's a tie-breaker, not a driver. It does not read article bodies, filings, or price-reaction.

### 6. Breakout meta-labeler — `breakout_meta_labeler.pkl` *(Scanner Lab only — not alpha)*

**What it is.** A **RandomForest** classifier (verified: 500 trees, max_depth 3, 8 features: `resist_s, tl_err, max_dist, vol, adx, quality, height_atr, rr_ratio`, binary classes) that scores whether a detected chart-pattern breakout will follow through.

**Honest caveats.** It is **explicitly removed from the signal/alpha path** — `SignalGenerator` sets `self._ml_labeler = None` (generator.py:135-138) and never calls it. It only runs in the **Scanner Lab UI.** Per project memory, the old "63.6% win-rate" claim for the underlying pattern engine is **retired** — real out-of-sample performance is ~35–40% with costs. Do not present this as a trading model.

### 7. AutoPilot — public name **"AutoPilot"** *(not a model — an orchestrator)*

**What it is.** Despite the "AI engine" branding, AutoPilot is **not a trained model.** It's an execution policy that takes the Qlib ranker's top names, sizes them with **Kelly-criterion weights** (decayed), applies **HMM-regime sizing** and a **VIX overlay**, caps each stock at 5%, and rebalances daily at 15:45 IST. Hard stops/targets remain authoritative. (Project memory `project_autopilot_supervised_2026_05_24`; brand copy in `frontend/lib/engines.ts`.)

**Caveat.** Its "edge" is entirely inherited from the Qlib ranker (IC ≈ 0.03, not real-money-grade per Qlib's own gate) plus position-sizing math. It is a disciplined wrapper, not an independent predictive brain.

### 8. Retired / never-promoted models (named but not in production)

The registry is littered with models that **never reached PROD or were retired** — important so nobody assumes they're live:
- **`finrl_x_*` (PPO/A2C/DDPG)** — reinforcement-learning traders, **all retired** (RL failed its Sharpe gate; project memory `project_rl_removed_2026_05_23`).
- **`momentum_timesfm`, `momentum_chronos`, `chronos2_macro`** — foundation-model forecasters (TimesFM/Chronos), **zero-shot "pointer + calibration only," retired or never-prod.** These are wrappers around pretrained models, not models trained on Indian data.
- **`vix_tft`** — VIX forecaster, retired.
- **`intraday_lstm`** (Bi-LSTM, 5 versions) — exists but **none marked PROD**; intraday ships rule-based.
- **`earnings_xgb`** — **skipped** (Supabase had <50 labeled rows to train on).

---

### The "no-fallbacks" discipline (this part is genuinely solid)

The one engineering claim that fully holds up: the system **refuses to fake a model.** `SignalGenerator.__init__` calls `_require(...)` for each artifact and **raises `RuntimeError` if any is missing** (generator.py:123-188); the app catches it and **leaves the signal pipeline disabled rather than degrading to heuristics** (app.py:401-427). At inference time it raises rather than defaulting when NIFTY/VIX history or Qlib ranks are unavailable (generator.py:295-325). The ensemble math has no weight-renormalisation precisely because every voter is guaranteed present (ensemble.py:17-36). This is a real, auditable guarantee — it just doesn't make the underlying models any stronger than their (modest) metrics.

### Public-brand vs real-model map (for marketing-honesty checks)

| Public brand (engines.ts) | Real model | Honest? |
|---|---|---|
| **Alpha** — "cross-sectional multi-factor model" | Qlib Alpha158 + LightGBM | Accurate (deliberately architecture-agnostic) |
| **Mood** — "domain-tuned sentiment model" | **LLM zero-shot classifier** (FinBERT only as fallback) | **Overstated** — "domain-tuned" implies the fine-tuned FinBERT, which usually isn't what runs |
| **Regime** — "probabilistic regime model" | 3-state Gaussian HMM | Accurate |
| **AutoPilot** — "supervised ranker + Kelly sizing" | Orchestrator over Qlib ranker | Accurate (it's honest that it's a ranker + sizing) |
| TFT forecaster | pytorch-forecasting TFT (on disk) / neuralforecast (registry) | No public brand; internal artifact mismatch |

---

<a name="the-llm-agents"></a>

## The LLM Agents — The Conversational / Reasoning Layer

This section documents every LLM-backed agent and role in the codebase: what each one is, the open-source model it runs on, what it can and cannot do, its tier/caps, and the honest gaps. **Bottom line up front:** there is a real, custom multi-agent runtime (`GraphRunner`) wired to ~10 distinct LLM roles, all running on **free open-weight models** through OpenRouter with a **$20/month hard kill-switch**. But several of the on-page "agents" the user sees are **not LLMs at all** — they are data fetches with a typewriter animation. And the LLM agents never directly place trades. Both points are detailed below.

### How the runtime works (GraphRunner — not LangChain/LangGraph)

The team ships a small custom orchestration runtime instead of LangChain. The pieces:

- **`Agent`** (`backend/ai/agents/base.py:48`) — one unit of work. It reads shared state, optionally calls an LLM + tools, and writes its result into `state.scratch[name]`. Every run is timed and recorded as an `AgentTurn` for the trace/"Context" tab.
- **`GraphRunner`** (`base.py:101`) — executes an ordered list of agents over one shared `AgentState`. A bare agent = a serial step; a *list* of agents = a parallel fan-out (`asyncio.gather`). An agent can raise `EarlyExit` to short-circuit the whole graph (used by the Copilot classifier to reject off-topic chat).
- **`AgentState`** (`state.py:38`) — the typed dataclass threaded through a run: `inputs` (read-only), `scratch` (per-agent working memory), `turns`, `tool_trace`, and `output` (final payload).
- **`LLM`** (`llm.py:165`) — the single access surface to OpenRouter (OpenAI-compatible HTTP via `httpx`). Supports `complete`, `complete_stream` (real SSE), `generate_json` (JSON mode), and `complete_vision` (image + text).
- **`tool_registry`** (`tools.py`) — a decorator-registered set of async data-fetch functions the Copilot planner can call.

The runtime is deliberately narrow: it supports exactly three composition patterns — sequential chain, parallel fan-out + aggregate, and linear tool-use. This is a reasonable, honest engineering choice for a solo-founder codebase (`llm.py` docstring is candid about why they didn't take the LangGraph dependency).

### The model routing + cost-control spine (this part is real and well-built)

Every agent role maps to a model via the `AGENT_MODEL_MAP` env JSON (`llm.py:103`, `llm_for(role)` at `llm.py:115`). The live map in `.env:144` assigns all 10 roles to **`:free` OpenRouter slugs**:

| Role | Model (from `.env`) | Free/Paid |
|---|---|---|
| classifier | `meta-llama/llama-3.3-70b-instruct:free` | FREE |
| tool_planner | `openai/gpt-oss-120b:free` | FREE |
| responder | `meta-llama/llama-3.3-70b-instruct:free` | FREE |
| doctor | `openai/gpt-oss-120b:free` | FREE |
| debate | `openai/gpt-oss-120b:free` | FREE |
| strategy_generator | `qwen/qwen3-coder:free` | FREE |
| fno | `openai/gpt-oss-120b:free` | FREE |
| scanner_thesis | `meta-llama/llama-3.3-70b-instruct:free` | FREE |
| vision | `google/gemma-4-31b-it:free` | FREE |
| sentiment | `meta-llama/llama-3.3-70b-instruct:free` | FREE |

**Honest note vs. the planning docs:** the locked "open-LLM strategy" memo specified a tiered brain map (Qwen3-8B classifier, Qwen3-32B planner, Llama-70B responder, **Qwen3-235B for the strategy generator** "where money is made", DeepSeek for debate). The *shipped* map is **flatter and cheaper** — everything is either Llama-3.3-70B-free or gpt-oss-120b-free, and the strategy generator runs on `qwen3-coder:free`, **not** the 235B model. So the aspiration of "spend the budget on the big generator" is not realized in the current config; it all runs free.

**Free → free → paid fallback** (`build_models`, `llm.py:81`): a rate-limited free slug transparently rolls to another free model, then to a cheap paid model only as a last resort. OpenRouter caps the candidate list at 3.

**The $20 kill-switch** (`observability/llm_budget.py`): an in-process `UsageMeter` tracks month-to-date spend (reconciled from the `llm_usage_events` table on a 60s TTL). `is_paid()` (`llm_pricing.py:82`) returns False for `:free` slugs (they're priced $0), so they never move the meter and are never blocked. When spend crosses `LLM_MONTHLY_BUDGET_USD` ($20), `_guard_budget` (`llm.py:190`) blocks *paid* calls and the free→paid spill is disabled — free models keep running. This is a genuine, working hard cap. Caveat (the code says so itself): it is single-instance accurate; on a multi-instance deploy each process can briefly overshoot by up to one TTL window.

**Gemini is fully removed** — a grep finds only a documentation URL comment in `llm_pricing.py`; no `genai`/Gemini runtime path remains, matching the 2026-06-04 memory.

### Count: ~10 distinct LLM roles across 3 graphs + 5 standalone callers

There are **10 named roles** in the model map. They are consumed by **3 multi-agent graphs** (Copilot = 3 LLM nodes, Doctor = 5, Debate = 7) plus **5 single-shot LLM call-sites** (strategy generator, scanner thesis, chart vision, F&O advisor, sentiment classifier). Counting individual agent *nodes* across the graphs, there are **15 distinct agent classes** plus the 5 standalone callers.

---

### Agent 1 — Copilot / "Main Chat" (the only true conversational agent)

- **Surface:** the `/copilot` page (Main Chat) and the `route='markets'` "Mood" embedded card. Backend: `POST /api/ai/copilot/chat` and `/copilot/chat/stream` (`api/ai_routes.py:213`, `:267`).
- **Graph** (`agents/copilot.py:421`): **Classifier → ToolPlanner → ToolCaller → Responder** (4 nodes, 3 of them LLM; ToolCaller is pure code).
- **Models:** classifier + responder = Llama-3.3-70B-free; tool_planner = gpt-oss-120b-free.
- **How it works:** The **Classifier** rejects off-topic prompts (with a regex fast-path that skips the LLM for >95% of obviously-finance messages — `copilot.py:40`). The **ToolPlanner** asks the LLM which of the registered tools to call (max 3). The **ToolCaller** runs them against `tool_registry` (real Supabase + market-data reads: `get_portfolio`, `get_watchlist`, `get_signal`, `get_todays_signals`, `get_stock_snapshot`, `get_current_regime`, `suggest_options_strategy`). The **Responder** writes the reply, "numbers-first senior-analyst voice", and on the stream path also emits **real GenUI artifacts** (price sparkline, regime bars, stat cards) built only from live tool data (`build_artifacts`, `copilot.py:289`).
- **Can do:** answer market/portfolio/signal questions grounded in the user's real data; stream tokens live; render charts from tool output; cite tools.
- **Cannot do:** place/modify trades, generate signals, or browse arbitrary websites. It only has the 7 registered tools. The classifier refuses non-finance topics.
- **Tier + caps:** Free 5/day · Pro 50 · Elite 200 messages. **This is the only cap actually enforced at runtime** (`_enforce_copilot_cap` → `AssistantCreditLimiter`, `ai_routes.py:88`). Admins bypass; the limiter fails *open* on DB errors (relies on the $20 switch as backstop).

### Agent 2 — Portfolio Doctor / AI SIP (FinRobot, 5-agent chain-of-thought)

- **Surface:** F7 Portfolio Doctor + F5 AI SIP. Backend: `POST /api/ai/finrobot/analyze` (`ai_routes.py`, gated `RequireTier(PRO)`).
- **Graph** (`agents/finrobot.py:269`): **4 specialists in parallel → 1 synthesizer**: FundamentalAgent, ManagementAgent (tone), PromoterAgent (holding/pledge), PeerAgent → SynthesizerAgent.
- **Model:** all 5 on `doctor` = gpt-oss-120b-free.
- **How it works:** each specialist grades one facet (returns JSON: grade/score/flags). The synthesizer blends them into a narrative + an `Action: add|hold|trim|exit` line, plus a weighted `composite_score` computed **in code** (`finrobot.py:241`), not by the LLM.
- **Can do:** produce a structured, multi-angle stock report from supplied fundamentals.
- **Cannot do / caveats:** the **ManagementAgent largely runs on empty inputs** — concall transcripts are stubbed and headlines are often absent (`finrobot.py:89` returns a neutral placeholder), so "management tone" is frequently a no-op. Grades come from the LLM reading yfinance/screener.in numbers — it is an LLM judgment, not a quant model. **No per-run cap is enforced on the backend** (route comment at `ai_routes.py:464` admits unlimited-rerun gating is "enforced client-side… backend refusal deferred"). The intended `portfolio_doctor` monthly cap (Free 1 / Pro 10 / Elite 60) exists in config but is not consumed.

### Agent 3 — Bull/Bear Debate (TradingAgents, 7-agent graph) → public "Counterpoint"

- **Surface:** signal detail page → "Debate" tab. Backend: `POST /api/ai/debate/signal/{id}` (Elite only, `ai_routes.py:540`).
- **Graph** (`agents/tradingagents.py:307`): **3 analysts in parallel (Fundamentals/Technical/Sentiment) → Manager → Bull + Bear researchers in parallel → RiskManager → Trader.** 7 LLM nodes.
- **Model:** all on `debate` = gpt-oss-120b-free.
- **How it works:** analysts produce stance JSON; manager distills a neutral briefing; bull/bear steelman each side; risk manager proposes a `size_multiplier` (0–1); the Trader emits a final `decision: enter|skip|half_size|wait` + confidence. Full transcript persists to `signal_debates`.
- **Can do:** generate a readable, multi-perspective argument for/against an existing signal, with a verdict.
- **Cannot do / caveats:** **the Trader's "enter" verdict is purely advisory and explanatory — it does NOT execute or gate any trade.** It runs *on* an already-generated signal. The whole debate is 7 sequential/parallel LLM calls on a free model, so latency and free-tier rate-limits are real risks. Tier gate is enforced (Elite via `RequireFeature("debate")`), but the **per-day debate cap (Elite 10/day) is not consumed at runtime.**

### Agent 4 — Strategy Studio generator (NL → DSL)

- **Surface:** Strategies/Studio. Module: `ai/strategy/studio.py`.
- **Model:** `strategy_generator` = `qwen/qwen3-coder:free`.
- **How it works:** a single-shot text→JSON call (JSON mode) that compiles a plain-English strategy into the validated **Strategy DSL** (`compile_strategy`, `studio.py:266`). The Pydantic schema is the safety boundary; invalid output is retried once with the error appended, else 422. It is **force-coerced to `mode="backtest"`** (`studio.py:315`) — Studio can never emit a live strategy.
- **Can / cannot:** can turn "RSI mean reversion on Nifty 50" into a runnable DSL doc. Cannot push to live, cannot invent indicators/engines (closed registry; engines limited to Alpha/Mood/Regime). **This is the agent that, per memory, sits behind the backtest+Sharpe `evaluate_gate` before anything can transition to live** — so the much-discussed "LLM can now generate trades" only holds *after* the out-of-sample gate passes. Caveat: it runs on a free coder model, not the planned 235B; complex prompts may need the 2 retries or fail. The `strategy_gen` cap (Free 1/Pro 10/Elite 30) is defined but not enforced at runtime.

### Agent 5 — Chart Vision (B2)

- **Surface:** stock + signal pages ("read this chart"). Module: `ai/vision/analyzer.py`.
- **Model:** `vision` = `google/gemma-4-31b-it:free` (multimodal).
- **How it works:** renders an 800×500 PNG server-side, base64-encodes it, sends it + a strict-JSON prompt to the vision endpoint (`complete_vision`). Returns trend/pattern/support/resistance/volume/setup/confidence/narrative. Never raises — returns `available=False` on any failure.
- **Caveats:** this is **genuinely multimodal** (image → vision LLM), the one remaining image use. Its outputs (e.g. "support at ₹X") are the *model's read of a picture*, not computed levels — treat as illustrative. The prompt explicitly forbids leaking real model names (TFT/FinBERT/etc.). Tier: Pro `finagent_vision` / Elite anywhere; the `chart_vision` cap (Pro 20/Elite 60) is defined but not enforced at runtime. Minor: the `gemma-4-31b-it:free` slug isn't in the price card, so it's treated as unpriced/$0 — fine since it's free anyway.

### Agent 6 — F&O AI advisor

- **Surface:** F&O page "ai-suggest". Backend: `POST /api/fo-strategies/ai-suggest` (Elite, `fo_strategies_routes.py:704`). Also reachable as the Copilot tool `suggest_options_strategy`.
- **Model:** `fno` = gpt-oss-120b-free.
- **How it works:** builds a prompt with spot/regime/VIX direction (+ optional real portfolio net-delta context for "hedge my book") and constrains the LLM to pick **one template from a fixed registry** (bull call spread, bear put spread, iron condor, straddle, strangle, iron butterfly). If the LLM returns an unknown template, code falls back to a regime-based default (`:867`).
- **Can / cannot:** suggests + sizes an options structure that then plugs into the rule-based pricer. **Explicitly advisory — the route docstring states it never auto-deploys; the user must click Deploy.** Tier gate enforced (Elite); the `fno_advisor` cap (Elite 20/day) is defined but not enforced.

### Agent 7 — Scanner / Screener "AI thesis"

- **Surface:** scanner + screener deep-dive panels. Modules: `services/screener_v2/enrich.py:280`, `services/chart_patterns/explain.py`.
- **Model:** `scanner_thesis` = Llama-3.3-70B-free.
- **How it works:** writes a 2-3 sentence *descriptive* narration of what a chart is showing, strictly **no buy/sell language** (locked rule). It's the "optional cherry on top": if the LLM is unavailable, a **deterministic, code-written thesis** is used instead (`_deterministic_thesis`, `enrich.py:326`). So this surface degrades to templated text gracefully.
- **Caveats:** because of the deterministic fallback, the "AI thesis" the user sees may often be the **non-LLM template**, especially if the free model rate-limits. The `scanner_thesis` cap (Pro 30/Elite 100) is defined but not enforced.

### Agent 8 — Sentiment classifier (the "FinBERT" replacement)

- **Surface:** feeds news sentiment into scanner enrich, digest, earnings features, and the debate sentiment analyst. Module: `ai/sentiment/llm_classifier.py`.
- **Model:** `sentiment` = Llama-3.3-70B-free, via the sync `complete_sync` path.
- **Honest naming point:** the public/branding layer references **FinBERT-India**, but the *primary* sentiment classifier in code is **an LLM prompt at runtime**, not the FinBERT model. The file documents this decision (FinBERT demoted per the 2025 FinDPO/FinSentLLM research; `llm_classifier.py:1`). FinBERT remains importable only as a shadow/fallback behind `USE_FINBERT_FALLBACK`. So "FinBERT sentiment" in any user-facing copy is, in the live path, **a free LLM doing zero-shot headline classification**, batched 20 at a time, rate-limited to 10 req/min, with neutral fallback on failure.

---

### The biggest honest caveat: most on-page "agents" are NOT live LLMs

This deserves emphasis because the UI strongly implies otherwise. `frontend/components/copilot/EmbeddedAgent.tsx` is a streaming-choreography **shell** (typing dots → tool-trace chip → token-by-token reveal → staggered artifacts). It has two modes: STATIC (timer-driven fake reveal) and LIVE (`run` prop hits an endpoint). Critically, on the real pages the `run` functions **fetch structured data and then build the "narration" string in TypeScript from the numbers** — the typewriter effect makes it *look* like an LLM is talking, but no LLM is called:

| Embedded card | Page | What `run` actually calls | LLM? |
|---|---|---|---|
| **Screener Agent** | /scanner | `api.screener.powerConfluence` (data); narration templated in TS | **No** |
| **Strategy Agent** | /strategies | `api.strategies.getCatalogSections` (data); narration templated in TS | **No** |
| **F&O Advisor** | /fno | FII/DII + VIX + OI data; narration templated in TS | **No** |
| **Analysis Agent** | /stock/[symbol] | `api.dossier.get` (engine tally, no LLM tokens — comment says so at `page.tsx:518`); narration templated in TS | **No** |
| **Mood** | /markets | `api.ai.copilotChat({route:'markets'})` → real Copilot graph | **Yes** |

So **4 of the 5** embedded "agents" are deterministic data renders dressed as chat; only the **Mood** card on /markets is a real LLM call. The real LLM otherwise only engages when the user clicks a prompt pill or hits Send — which doesn't answer inline, it **hands the question off to Main Chat** (`/copilot?q=...`, `EmbeddedAgent.tsx:58`). This is a defensible design (keeps cost down, keeps numbers exact), but anyone evaluating "how many AI agents talk to the user" should know four of the most visible ones are animated templates, not models.

### Cross-cutting caveats (the founder should know these)

1. **Agents never directly trade or generate signals.** Every decision-flavored output (debate "enter", doctor "add", F&O suggestion, strategy DSL) is advisory. The only path where an LLM-authored artifact reaches live execution is the Strategy Studio DSL — and only after the out-of-sample backtest/Sharpe **gate** passes. This matches the locked policy.
2. **The per-feature LLM caps are mostly decorative.** `LLM_FEATURE_CAPS` in `core/tiers.py:129` defines daily/monthly limits for chat, strategy_gen, scanner_thesis, chart_vision, debate, fno_advisor, portfolio_doctor — but a repo-wide grep shows `llm_feature_cap()` is **never called at runtime**. Only the **copilot "chat" cap is actually enforced**, and via a *separate* `COPILOT_DAILY_CAPS` table (`tier_gate.py:182`) that duplicates the values. The other features are protected only by their tier gate (who can access) and the global $20 kill-switch — not by per-user/day quotas. A single Elite user could call debate/doctor/vision far more than the documented caps.
3. **Everything runs on free models.** Quality is whatever Llama-3.3-70B-free / gpt-oss-120b-free / Gemma-4-free / Qwen3-coder-free deliver, subject to OpenRouter free-tier rate limits. The planned investment in a large 235B generator is not in the live config.
4. **Sentiment "FinBERT" and several "models" are LLM prompts at runtime** (sentiment classifier especially). The public engine brands (Alpha/Mood/Regime/Counterpoint) are descriptive labels; the conversational layer behind them is open LLMs + tool calls, not bespoke trained models.
5. **Graceful degradation is real and pervasive** — when `OPENROUTER_API_KEY` is unset or budget is hit, agents return empty/neutral/canned output rather than crashing. Good for uptime, but means a user can receive a deterministic template while believing they got an AI answer.

---

<a name="the-features"></a>

## The Features — What the User Actually Gets (Honest Inventory)

This section catalogs every user-facing feature, what it concretely does, who can use it (Free / Pro / Elite), the page it lives on, and **what actually powers it under the hood** — separating real trained ML models from rule-based logic and from LLM-prompt-at-runtime "AI agents."

### The three buckets you need to keep straight

Before the table, internalize the single most important distinction in this product:

1. **Real trained ML models** — files on disk, trained offline, run as math at inference. There are exactly **4 in production**: a LightGBM signal gate, a Temporal Fusion Transformer (TFT) swing forecaster, a Qlib/Alpha158 cross-sectional ranker, and an HMM regime detector. Verified on disk in `ml/models/` (`lgbm_signal_gate.txt` 15 MB, `tft_model.ckpt`, `regime_hmm.pkl`) plus the Qlib LightGBM and FinBERT-India sentiment model. These back **F2 swing signals, F8 regime, F4 AutoPilot, F5 SIP**.
2. **Rule-based engines** — deterministic formulas, no learning. This is **F6 F&O** (Black-Scholes + a regime×VIX lookup table), the **77 scanners**, the **Portfolio Doctor risk flags**, and most of the **daily digest**.
3. **LLM-prompt-at-runtime "agents"** — these are NOT models you trained. They are prompts sent to an open-weight LLM through OpenRouter every time the user clicks. This is **Main Chat, Portfolio Doctor's "4 agents," the 7-agent Counterpoint debate, the F&O advisor, Strategy Studio's NL→DSL compile, and Chart Vision.** Source: `backend/ai/agents/llm.py`.

> **Critical honesty note on the LLMs:** by default *every* agent above hits **one free model — Llama-3.3-70B** (`LLM_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"`, `config.py:253`). The per-agent premium routing from your project notes (Qwen3-235B "generator," DeepSeek debate, Qwen3-32B planner) lives in an **empty env var** `AGENT_MODEL_MAP` (`config.py:258`) and is **not wired unless you set that JSON in production.** So today, the "235B strategy generator where the money is made" is, in the default deploy, the same free 70B model as the chat.

---

### Master feature table

| Feature | Tier | Route / Page | What it does (concretely) | Backed by (honest) |
|---|---|---|---|---|
| **F1 Intraday signals** | Pro | `/signals` (intraday tab), `GET /api/signals/intraday` | Returns intraday (`signal_type='intraday'`) BUY signals from the last N-minute window | Same 5-model ensemble as F2, but **the intraday training/model modules were deleted** (`backend/ai/intraday/training/*` removed in this branch) — depends on intraday signals being produced by the pipeline; reads from `signals` table. Non-Pro get an empty list. |
| **F2 Swing signals** | Pro (Free: 1/day) | `/signals`, `GET /api/signals/today` | The flagship. Ranks NSE names, emits BUY signals with entry/SL/targets for 3–10 day holds. Requires ≥3 of 5 models to agree + weighted confidence ≥ threshold | **REAL ML.** 5-voter ensemble: LGBM gate (0.30) + TFT forecast (0.30) + Qlib Alpha158 rank (0.20) + FinBERT-India sentiment (0.10) + HMM regime (0.10). `ai/signals/voters.py`, `generator.py`. **This is the genuine product.** |
| **F3 Momentum picks** | Pro | (no dedicated page/route) | Marketed as "weekly top-10 momentum, auto-rotation" | **Not a real distinct feature.** No `momentum_weekly` endpoint, no momentum model, no momentum `signal_type` in the generator. It's a marketing label served via the Alpha ranker / screener. |
| **F4 AutoPilot (auto-trader)** | Elite | `/autopilot`, `/api/auto-trader/*` | Daily 15:45 IST rebalance: ranks stocks → Kelly-tilted weights → regime multiplier → VIX overlay → 5%/stock, 80% gross caps → diffs vs live positions → places broker orders | **REAL, supervised-only.** `trading/autopilot_service.py` uses Qlib ranker + HMM + VIX + Kelly. Orders go through a real broker bridge (`services/live_executor.py` → `TradeExecutionService.place_order`). RL was removed. Hard stops + −10% drawdown breaker. |
| **F5 AI SIP** | Elite | `/portfolio/sip`, `/api/ai-portfolio/*` | Monthly-rebalanced quality portfolio (6–20 names, 7%/asset cap), last Sunday of month | **REAL.** Qlib quality screen → Black-Litterman optimizer (`ai/portfolio/engine.py`, `black_litterman.py`). |
| **F6 F&O strategies** | Elite | `/fno` (hub) + `/fo-strategies` (deep) | Picks a weekly options strategy (Iron Condor / Straddle / spreads) for index underliers; prices every leg, shows max P/L, breakevens, PoP; payoff calculator; OI tracker | **Mostly RULE-BASED** (it says so: `ai/fo/strategies.py:16` "the rule layer that ships today… no RL yet"). Black-Scholes + a regime×VIX→strategy lookup. Optional `/ai-suggest` "hedge my book" advisor is an LLM (never auto-deploys). |
| **F7 Portfolio Doctor** | Free (1/mo) / Pro (10/mo) / Elite (60/mo) | `/portfolio/doctor`, `/api/portfolio/doctor` | "4-agent" per-holding review: fundamentals, management, promoter, peers → per-position scores + composite + risk flags + action (rebalance/hold/reduce) | **Hybrid.** The "4 agents" are **4 LLM prompts** (`ai/agents/finrobot.py`) over screener.in fundamentals, PLUS **deterministic rule flags** (concentration, sector, regime). |
| **F8 Market regime** | Free | `/regime`, `/markets` | Classifies the market bull / sideways / bear daily; every other engine sizes down in bear | **REAL ML.** 3-state HMM on Nifty returns + India VIX + breadth (`ml/regime_detector.py`, `regime_hmm.pkl`). |
| **F9 Earnings predictor** | Pro (basic) / Elite (+strategy) | `/api/earnings/*` | Earnings calendar + "surprise probability" (beat prob) per upcoming result | **PARTIALLY BUILT / NOT RUNNING.** Calendar works. The prediction is an XGBoost classifier that **does not exist on disk** (only `scripts/train_earnings_scout.py`). `predict_surprise()` raises `ModelNotReadyError` → **503** (`ai/earnings/predictor.py:42-100`). |
| **F11 Paper trading + league** | Free | `/paper-trading`, `/api/paper/*` | Virtual ₹10L account, place/track paper orders, equity curve, anonymized weekly top-20 leaderboard, achievements | **REAL & rule-based.** `INITIAL_CASH = ₹10,00,000` (`paper_routes.py:35`), live league query. Conversion funnel — stays free. |
| **F12 Daily digest** | Free (Telegram) / Pro (WhatsApp) | settings + cron | Pre-market brief (7:30 IST) + evening summary (17:30 IST): regime, index levels, your positions/triggers, on Telegram/WhatsApp/email | **Mostly TEMPLATED.** Deterministic header + per-user block; optional LLM prose intro that **silently falls back to a template** if the LLM is down (`ai/digest/generator.py`). |
| **Scanner / Screener** | Pro (`scanner_lab`) | `/scanner`, `/api/screener/*` | 77 technical/fundamental scanners (breakouts, RSI, volume, OI buildup, etc.), confluence scoring, per-scanner results | **RULE-BASED + a meta-labeler.** Live menu = **77 scanners** (`data/screener/engine.py`); confluence engine weights **24** categories. Optional AI "thesis" per result is an LLM (Pro 30/day, Elite 100/day). "50+ scanners" claim is honest. |
| **Strategy Studio** | Free 1 / Pro 10 / Elite 30 gen/day | `/strategies`, `/api/strategies/studio/compile` | Type a strategy in English → get a validated DSL strategy → backtest → must pass a gate before going live | **LLM (NL→DSL) + REAL enforced gate.** The compile is an LLM. The **valuable** part is real: `evaluate_gate` + `run_walk_forward` (out-of-sample Sharpe, drawdown, holdout, regime coverage) **blocks live deploy with a 422** (`strategies_routes.py:648-701`). Genuinely built. |
| **Main Chat (Copilot)** | Free 5 / Pro 50 / Elite 200 msgs/day | `/copilot` | Context-aware chat: classifier → tool-planner → tool-caller → responder; calls 7 real tools (portfolio, watchlist, signals, regime, stock snapshot, options-strategy) with live SSE streaming + chart artifacts | **LLM agent.** `ai/agents/copilot.py` + `tools.py`. Real tool data, real model — but the "model" is the free Llama-70B by default. |
| **Chart Vision** | Pro (signal stocks) / Elite (any) | stock + signal pages, `/api/ai/vision/*` | Upload/render a chart image → LLM vision describes pattern/levels | **LLM vision agent** (Pro 20/day, Elite 60/day). Feature key is `finagent_vision*`; users see "chart vision." |
| **Counterpoint (Bull/Bear debate)** | Elite (10/day) | signal detail, `POST /api/ai/debate/signal/{id}` | A 7-agent Bull-vs-Bear debate on one signal, ending in a verdict | **LLM agent pipeline** (`ai/agents/tradingagents.py`). Explanation/narration only — never gates or generates the underlying signal. |
| **Alerts Studio** | Pro | settings/inbox, `/api/alerts/*` | Per-event × per-channel routing matrix (push / Telegram / WhatsApp / email), test fires | **RULE-BASED** routing layer (`alerts_routes.py`). No model. |
| **Marketplace** | Free browse / Pro deploy / Elite publish | `/strategies` (catalog), `/api/marketplace/*` | Browse algo strategies + backtests, deploy to your account (deployment cap by tier) | **REAL browse + deploy**, tier-gated. **Publish/creator side is deferred** — no creator-apply or revenue flow in the matrix-promised `marketplace_publish`. |
| **AI Stock Dossier** | Free basic / Pro full | `/stock/[symbol]`, `/api/dossier/*` | Aggregates the latest signal, regime, sentiment, Alpha rank, forecast quantiles, upcoming earnings into one stock page | **Aggregator** over real model outputs in DB (`dossier_routes.py`). Free sees a subset; Pro sees the full model-output grid. |
| **Weekly Review** | Pro | inbox, `/api/weekly-review/*` | Sunday 08:00 IST personalized review of your week's trades/signals | LLM prose over your data (`weekly_review/generator.py`). |
| **Referrals / Paper League / Kill-switch / Onboarding quiz** | Free | various | Standard growth + safety surfaces | Rule-based. Kill-switch is a real Free-tier safety control. |

---

### The "engine" brand layer — what the names really mean

The public UI (`frontend/lib/engines.ts`) shows exactly **four engines**: **Alpha · Mood · Regime · AutoPilot.** This is a deliberate **brand firewall** — real architectures (TFT, Qlib, FinBERT, HMM) never ship to users. That's a reasonable product choice, but be clear internally:

- **Alpha** = Qlib Alpha158 LightGBM ranker (real).
- **Mood** = FinBERT-India sentiment model (real).
- **Regime** = HMM regime detector (real).
- **AutoPilot** = **not a model.** It's a rebalancing *executor* that consumes the other three. Calling it an "engine" alongside three actual models is a slight category error.

Separately, the **landing-page `FeatureGrid.tsx` invents engine names that map to no code**: "VolCast," "AllocIQ," "InsightAI," "EarningsScout," "Trajectory," "Forecast quantile forecast." These are marketing flourishes — `EarningsScout` in particular points at a model that **doesn't exist on disk**. If asked "what is VolCast," the honest answer is "a marketing name for the rule-based VIX logic feeding F&O."

### What's genuinely strong vs. what's thin

**Genuinely strong / real:**
- F2 swing signals (true 5-model ensemble, no-fallback discipline enforced).
- F8 regime (real HMM, used everywhere for sizing).
- F4 AutoPilot + F5 SIP (real optimizers wired to a real broker order path with hard risk caps).
- Strategy Studio's **backtest gate** (out-of-sample, walk-forward, blocks live deploy) — this is the single best-engineered safety feature.
- The 77-scanner screener and paper-trading league are real and complete.

**Thin / overstated / not running:**
- **F3 momentum** — a label, no feature behind it.
- **F9 earnings prediction** — code exists, model doesn't; returns 503.
- **F1 intraday** — depends on a pipeline whose training modules were just deleted in this branch.
- **All "AI agents"** — they're free-LLM prompts, not your IP; the premium per-agent routing you designed isn't active by default.
- **Marketplace publish** — promised in the tier matrix, not built (creator side deferred).
- **Stale docstring in `generator.py`** still advertises "6 backtested strategies" and win-rate numbers the code itself disclaims (those strategies are Scanner-Lab-only, ~35-40% out-of-sample).

---

<a name="how-it-all-connects"></a>

## How It All Connects, Tiers, and Money

This section traces a single thread from "where the numbers come from" all the way to "how the company charges for it." It is written for a smart founder who does not do ML. Every claim is tied to a real file and line. Where something is weaker than it looks, it says so in plain English.

### 1. The end-to-end pipeline (data → models → signal → feature → trade)

Here is the whole machine in one diagram. Read it top to bottom.

```
                         ┌────────────────────────────────────────┐
                         │  DATA PROVIDER  (one source, swappable) │
                         │  DATA_PROVIDER env: "free" | "kite"     │
                         └────────────────────────────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            │ "free"  → YFinanceProvider  (yfinance, no key, rate-limited)│  ← DEFAULT
            │ "kite"  → KiteDataProvider  (admin Zerodha; jugaad fallback)│
            └─────────────────────────────┬─────────────────────────────┘
                                          │  OHLCV bars + quotes + option chains
                                          ▼
                    ┌───────────────────────────────────────┐
                    │  MarketDataProvider (data/market.py)   │  one factory, used everywhere
                    └───────────────────────────────────────┘
                                          │
         ┌────────────────────────────────┼───────────────────────────────────┐
         ▼                                ▼                                     ▼
 ┌───────────────┐            ┌──────────────────────────┐          ┌────────────────────┐
 │ FEATURES      │            │  5 MODEL "VOTERS"         │          │ SCANNER / SCREENER │
 │ indicators,   │──────────▶ │  (all real, all loaded)  │          │ (rule filters,     │
 │ ATR, etc.     │            │  LGBM .30  TFT .30        │          │  Scanner Lab only) │
 └───────────────┘            │  Qlib .20  HMM .10        │          └────────────────────┘
                              │  FinBERT .10              │
                              └────────────┬─────────────┘
                                           │  weighted ensemble
                                           ▼
                  ┌──────────────────────────────────────────────┐
                  │  SIGNAL GATE  (ai/signals/generator.py)       │
                  │  emit ONLY if: ≥3/5 voters agree BUY          │
                  │   AND confidence ≥ 40  AND risk:reward ≥ 1.5  │
                  └──────────────────────┬───────────────────────┘
                                         │  rows → `signals` table (Supabase)
                                         ▼
            ┌──────────────────────────────────────────────────────────┐
            │  /api/signals/today  → FREE capped to 1/day, Pro/Elite ∞  │
            └──────────────────────────────────────────────────────────┘

 SEPARATE TRACK (Elite only, real money):
   Qlib ranks universe ──▶ Kelly-decayed weights ──▶ HMM regime ×, VIX overlay,
   5%/stock + 80% gross caps ──▶ diff vs current positions ──▶ AutoPilot emits trades

 STRATEGY TRACK (user/LLM-built strategies):
   DSL strategy ──▶ walk-forward backtest ──▶ STRATEGY GATE (out-of-sample) ──▶ live
```

#### 1a. The data layer

There is exactly **one switch** that decides where all market data comes from: `DATA_PROVIDER` (`backend/core/config.py:77`). It defaults to `"free"`. `MarketDataProvider._get_kite_provider` (`backend/data/market.py:117-125`) reads that switch: `"kite"` loads the Zerodha admin provider, **anything else loads yfinance**.

| Provider | File | What it is | Honest caveat |
|---|---|---|---|
| **yfinance** (default) | `data/providers/yfinance.py` | Free Yahoo Finance scraper, no API key. 5-min history cache, 30-sec quote cache. | This is **Tier-3, lowest-quality** data per the project's own memory. It is rate-limited; the code has a backoff circuit-breaker (`_rl_record_failure`, line 162) precisely because Yahoo throttles. Quotes are delayed/approximate, not a real-time feed. **This is what runs unless someone sets `DATA_PROVIDER=kite`.** |
| **Kite (Zerodha)** | `data/providers/kite.py` | Admin's paid Zerodha account = app-wide data. Token-bucket limiter (180/min). Falls back to free `jugaad-data` NSE bhavcopy when the token is stale. | Requires a live admin token that **expires at 6 AM IST daily** (`is_token_valid`, line 259). There is an auto-refresh that logs in headlessly with stored password + TOTP (`auto_refresh_kite_token`, line 947) — if that ever breaks, data silently falls back to jugaad or returns nothing. |

**Founder takeaway:** the premium Kite data path exists and is well-built, but the system ships pointed at free yfinance by default. The quality of every downstream signal depends on which one is actually configured in production.

#### 1b. The models and the ensemble

The signal engine (`ai/signals/generator.py`) is **model-first and "no fallback."** All five models load at construction; if any artifact is missing it **raises and refuses to start** (lines 60-63, 144-188) rather than faking a score. The artifacts genuinely exist on disk (`ml/models/`: `lgbm_signal_gate.txt` 15 MB, `tft_model.ckpt` 1.5 MB, `regime_hmm.pkl`, plus the Qlib engine and FinBERT). These are **real trained models, not LLM prompts** — an important distinction the rest of this audit flags elsewhere.

| Voter | Weight | Role |
|---|---|---|
| LGBM gate | 0.30 | Buy/sell/hold probability gate |
| TFT | 0.30 | Price forecast → also sets entry/stop/target levels |
| Qlib (Alpha158) | 0.20 | Cross-sectional relative-strength rank |
| HMM regime | 0.10 | Bull/sideways/bear classifier (also size multiplier) |
| FinBERT-India | 0.10 | News sentiment; **0.5 neutral when no news** (line 432) |

A signal is emitted **only** when all three of these clear (`generator.py:444-475`):
1. **≥ 3 of 5 voters agree** on BUY (`min_agreement`, line 447)
2. weighted **confidence ≥ 40** (line 458; halved further in a bear regime, line 457)
3. TFT-derived **risk:reward ≥ 1.5** (line 474)

It then writes a row to the `signals` table. Note the system **only ever emits LONG equity signals** here (`direction="LONG"`, line 494) — there is no short side in the swing engine. `is_premium` is just a flag set when confidence ≥ 75 (line 508).

### 2. The Strategy Gate — the thing that "protects the money"

This is the most important safety mechanism in the codebase, and it is real and enforced. It exists for user-built and LLM-built DSL strategies (separate from the daily signal feed above).

**The problem it solves:** a rule strategy has no fitted weights, so the way you overfit it is by *selection* — generate 100 candidates (a person or an LLM can), keep the one whose full-history backtest looks best. In-sample Sharpe is exactly the number that's easiest to cherry-pick (`evaluation.py:9-16`).

**The defence:** the gate deliberately **ignores in-sample numbers** and scores only the **out-of-sample (walk-forward) block** (`evaluate_gate`, `ai/strategy/evaluation.py:59-175`). `run_walk_forward` (`ai/strategy/backtest.py:629`) runs the backtest, then chops it into multiple time windows plus a most-recent **holdout** the strategy was never selected against.

A strategy can reach **live only if it clears every bar** (defaults in `GateThresholds`, lines 40-46, all env-tunable):

| Threshold | Default | Meaning |
|---|---|---|
| `min_oos_sharpe` | 0.5 | Out-of-sample risk-adjusted return floor |
| `min_trades` | 20 | Enough OOS trades to be statistically meaningful |
| `max_drawdown_pct` | 35% | Worst-window loss ceiling |
| `min_consistency` | 0.5 | Half the time-windows must be profitable |
| `require_holdout_positive` | True | The most-recent unseen window must not lose money |
| `min_symbol_breadth` | 0.5 | Universe strategies: must work on ≥half the symbols (kills "great on RELIANCE only") |
| `min_regime_coverage` | 0.8 | Regime strategies validated on a real regime, not the default |

**Where it's enforced:** `transition_strategy` (`api/strategies_routes.py:648-710`). To go `→ live` you hit **two barriers**: a tier gate (Pro/Elite, returns 403, line 669) and the quality gate. If the OOS backtest fails, the API returns **HTTP 422 `gate_failed`** with the exact list of breached thresholds (line 698-710). It's controlled by `STRATEGY_GATE_ENABLED` (default True, `config.py:217`). Options strategies have no OOS path and so are **blocked from live by design** (paper only).

**Honest caveats:** (1) The gate only proves a strategy held up on *past* data across windows — it is a strong anti-overfitting filter, **not** a guarantee of future profit. (2) The bars are tunable via env, so a deployment could weaken them. (3) This gate guards *strategies*; the **daily signal feed in §1 does not pass through this gate** — its quality rests on the models themselves.

### 3. AutoPilot — turning rankings into real trades (Elite only)

`AutoPilotService` (`backend/trading/autopilot_service.py`) is the daily auto-trader, run by the **15:50 IST** scheduler job (`platform/scheduler.py:224-229`). It is a clean supervised pipeline — **RL was removed**, and the docstring (lines 22-24) is candid that the previous RL version was a silent no-op. The chain:

1. Qlib ranks the NSE universe **once** for the whole batch (`rank_universe`, line 95).
2. Top-10 names → **Kelly-decayed weights** (0.85 geometric decay, so rank-1 gets the most capital), normalized and scaled to an 80% gross cap (`_compute_base_weights`, lines 142-180).
3. **Regime multiplier** applied: bull ×1.0, sideways ×0.7, bear ×0.3 (`REGIME_SIZING`, line 52).
4. **VIX overlay** + a hard **5%-per-stock cap** (lines 178, 219-235).
5. Diff target weights against current live positions → emit the **minimum set of orders** (`_emit_trades`, line 361).

**Safety rails that are genuinely there:** per-user eligibility is re-checked before every rebalance (`check_live_trade_eligibility`, `trading/eligibility.py`) — it requires Elite tier, a connected broker, no global kill-switch, and no per-user pause. There's a `dry_run` mode that records the decision but emits nothing (line 241), and a per-stream toggle so a user can disable swing auto-trading.

**Honest caveats:** (1) Each AutoPilot trade ships with a **rule-based, brand-safe "thesis"** (`_build_brand_safe_thesis`, line 494) — it is a deterministic template string ("combines our trend, momentum, sentiment… engines"), **not** an LLM explanation and not per-trade reasoning; the code itself says LLM enrichment comes "LATER." (2) AutoPilot has the same data-quality dependency as everything else (yfinance by default). (3) It only goes long.

### 4. The money — 3 tiers, what each gets, and how billing flips

Prices and caps are not hardcoded in app logic; they live in the `subscription_plans` DB table (seeded by `migrations/2026_05_28_pru_subscription_plans_seed.sql`, in **paise**), and feature access is governed by `core/tiers.py`.

| | **Free** | **Pro** | **Elite** |
|---|---|---|---|
| **Price/month** | ₹0 | **₹999** (`99900` paise) | **₹1,999** (`199900` paise) |
| Annual (20% off) | — | ₹9,599 | ₹19,199 |
| Swing signals/day | **1** (`FREE_DAILY_SIGNAL_CAP`, `signals_routes.py:39`) | Unlimited | Unlimited |
| Intraday / Momentum | — | ✓ | ✓ |
| Watchlist symbols | **5** (`FREE_WATCHLIST_CAP`, `watchlist_routes.py:31`) | Unlimited | Unlimited |
| Scanner Lab | — | ✓ | ✓ |
| Build & deploy own strategy live | — | ✓ (still must pass the gate) | ✓ |
| **AutoPilot auto-trading (F4)** | — | — | **✓** |
| AI SIP (F5), F&O (F6) | — | — | ✓ |
| Bull/Bear Debate (B1) | — | — | ✓ |
| Portfolio Doctor | ₹199 one-off | included | unlimited |
| Copilot chat | 5/day | 50/day* | 200/day* |
| Chart-vision (B2) | — | 20/day | 60/day |
| Strategy generation (NL→DSL) | 1/day | 10/day | 30/day |

\* `LLM_FEATURE_CAPS` in `core/tiers.py:129-137` says chat is **5 / 50 / 200**; the marketing/pricing matrix says "**150**/day" for Pro (`FeatureComparisonMatrix.tsx:63`). **These two numbers disagree** — the enforced backend cap is 50, not 150.

**How a payment flips a tier:** Razorpay webhook → `process_successful_payment` (`api/payment_routes.py:133`) verifies the HMAC signature (`verify_webhook_signature`, line 70), resolves plan → tier, writes `user_profiles.tier`, and emits a `tier_change` event that **invalidates the 60-second tier cache** (`resolve_user_tier` in `tiers.py:181`, cache TTL line 178) so new features unlock within a minute. Refund/cancellation downgrades to free (line 263).

**The LLM cost kill-switch (protects the burn, not the trades):** all paid LLM calls go through a month-to-date meter (`observability/llm_budget.py`). Once spend hits `LLM_MONTHLY_BUDGET_USD` (default **$20**, `config.py:249`), `enforce()` raises `BudgetExceededError` and paid model calls stop until next month (`ai/agents/llm.py:130-206`). Free-tier models cost $0 and are never blocked. The per-tier LLM caps above are abuse-ceilings layered under that hard $20 cap.

### 5. The biggest honest caveats (read these)

- **Default data is the weakest data.** `DATA_PROVIDER` defaults to `"free"` yfinance (Tier-3 per the project's own memory). Premium Kite data is built but only active if explicitly configured — and the Kite token needs a daily 6 AM refresh that depends on a stored password + TOTP login.
- **SEBI registration is not in place.** `SEBI_RA_REG_NUMBER` defaults to `"PENDING_APPROVAL"` (`config.py:94`). The product distributes buy signals and runs an auto-trader; the Research Analyst number that's supposed to appear on every signal disclaimer is a placeholder until approval lands.
- **No proven live alpha.** The strategy gate proves *historical, out-of-sample robustness*, not future profit. The daily signal feed doesn't even go through that gate — its edge rests entirely on the trained models. There is no live track record substantiating profitability in this code.
- **The signal feed is long-only equity.** The swing engine only emits LONG NSE-equity signals; no shorts.
- **The README is materially stale.** `README.md` still advertises "11 trained ML models," "FinRL-X PPO + DDPG + A2C" for AutoPilot (RL was removed), several deprecated descriptive engine brand names, and "Copilot (Gemini)" (Gemini was fully removed; it's now OpenRouter). The **real** stack is 4–5 supervised models + a rule-based AutoPilot. Treat the README as marketing, not spec.
- **One documented cap mismatch:** Pro chat is enforced at 50/day in code but shown as 150/day in the pricing UI.
- **AutoPilot's per-trade "thesis" is a template, not analysis** — deterministic filler text, by design, until LLM enrichment is added.

---

<a name="honesty-scorecard"></a>
## Honesty scorecard — real vs claimed

A blunt sort of everything above into four buckets.

### ✅ Genuinely strong (real, verified, rare)
- **Five real trained models on disk**, fused into one signal — not prompt wrappers.
- **No-fallback discipline is enforced**: `SignalGenerator` raises if any model is
  missing; the pipeline goes dark rather than degrading to heuristics.
- **The Strategy Studio backtest GATE is real and enforced** — out-of-sample only,
  walk-forward + holdout, returns `422 gate_failed` before anything goes live. A real
  enforced gate behind an LLM strategy generator is genuinely uncommon.
- **The $20/mo LLM kill-switch is real** — free models priced $0 bypass it, paid
  calls are blocked over budget (single-instance accurate).
- **Custom GraphRunner agent runtime is real** (3 graphs), not a LangChain demo.
- **Gemini is 100% removed** — the conversational layer is all OpenRouter open models.
- **AutoPilot is supervised-only** (RL removed) with real broker order path + 5%/80%
  position caps + a −10% drawdown breaker.

### ⚠️ Overstated — brand vs reality
- **"FinBERT-India" / "Mood" is an LLM at runtime by default**, not FinBERT (FinBERT
  only runs with `USE_FINBERT_FALLBACK=1`). The "domain-tuned sentiment model"
  branding overstates it.
- **Public engine names are a 4-name brand layer** (Alpha/Mood/Regime/AutoPilot) over
  the real architectures — and **AutoPilot isn't a model**, it's a rebalancer.
- **Marketing names with no code behind them**: VolCast, AllocIQ, InsightAI,
  EarningsScout, Trajectory correspond to no real modules.
- **Measured edge is weak** and the code's own metadata says so (Qlib
  `realmoney_pass: false`, gate Sharpe ≈ 0.1, HMM confidence ≈ 1.0). Display-grade.
- **README is badly stale** ("11 ML models", FinRL-X RL, Gemini Copilot — all wrong).

### 🔧 Broken / unbuilt / unenforced
- **F9 Earnings predictor cannot run** — the XGBoost model isn't on disk; the endpoint
  503s. (The earnings *calendar* works.)
- **F3 "Momentum picks" has no model or endpoint** — a label over the Alpha ranker.
- **Per-feature LLM caps are defined but never enforced** (only the copilot chat cap
  is); strategy_gen / scanner_thesis / vision / debate / doctor caps are dead config.
- **Artifact-vs-registry mismatch** — on-disk TFT (hidden=32) ≠ the registry's PROD
  `tft_swing v3` (hidden=128); on-disk LGBM is the legacy 15-feature model, not v2.
- **Embedded on-page "agents" were templated text** (4 of 5) — now fixed to hand off
  to real Main Chat.
- **Pro chat cap mismatch** — backend 50/day vs pricing UI 150/day.

### 🚨 Business / legal risk (non-code)
- **SEBI Research-Analyst registration is `PENDING_APPROVAL`** while the app
  distributes buy signals and runs an auto-trader. Biggest single risk.
- **No live alpha is proven anywhere** — the gate proves historical robustness, not
  future profit, and the daily signal feed bypasses the gate entirely.
- **Data defaults to Tier-3 yfinance** unless `DATA_PROVIDER=kite` is set in prod.

---
<a name="appendix--every-caveat-by-domain"></a>
## Appendix — every caveat, by domain


### AI/ML models

- 'FinBERT-India' / public 'Mood' engine is NOT FinBERT at runtime by default — it's an LLM zero-shot classifier via OpenRouter (engine.py:52-75, llm_classifier.py). Real FinBERT only runs with USE_FINBERT_FALLBACK=1. The public 'domain-tuned sentiment model' branding overstates this.
- Artifact-vs-registry mismatch: on-disk TFT is a small pytorch-forecasting net (hidden=32, encoder=120, ~93 symbols) but registry PROD tft_swing v3 describes a different neuralforecast model (hidden=128, 0.68 acc). The loader only matches the small on-disk one, so the 0.68 accuracy is unverified for what actually runs locally.
- On-disk LGBM gate is the LEGACY 15-feature model (verified num_feature()=15), fed by split_feature_sets() in the live loop — NOT the 30-feature v2 the DB metrics describe (compute_lgbm_v2_features exists but is unwired). lgbm_signal_gate is also NOT flagged is_prod in model_versions yet is required by SignalGenerator.
- Honest performance: Qlib's own metadata says qlib_realmoney_pass=false ('not safe for autonomous trading'), IC~0.031, ICIR~0.36. LGBM gate CV Sharpe ~0.097 (≈ no edge), per-fold Sharpe swung -7.9 to +5.6. These are display/ranking-grade, not proven alpha.
- HMM 'confidence' is degenerate ~1.0: transition matrix diagonal is 0.987/0.931/0.966 (verified from the PROD pickle), so posterior of the decoded state is structurally near-100% daily. No validated regime-call accuracy metric exists — only avg log-likelihood (-5.1/obs).
- AutoPilot is NOT a trained model — it's a Kelly+VIX+HMM orchestrator over the Qlib ranker; its edge is entirely inherited from the (weak) Qlib IC.
- breakout_meta_labeler (RandomForest 500x3) is Scanner-Lab-only: SignalGenerator sets _ml_labeler=None and never calls it. Old '63.6% WR' claim is retired (~35-40% OOS with costs).
- Many named models (finrl_x_*, momentum_timesfm/chronos, vix_tft, chronos2_macro, intraday_lstm, earnings_xgb) appear in the registry but are RETIRED or never-PROD — none are live. TimesFM/Chronos entries are zero-shot pretrained wrappers, not India-trained.
- Qlib is initialised with region=REG_CN (China calendar) on NSE data (engine.py:84) — a known Qlib-on-India config quirk to be aware of.
- The genuinely solid claim: no-fallback discipline is real and enforced — SignalGenerator raises if any model artifact is missing and the pipeline stays dark rather than degrading to heuristics (generator.py:123-188, app.py:401-427).

### LLM agents

- Real, working custom GraphRunner runtime exists (base.py) with 3 graphs: Copilot (4 nodes), Doctor (5), Debate (7) — not LangChain. This part is genuine.
- All 10 agent roles run on FREE OpenRouter model slugs (.env:144) — the planned 235B strategy generator and tiered brain map from the memory docs are NOT in the live config; strategy_generator runs on qwen3-coder:free.
- BIGGEST GAP: 4 of 5 on-page 'embedded agents' (Screener/Strategy/F&O/Analysis) are NOT LLMs — they fetch data and template the 'narration' string in TypeScript with a typewriter animation. Only the /markets 'Mood' card calls a real LLM. Real LLM chat otherwise only happens on /copilot.
- LLM_FEATURE_CAPS (tiers.py:129) for strategy_gen, scanner_thesis, chart_vision, debate, fno_advisor, portfolio_doctor are DEFINED BUT NEVER ENFORCED at runtime — grep shows llm_feature_cap() is never called outside tiers.py. Only the copilot 'chat' cap is enforced, via a separate duplicated COPILOT_DAILY_CAPS table.
- The $20/mo kill-switch (llm_budget.py) is real and correct: free slugs priced $0 bypass it, paid calls blocked over budget. Caveat per the code: single-instance accurate; multi-instance can overshoot by one 60s TTL window.
- The 'FinBERT-India' sentiment classifier is, in the live path, an LLM prompt at runtime (Llama-70B-free zero-shot), not the FinBERT model — FinBERT is only a shadow/fallback behind USE_FINBERT_FALLBACK.
- Agents never directly execute or gate trades. Debate 'enter', Doctor 'add', F&O suggestion are all advisory. Only the Studio NL->DSL path reaches live, and only after the backtest/Sharpe gate.
- FinRobot ManagementAgent often runs on empty inputs (concall transcripts stubbed, headlines usually absent) so 'management tone' is frequently a neutral no-op.
- Portfolio Doctor and Debate enforce only the tier gate (RequireFeature/RequireTier), not the documented per-run caps; the route code itself admits Doctor rerun-limiting is deferred/client-side only.
- Gemini is fully removed (only a doc-URL comment remains in llm_pricing.py); the conversational layer is 100% OpenRouter open models.
- Scanner/screener 'AI thesis' has a deterministic code-written fallback (_deterministic_thesis), so under free-tier rate-limits the user may be shown templated text rather than an LLM thesis.
- Vision (Gemma-4-free) is genuinely multimodal but its support/resistance levels are the model's read of a PNG, not computed levels — illustrative, not authoritative.

### Features

- F2 swing signals are the genuine ML product: real 5-model ensemble (LGBM+TFT+Qlib+FinBERT+HMM) with files on disk (ai/signals/voters.py, ml/models/*). But generator.py's docstring still claims '6 backtested strategies' / win-rate numbers the code itself disclaims — stale and overstated.
- F3 'Momentum picks' has NO dedicated backend endpoint or model — no momentum_weekly route, no momentum signal_type. It's a marketing label served via the screener/Alpha ranker.
- F9 Earnings predictor is wired but cannot run: the XGBoost model is not on disk (only scripts/train_earnings_scout.py exists); predict_surprise raises ModelNotReadyError -> 503 (ai/earnings/predictor.py:42-100). The earnings calendar works; the prediction does not.
- All 'AI agent' features (Main Chat, Portfolio Doctor's 4 agents, 7-agent Counterpoint debate, F&O advisor, Strategy Studio NL->DSL, Chart Vision) are OpenRouter LLM prompts at runtime, not trained models (ai/agents/llm.py).
- By default they all hit ONE free Llama-3.3-70B: LLM_DEFAULT_MODEL=meta-llama/llama-3.3-70b-instruct:free (config.py:253) and AGENT_MODEL_MAP is an empty env var (config.py:258). The per-agent premium models (Qwen-235B generator, DeepSeek debate) from project memory are NOT wired unless that env JSON is set.
- Public 'engines' (lib/engines.ts) are a 4-name brand layer (Alpha/Mood/Regime/AutoPilot) hiding real architectures; AutoPilot isn't a model — it's a rebalancer over the other three.
- Marketing engine names in FeatureGrid.tsx — VolCast, AllocIQ, InsightAI, EarningsScout, Trajectory — correspond to NO real code modules.
- Scanner count: the '50+ scanners' header claim is honest — live SCANNER_MENU has 77 entries (data/screener/engine.py); the confluence scoring system separately uses 24 weighted categories.
- F6 F&O is mostly rule-based by its own admission (ai/fo/strategies.py:16 'the rule layer that ships today; no RL yet'); only the optional /ai-suggest advisor is an LLM and it never auto-deploys.
- F7 Portfolio Doctor's '4 agents' are 4 LLM prompts (Fundamental/Management/Promoter/Peer in ai/agents/finrobot.py) plus deterministic concentration/sector/regime rule flags.
- Strategy Studio's real value is the ENFORCED backtest gate (evaluate_gate + run_walk_forward, out-of-sample Sharpe/drawdown/holdout) that blocks going live with a 422 (strategies_routes.py:648-701) — genuinely built. The NL->DSL compile is just an LLM.
- Marketplace browse (Free) + deploy (Pro) are real and tier-gated; the publish/creator side (Elite in FEATURE_MATRIX) has no creator-apply or revenue flow — deferred per project memory.
- AutoPilot (F4) and AI SIP (F5) are real and wired to an actual broker order path (services/live_executor.py -> TradeExecutionService.place_order); AutoPilot is supervised-only (RL removed), with 5%/80% caps and a -10% drawdown breaker.
- F1 intraday reads from the signals table but the intraday training modules (backend/ai/intraday/training/*) were DELETED in this branch (git status), so intraday signal production is uncertain.
- F12 daily digest is mostly templated/deterministic with an optional LLM prose intro that silently falls back to a template — honestly labeled in-code as 'presentation, not a model substitution'.
- Tier source of truth is tiers.py FEATURE_MATRIX; some keys are aspirational (marketplace_publish) and FeatureGrid tier badges don't always match the matrix (e.g. F7 Doctor badged 'Pro' but the matrix gives Free 1/mo).

### Pipeline / tiers / money

- DATA_PROVIDER defaults to 'free' yfinance (Tier-3 data) unless prod sets DATA_PROVIDER=kite — config.py:77, market.py:117-125. The premium Kite path exists but is off by default.
- Kite data depends on an admin token expiring 6 AM IST daily; auto-refresh logs in headlessly with stored password+TOTP (kite.py:947). If it breaks, data silently degrades to jugaad or nothing.
- SEBI RA registration is NOT obtained — SEBI_RA_REG_NUMBER defaults to 'PENDING_APPROVAL' (config.py:94), yet the app distributes buy signals and runs an auto-trader.
- No live alpha is proven anywhere in code. The strategy gate only proves out-of-sample historical robustness, not future profit; the daily signal feed bypasses the gate entirely.
- The 5 ensemble voters (LGBM/TFT/Qlib/HMM/FinBERT) are REAL trained model artifacts on disk (ml/models/), not LLM prompts — this part is genuine.
- Pro chat LLM cap mismatch: backend enforces 50/day (tiers.py:130) but pricing UI shows 150/day (FeatureComparisonMatrix.tsx:63).
- README.md is badly stale/misleading: claims '11 ML models', FinRL-X RL for AutoPilot (RL removed), deprecated brand names, and 'Copilot (Gemini)' (Gemini fully removed, now OpenRouter).
- AutoPilot per-trade 'thesis' is a deterministic rule-based template string (autopilot_service.py:494), not an LLM explanation — code says enrichment comes later.
- Swing signal engine is long-only NSE equity (generator.py:494) — no short signals.
- Prices confirmed in DB seed in paise: Free 0 / Pro 99900 (₹999) / Elite 199900 (₹1,999); caps: 1 signal/day free, 5 watchlist symbols free.
- The $20/mo LLM kill-switch is real and enforced (llm_budget.py + ai/agents/llm.py), but is single-instance accurate and can briefly overshoot by one TTL window on multi-instance deploys (its own docstring admits this).

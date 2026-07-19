# Quant X — Full App Audit: Every Feature, Agent & Model (Honest)

*Internal whole-app engineering audit for the founder. Plain English, not marketing.
Where something is **claimed but not real / placeholder / rule-based / unenforced**,
this says so. Generated 2026-06-06 by a five-way parallel code audit across all
**62 frontend pages**, **~300 backend routes**, the ML models, the LLM agents, and
the automation/data/billing/security layer. Grounded in the live codebase + the
Supabase `model_versions` table. Companion to `QUANTX_AI_SYSTEMS_EXPLAINED.md`.*

---

## How to read this

The app has three layers people conflate — keep them separate:

1. **Features** — the 62 pages a user clicks (most fetch real backend data).
2. **AI agents** — open-source LLMs that explain / narrate / build (never trade live).
3. **AI models** — the 5 trained models that produce the actual signals.

Below: the full user-facing feature map → the backend that serves it → the models →
the agents → the background machinery (crons, data, billing, security) → an honest
scorecard → every caveat.

---

## Executive summary (the 90-second version)

**Scale & polish.** 62 pages, ~300 routes. The app is **large, polished, and mostly
real** — almost every page fetches live backend data, not mockups. It *looks* like a
finished institutional terminal.

**The signal brain (ML).** Live swing signals come from **5 real trained models** fused
into one ensemble — Qlib **Alpha158** (rank), **TFT** (Forecast/levels), **LightGBM**
(Gate), **HMM** (Regime), **sentiment** (Mood). Real artifacts, real *no-fallback
discipline* (pipeline goes dark rather than fake it). **But** measured edge is weak and
the code's own metadata says so (Qlib `realmoney_pass: false`, IC ≈ 0.03; gate Sharpe
≈ 0.1). **Display/ranking-grade, not proven alpha.**

**The talking layer (LLM).** A custom **GraphRunner** runtime (not LangChain) runs 3
graphs — Copilot chat (4), Portfolio Doctor (5), Counterpoint debate (7) — plus
single-shot roles. **All ~10 roles on FREE OpenRouter models** under a hard **$20/mo
kill-switch**. Agents are **advisory only**; the single path to a live trade is the
Strategy Studio, behind an **enforced out-of-sample backtest gate**.

**The product (features).** ~15 headline features across **Free / Pro ₹999 / Elite
₹1,999**, plus a full admin suite. Crown jewels: real ML signals + the enforced gate +
a genuinely good UI + honest data-gating (hides empty track records rather than faking).

**The honest gaps a founder must own:**
- **Sold-but-not-built:** `/portfolio/sip` and `/portfolio/rebalance` are **skeleton
  placeholders**, yet AI SIP is sold as an Elite feature. AutoPilot's **Enable button is
  hardcoded OFF** (`AUTOPILOT_LIVE_TRADING=false`). F9 Earnings predictor **503s** (model
  never trained; the calendar works).
- **Overstated:** `/engines` says "Eight engines" but the code defines **four**; "Mood"
  is an **LLM at runtime**, not FinBERT; "12 models / institutional AI" marketing
  outruns the real 5-model, weak-edge reality.
- **Unenforced:** per-feature LLM caps are defined but only the chat cap is enforced
  (the **$20 kill-switch** is the real backstop).
- **Risk:** **SEBI RA = PENDING_APPROVAL** while distributing signals + an auto-trader;
  **no proven live alpha**; data defaults to **Tier-3 yfinance**; WhatsApp delivery is
  **dormant**; the kill-switch **fails open** (a DB outage won't halt trading).

**Bottom line:** a genuinely well-engineered platform with rare discipline (no-fallback
models, enforced gate, budget cap, removed RL, honest empty-states) — wrapped around an
**unproven edge**, a **few placeholder features that are being sold**, and **marketing
that outruns the substance**. The engineering is the moat; closing the
sold-vs-built gaps and proving alpha is the work.

---

## Table of contents
1. [User-facing features — the whole frontend](#user-facing-features)
2. [Backend API surface](#backend-api-surface)
3. [AI/ML models](#aiml-models)
4. [AI agents](#ai-agents)
5. [Automation, data, billing & security](#automation-data-billing--security)
6. [Honesty scorecard — whole-app real vs claimed](#honesty-scorecard)
7. [Appendix — every caveat, by domain](#appendix--every-caveat-by-domain)

---

<a name="user-facing-features"></a>

## User-Facing Features — The Whole Frontend, Walked Page by Page

This is the honest, plain-English map of *everything a user can actually do* in the app — all 62 routes under `frontend/app/`. For each feature I say what it is, who it's for, what backs it, and where it's thin, fake, or unbuilt. I read every page file; citations are `path:line`.

> **One-line truth up front:** The frontend is large, polished, and *mostly real* — almost every page fetches live backend data, not mockups. But three things a founder must internalize: (1) a handful of *advertised* features are visibly stubbed or hardcoded-off in the UI (AutoPilot live trading, SIP, Rebalance, Earnings predictor); (2) the AI "engines" copy oversells (the `/engines` page says "Eight engines" but the code defines four — `lib/engines.ts:26`); and (3) the underlying signal edge is weak (known ground truth), yet the marketing surfaces (`/`, `/pricing`) lean hard on "institutional AI / 12 models." The product *looks* like a finished institutional terminal; the substance is an honest-but-early signal engine wrapped in a very good UI.

---

### Navigation & shell — what the user sees first

The sidebar (`frontend/components/shell/nav.ts:32`) is the real IA, and it's deliberately *smaller* than the 62 routes. Only 12 items show: **Workspace** (Command Center, Markets, Signals, Strategies, F&O `elite`, Scanner, Stocks, Portfolio, Watchlist, AutoPilot `elite`), **AI** (Main Chat), **Account** (Inbox, Settings). Everything else — `/trades`, `/engines`, `/paper-trading`, `/referrals`, `/track-record`, `/models` — is reachable only via deep-links, the command palette, or in-page CTAs. So the route count overstates the navigable surface. Note the persistent "Cursor-style copilot rail" described in MEMORY is **not** what shipped here — the nav has a single "Main Chat" link, and agents are embedded inside feature pages instead.

**Auth & redirects** (`middleware.ts`): public pages are `/`, auth pages, `/pricing`, `/privacy`, `/terms`, plus public-trust prefixes `/models /track-record /regime /markets /chart-test`. Everything else requires a session cookie. 26 retired v1 URLs 301-redirect to their v2 homes (`middleware.ts:11`) — so old links in `/assistant`'s sidebar (`/screener`, `/swingmax-signal`, `/pattern-detection`, `/alerts`) still resolve, they just bounce to `/scanner`, `/signals`, `/scanner`, `/inbox`. Good hygiene, but it means some in-app links point at dead names that *look* broken in code.

---

### MASTER TABLE — every user-facing page

| Feature / Page | Route | Tier | What the user does | Backed by |
|---|---|---|---|---|
| **Landing** | `/` | Public | Marketing hero, live Nifty/regime/top-signal tiles, feature grid, pricing preview | Real public endpoints (`indices`, `regimeHistory`, `signalOfTheDay`) `page.tsx:198` |
| **Signals hub** | `/signals` | Free (1/day) | See 3 horizon cards w/ live open counts + KPIs, jump into a horizon | `api.signals.getToday()` `signals/page.tsx:25` |
| **Swing / Intraday / Positional** | `/signals/swing` `/intraday` `/positional` | Free→Pro/Elite | Per-horizon explainer, win-rate/annual-return (gated ≥8 closed trades), Opening/Closed signal cards | Shared `CategorySignalsPage`; real today+history feeds |
| **Signal detail** | `/signals/[id]` | Free; debate=Elite | Chart, AI thesis, levels, paper/live trade buttons, alert toggles, Counterpoint debate, prior signals | `getById`, `earnings.symbol`, `ai.debate` (Elite) |
| **Stocks browser** | `/stocks` | Free | Browse ~48 popular NSE names, live price/change, sector filter, signal chips | `screener.getLivePrices`, real regime banner |
| **Stock terminal** | `/stock/[symbol]` | Free; vision Pro+ | Full-viewport TradingView chart + AI side-drawer (Dossier/Vision/Insight) | TradingView widget + per-stock AI endpoints |
| **Markets desk** | `/markets` | Public (Mood=auth) | Pre-market research: global cues, regime, FII/DII, sector rotation, headlines, earnings, Mood agent | Real market/screener/news feeds; Mood = LLM, user-triggered |
| **Regime** | `/regime` | Public | Current regime hero, 90-day timeline, regime→sizing table, methodology | `publicTrust.regimeHistory(90)` |
| **Watchlist** | `/watchlist` | Free cap 5, Pro+ ∞ | Track symbols w/ engine consensus + sentiment + events; cap enforced server-side | `watchlist.live()`, `UsageMeter` reads `data.cap` |
| **Strategies** | `/strategies` | Free browse | Library (tested templates), My strategies, **Builder** (NL→DSL→backtest), Deployed, **Discovered** | `catalog/sections`, `studioCompile`, `backtest`, discovery engine |
| **Strategy template / mine** | `/strategies/[slug]` `/mine/[id]` | mixed | View a template or own strategy, transition draft→paper→live | `strategies.*` |
| **Deployed strategies** | `/strategies/deployed` | Pro+ | Live MTM P&L per deployed strategy, open positions, pause/resume | `strategies.deployed()` (30s refresh) |
| **F&O Desk** | `/fno` | Elite | Overview, stock OI scanners, OI heatmap, payoff calc, link to Lab; F&O Advisor agent | Real Kite/NSE feeds; advisor agent |
| **F&O Strategy Lab** | `/fo-strategies` | Elite | Rule-based multi-leg recs, deploy-to-paper, live MTM, backtest modal, **Ask AI** leg builder, 5-view option chain | `foStrategies.*`; recs **rule-based**, AI suggest = LLM advisory |
| **Portfolio** | `/portfolio` | Free | KPIs, equity curve, positions table, Doctor + Ask-AI CTAs | `positions.getOpen`, `portfolio.getHistory` |
| **Portfolio Doctor** | `/portfolio/doctor` | Pro (₹199 free one-off), Elite ∞ | Enter holdings → 4-agent per-position scoring + composite + risk flags + PDF | `portfolioDoctor.analyze` (CoT agents + real fundamentals) |
| **Rebalance** | `/portfolio/rebalance` | — | **PLACEHOLDER** — only `<Skeleton>` rows, "Plan 3 wires backend" | Nothing wired `rebalance/page.tsx:1-100` |
| **AI SIP** | `/portfolio/sip` | Elite (label) | **PLACEHOLDER** — all skeletons, no data fetch | Nothing wired `sip/page.tsx:1-85` |
| **AutoPilot** | `/autopilot` | Elite | Status, regime/VIX overlay, safety-rail sliders, kill switch, today's plan, rebalance log | `autoTrader.*` reads real state; **Enable is hardcoded OFF** `autopilot/page.tsx:58` |
| **AutoPilot track record** | `/autopilot/track-record` | Elite | Realised P&L / win-rate / Sharpe / drawdown, paper vs live, window selector | `autoTrader.trackRecord` (real trades, honest "0 trades yet") |
| **Paper trading** | `/paper-trading` | Free | ₹10L virtual equity curve vs Nifty, achievements, Paper League, reset | `paper.*` (real) |
| **Main Chat (Copilot)** | `/copilot` | Free 5 / Pro 50 / Elite 200 msgs/day | Streaming chat orchestrator, agent pills, marketing bands w/ live data | `/api/ai/copilot/chat` (LLM, free OpenRouter) |
| **Assistant (legacy)** | `/assistant` | Free | Finance Q&A chat | **Redirects to `/copilot`** via middleware; page file still exists |
| **Engines** | `/engines` | (in-app) | Grid of engine cards → per-engine detail | `lib/engines.ts` — **4 engines, but header says "Eight"** |
| **Engine detail** | `/engines/[slug]` | (in-app) | Per-engine methodology, accuracy qualifier | `getEngineBySlug` |
| **Models (public)** | `/models` | Public | Per-model accuracy cards, window selector | `publicTrust.models` (weekly `model_rolling_performance`) |
| **Track record (public)** | `/track-record` `/[id]` | Public | Every closed signal w/ realised P&L, stats, engine chips | `publicTrust` track feed |
| **Trades journal** | `/trades` `/[id]` | (in-app) | Full trade log, filters, export | `api` trades feed |
| **Settings** | `/settings` | All | Profile, trading prefs, broker OAuth, notifications, appearance (theme), tier, kill switch, data | `api` + sub-panels |
| **Pricing** | `/pricing` | Public | 3 tiers, Razorpay checkout, full feature×tier matrix | `api` plans + `FeatureComparisonMatrix` |
| **Inbox** | `/inbox` `/[id]` | All | Real notifications feed, tabbed, mark-read | `notifications.getAll()` |
| **Referrals** | `/referrals` | All | Share links, referral status table, +1mo Pro credit copy | `referrals.status` |
| **Onboarding** | `/onboarding/{risk-quiz,broker-connect,complete}` | New users | 5-Q risk quiz → profile + tier rec; broker connect; finish | `/api/onboarding/quiz` |
| **Auth** | `/login` `/signup` `/forgot-password` `/verify-email` `/auth/callback` `/broker/callback` | Public | Sign in/up, reset, email verify, OAuth callbacks | Supabase auth |
| **Admin** | `/admin` + `/admin/{ml,model-performance,payments,signals,system,training,users,users/[id]}` | Admin only | Internal dashboards: users, payments, signals, ML/training, system health | Gated on `is_admin` `admin/layout.tsx:89` |
| **Legal** | `/privacy` `/terms` | Public | Static legal pages | static |
| **chart-test** | `/chart-test/[symbol]` | Public | Dev chart harness (not real feature) | dev only |

---

### Feature group walkthrough (prose)

#### 1. Signals — the core product, and it's real
The signals surface is the most mature part of the app. `/signals` is a hub showing three horizon cards (Intraday / Swing / Positional) with **live open-signal counts** from `api.signals.getToday()` (`signals/page.tsx:25`). All three horizon pages share one component (`CategorySignalsPage.tsx`) — so they're genuinely identical in behaviour, just different copy and filters. **Honest mechanic worth knowing:** win-rate and annualised return are *suppressed until ≥8 closed trades* (`categories.ts:149`, `MIN_SAMPLE = 8`) and show "—" otherwise. This is the no-fabricated-numbers discipline showing up in the UI, and it's the right call. The signal detail page (`/signals/[id]`) is genuinely rich: real TradingView chart, AI thesis (falls back to a templated explanation if the LLM text is empty — `signals/[id]/page.tsx:625`), levels, working alert toggles wired to global push prefs, and the Elite-only Counterpoint Bull/Bear debate (`api.ai.debate`). Caveat: the "Paper-trade" and "Live trade" buttons both open the *same* QuickTrade modal (`signals/[id]/page.tsx:439-447`) — "Live trade" doesn't yet branch into a broker flow at this layer.

A subtlety for the founder: the three horizons are a **presentation split, not three models**. `categoryOf()` (`categories.ts:122`) buckets signals by a string tag, and momentum + anything untagged falls into "swing." Per ground truth, F3 momentum has no dedicated model and intraday relies on the same swing stack — so "Intraday signals" and "Positional signals" are largely the swing engine re-sliced by holding period, dressed in distinct copy. That's not dishonest, but it's not three independent engines either.

#### 2. Stocks & Markets — strong research surfaces, mostly live data
`/stocks` is an honest browser (the file comment at `stocks/page.tsx:18` explicitly says it *removed* the old fake hashed "AI score" badges) — real prices, real regime banner, real signal chips. It's capped to ~48 hardcoded popular names (`POPULAR`, `stocks/page.tsx:100`) with a client-side sector map because the backend doesn't return sector on the live-prices endpoint — a real limitation, not the full 2,000-name universe the landing claims. `/stock/[symbol]` is a chart-first terminal with an AI side-drawer (Dossier/Vision/Insight). `/markets` is the best pre-market page: global cues, FII/DII smart-money, sector rotation, headlines, earnings — all real feeds — plus the **Mood agent** (the one LLM on the page, auth-gated, user-triggered). `/regime` is a clean public methodology page; note its explainer still describes **"five AI voters"** (`regime/page.tsx:272`) including a "Forecast (5-day price forecast)" and "Gate" — consistent with the 5-model ground truth, but the public `/engines` page only lists 4. The user gets two different engine counts depending on which page they're on.

#### 3. Strategies — genuinely powerful, the standout AI feature
This is the most impressive build. `/strategies` has five tabs: **Library** (tested DSL templates ranked by Sharpe, with legacy-catalog fallback), **My strategies** (real draft→paper→live state machine with transition + archive), **Builder** (the real one: plain-English → `studioCompile` → DSL preview → save → inline backtest — `strategies/page.tsx:661`), **Deployed**, and **Discovered** (a Strategy Discovery Engine that runs batch discovery and persists candidates — `DiscoveredTab.tsx`). The Builder is the clearest expression of the "describe it, we test it" promise and it's wired end-to-end, including the backtest gate the strategy must clear before going live. `/strategies/deployed` shows live 30s-refresh MTM P&L per strategy with win-rate-vs-backtest deltas. This group is real and differentiated. The honest caveat is upstream: the strategies *compile and backtest* fine, but per ground truth there's no proven live alpha, so a beautiful walk-forward Sharpe in the Builder is not evidence of money-making.

#### 4. F&O — two overlapping pages, both Elite, recommendations are rule-based
There are deliberately two F&O surfaces and they are **not duplicates**: `/fno` is the orientation hub (overview, OI scanners, heatmap, payoff calc) and `/fo-strategies` is the deep workspace (recommendations, paper deploy, live MTM, backtest, AI leg builder, 5-view option chain). The `/fno` page links into `/fo-strategies` rather than embedding it (`fno/page.tsx:141`). **Critical honesty point the UI itself states:** the recommendation grid is **rule-based**, not ML — the page header literally says "Rule-based weekly multi-leg recommendations" (`fo-strategies/page.tsx:267`) and the engine produces 1–2 ranked structures per index by regime + VIX slope. The "Ask AI" modal *is* an LLM, but it's advisory only — it picks a template and the user must click Deploy (`fo-strategies/page.tsx:601`, honouring the "LLM never places trades unilaterally" lock). Both pages degrade gracefully when the OI feed is down (show source/error tags, not synthetic numbers). This is well-built but a founder should never describe F&O recs as "AI" — they're rules.

#### 5. Portfolio — real core, two dead sub-pages
`/portfolio` is real (live positions, equity curve, Recharts). **Portfolio Doctor** (`/portfolio/doctor`) is a genuine, valuable feature: enter holdings (must sum to 100%), four AI agents score each position (risk/exposure/momentum/sentiment) and roll up to a composite + risk flags + printable PDF, with a real monthly quota meter and persistence to `portfolio_doctor_reports`. Per ground truth it's now fed real screener.in fundamentals — so this is one of the strongest AI surfaces.

**But two advertised portfolio pages are pure placeholders.** `/portfolio/rebalance` is *only* skeleton loaders with the comment "Plan 3 wires the actual order placement" (`rebalance/page.tsx:21`) — the "Approve & place" button does nothing. `/portfolio/sip` (AI SIP, labelled Elite) is *entirely* `<Skeleton>` with "Plan 3 wires backend" (`sip/page.tsx:18`). Yet **AI SIP (F5) is sold as an Elite feature on the pricing matrix** (`FeatureComparisonMatrix.tsx:77`). A paying Elite user clicking SIP gets a blank skeleton screen. This is the most direct buy-vs-get gap in the product.

#### 6. Automation — AutoPilot dashboard real, live trading hardcoded off
`/autopilot` reads real state (broker status, regime, VIX overlay, configurable safety rails, kill switch, today's target weights, rebalance log). **But the Enable button is permanently disabled by a hardcoded flag:** `const AUTOPILOT_LIVE_TRADING = false` (`autopilot/page.tsx:58`), and the page renders a "Closed beta — live trading currently disabled" banner. The user can configure everything but cannot turn it on. Two notes: (a) the page comments still reference "RL ensemble (PPO + DDPG + A2C)" and a "FinRL-X engine" (`autopilot/page.tsx:52`, `:440`) — stale, since MEMORY says RL was fully removed and AutoPilot is now a supervised ranker + Kelly sizing; the user-visible banner correctly says "Sharpe ≥ 1.0 calibration bar" but the code comments lie. (b) AutoPilot is sold as an Elite feature on pricing and prominent in nav, but is functionally a preview. `/paper-trading` (the free gamified sandbox with Paper League) is fully real and is the actual hands-on execution experience most users will get.

#### 7. AI surfaces — Main Chat real, Engines page oversells
`/copilot` (Main Chat) is the real streaming orchestrator with per-tier message caps (5/50/200) and a marketing home built on live public data. `/assistant` is **retired** — middleware redirects it to `/copilot` (`middleware.ts:37`), though the page file lingers with stale sidebar links to dead route names. So `/assistant` vs `/copilot` is **not** a live duplicate — only `/copilot` is reachable. The weakest AI surface is `/engines`: its header says **"Eight engines power every signal"** and "the 8 ML/DL engines" (`engines/page.tsx:19`, `:33`), but `lib/engines.ts` defines exactly **four** (Alpha, Mood, Regime, AutoPilot — `engines.ts:26`). So the page renders 4 cards under a headline promising 8. Combined with the landing's "12 AI models" badge (`page.tsx:57`) and "Five AI engines" body copy, the user is told 12, then 8, then 5, then shown 4. The real count is 5 signal models. This inconsistency is the single most fixable credibility problem in the UI.

#### 8. Account, auth, onboarding, admin — solid plumbing
Settings is comprehensive (broker OAuth, real kill switch, theme toggle lives here per the header-less IA decision, tier panel). Onboarding is a real 5-question risk quiz driving tier recommendations. Inbox is a live notifications feed (no longer the old stub). Referrals, trades journal, pricing+Razorpay, public track-record/models are all real. Admin is properly gated on `is_admin` server flag (`admin/layout.tsx:89`) and covers users/payments/signals/ML/training/system — internal tooling, not user-facing.

---

### Where claimed ≠ real (the founder's punch-list)

1. **`/portfolio/sip`** — sold as Elite F5, but the page is 100% skeleton placeholder, no backend (`sip/page.tsx`). A paying user gets a blank screen.
2. **`/portfolio/rebalance`** — skeleton placeholder; "Approve & place" is inert (`rebalance/page.tsx`).
3. **AutoPilot live trading** — hardcoded `false` (`autopilot/page.tsx:58`); Enable button can never activate from the UI. Sold as the Elite flagship.
4. **`/engines` "Eight engines"** — copy says 8, code defines 4 (`engines/page.tsx:19` vs `engines.ts:26`). Landing says "12 models," "Five engines." Real = 5 signal models. Four different numbers.
5. **F&O recommendations are rule-based, not AI** — the page says so honestly (`fo-strategies/page.tsx:267`), but pricing/marketing frame F&O as AI. Only the "Ask AI" leg-builder modal is an LLM, and it's advisory-only.
6. **F9 Earnings predictor** — on the pricing matrix (`FeatureComparisonMatrix.tsx:66`) but per ground truth the model isn't trained (calendar/heuristics only).
7. **Marketplace "Publish + earn revenue share"** — on pricing (`FeatureComparisonMatrix.tsx:97`) but MEMORY says the creator side is deferred out of v1.
8. **Stale RL/FinRL-X comments** in AutoPilot code contradict the removed-RL decision (cosmetic, but misleads any engineer reading it).
9. **Stale dead-route links** in `/assistant` sidebar (`/swingmax-signal`, `/pattern-detection`) — they redirect, so they work, but look broken in source.
10. **Three "horizons" of signals** are one swing stack re-sliced by hold period, not three independent engines.

### What's genuinely strong
The Strategy Builder + Discovery + Deployed flow, Portfolio Doctor (now on real fundamentals), the Markets pre-market desk, the signals detail page, the no-small-sample-numbers discipline (`MIN_SAMPLE=8`), honest "feed down → show error tag, not fake data" handling across F&O and Markets, server-side tier caps (watchlist, copilot msgs), and clean auth/redirect/admin gating. The UI quality is genuinely competitive-terminal grade — the gap is in a few specific unbuilt-but-advertised features and inconsistent engine-count marketing, not in build quality.

---

<a name="backend-api-surface"></a>

## Backend API Surface

This is the engine room: the FastAPI server that every screen, mobile push, and broker connection talks to. It exposes roughly **300 HTTP endpoints across 43 route modules** (the "305 across 43" figure in our planning docs; the precise count of route decorators in `backend/api/` today is **298**, including 12 deliberately hidden back-compat aliases). Routers are assembled in `backend/api/app.py` (lines 938–1334), where each feature module is mounted, most under the `/api` prefix.

The headline finding for a founder: **the API surface is overwhelmingly real, not a Potemkin village.** Where a feature isn't ready, the server is honest about it — it returns a `503 model_not_ready` or a `false` availability flag rather than faking output with a heuristic. That "no-fallbacks" discipline is genuinely wired into the routes (verified below). The real gaps are narrower and specific: a few per-feature spend caps are defined but not enforced, the compliance disclaimer ships a placeholder SEBI number, and a couple of models the API knows how to serve simply aren't trained yet.

### How access control actually works (the gating spine)

Before the feature tour, the mechanism matters because it's the same everywhere and it's solid.

- **Authentication**: 148 endpoints depend on `get_current_user` (real JWT verification, `core/security.py:15`). Public/marketing endpoints intentionally skip it.
- **Tier gating** is real and centralized. There is one source of truth — `FEATURE_MATRIX` in `core/tiers.py:53` — mapping each feature to its minimum tier (Free / Pro ₹999 / Elite ₹1,999). Routes enforce it two ways: a clean `RequireFeature("...")` dependency (`middleware/tier_gate.py:118`) that raises **HTTP 402 Payment Required** with an upgrade payload and fires a `TIER_GATE_HIT` analytics event, or inline `tier_rank(...)` comparisons for content that degrades rather than rejects. Admins bypass. I counted 66 `RequireFeature` guards across the API (e.g. 12 on F&O, 10 on AutoPilot, 5 on Scanner Lab).
- **The honest caveat on caps**: per-feature LLM usage caps are defined in `LLM_FEATURE_CAPS` (`core/tiers.py:129`) for chat, strategy-gen, scanner-thesis, chart-vision, debate, F&O-advisor and portfolio-doctor. **Only the `chat` cap is actually enforced** (copilot at `ai_routes.py:97`, assistant at `assistant_routes.py:69`, both returning 429 when exhausted). The other five caps are written down but **not checked at their endpoints** — a `grep` for cap enforcement in the F&O, screener, strategy and vision routes returns nothing. The $20/mo OpenRouter kill-switch is the real backstop, so the budget can't blow up, but the *per-user, per-feature* abuse ceilings are paper rules today.

### Capability map

| Cluster | Key modules | Representative endpoints | Tier | What backs it / honesty notes |
|---|---|---|---|---|
| **Swing signals** | `signals_routes` | `/api/signals/today`, `/intraday`, `/history`, `/performance`, `/{id}` | Free (1/day cap), Pro+ unlimited | Reads the real model-first pipeline output from the `signals` table. Free cap enforced (`signals_routes.py:44`). |
| **Strategy builder + gate** | `strategies_routes`, `strategy_runner_routes` | `/api/strategies` CRUD, `/validate`, `/studio/compile`, `/{id}/backtest`, `/{id}/transition`, `/deployed`, `/runner/run-now` | Build Free; live deploy Pro+ | DSL builder + real backtests. **Backtest quality gate is enforced** — going live raises 422 `gate_failed` unless the out-of-sample walk-forward passes (`strategies_routes.py:647`). This is the protection on the money. |
| **Strategy discovery** | `discovery_routes` | `POST /api/discovery/runs`, `/runs/{id}/candidates`, `/candidates/{id}/promote` | Gated | Real async LLM strategy generator (`ai/strategy_discovery/run_discovery`) whose candidates must clear the same backtest gate before promotion. |
| **Screener / QuantScan** | `screener_routes` (2,588 lines, 62 routes) | `/scanners/all`, `/scan/{id}`, `/v2/confluence`, `/v2/mtf-scan`, `/fno/oi-heatmap/{sym}`, `/prices/live`, `/ai/market-regime`, `/ai/swing-forecast/{sym}` | Mostly Pro (`scanner_lab`) | The biggest module. Most scanners are **rule-based technical filters** (correctly so). AI sub-endpoints are honest: `ai/nifty-prediction` returns 503 rather than a heuristic SMA/RSI stand-in (`:263`); the AI swing-forecast endpoint uses the real TFT predictor or 503s; `ai/ml-signals` reads real pipeline signals. 8 old paths (`/breakouts`, `/vcp`, `/fii-dii`, etc.) are hidden back-compat aliases. |
| **AI agents** | `ai_routes`, `assistant_routes`, `vision_routes` | `/ai/copilot/chat` (+`/stream`, conversations CRUD), `/ai/finrobot/analyze`, `/ai/debate/signal/{id}`, `/api/assistant/chat`, `/ai/vision/analyze/{sym}` | copilot Free (5/day); debate Elite; vision Pro/Elite | Backed by the custom GraphRunner LLM runtime on **free OpenRouter models**. Copilot chat has real SSE token streaming. Chat caps enforced; debate/vision tier-gated but their *usage* caps are not. Assistant is a separate finance-only chat with its own credit limiter. |
| **Stock dossier** | `dossier_routes` | `/dossier/{symbol}` (one large composed endpoint) | basic Free / full Pro | Aggregates model outputs + fundamentals into the stock page. Several internal `return {}` are graceful empty-fallbacks for missing sub-data, not stubs. |
| **Portfolio + Doctor** | `portfolio_routes`, `portfolio_doctor_routes`, `ai_portfolio_routes` | `/api/portfolio[/history,/performance]`, `/portfolio/doctor/analyze`, `/reports`, `/quota`, `/ai-portfolio/status`, `/rebalance/preview` | Doctor: Free one-off / Pro included / Elite unlimited; AI-SIP Elite | Doctor is the 5-node LLM graph fed **real screener.in fundamentals** (recent fix). Quota tracked. AI-portfolio (SIP) is Elite-gated. |
| **AutoPilot (auto-trader)** | `auto_trader_routes`, `autopilot_streams_routes` | `/auto-trader/status`, `/config`, `/toggle`, `/trades`, `/plan/today`, `/track-record`, `/compliance`; `/api/autopilot/streams` | Elite (`auto_trader`) | Supervised stack (Qlib + HMM + VIX). `track-record` returns realized P&L outcomes only (never per-model internals). **`/compliance` ships `SEBI_RA_REG_NUMBER` which defaults to `PENDING_APPROVAL`** (`config.py:94`) — the legal disclaimer is live but the registration is not yet real. |
| **F&O strategies** | `fo_strategies_routes` (1,155 lines) | `/fo-strategies/overview`, `/recommend/{sym}`, `/chain/{sym}`, `/vol-cone/{sym}`, `/term-structure/{sym}`, `/backtest`, `/ai-suggest`, paper open/close | Elite (`fo_strategies`) | Rule-based options strategies + real chain/vol-cone/term-structure math + an LLM `ai-suggest` advisor. F&O advisor cap defined but unenforced. |
| **Earnings** | `earnings_routes` | `/earnings/upcoming`, `/symbol/{sym}`, `/strategy/{sym}`, `POST /predict/{sym}` | basic Pro / strategy Elite | Calendar works (yfinance bootstrap). **The predictor model is not trained — `POST /predict` returns 503 `model_not_ready`** (`earnings_routes.py:249`). Honest. |
| **Market data** | `market_routes`, `public_routes` | `/api/market/{status,quote,indices,global,regime,ohlc}`, `/api/ai/performance`; `/public/{regime/history,track-record,models,models/status,signal-of-the-day,indices,system/status}` | Market Free; public no-auth | Real quotes/indices/regime. `public/models/status` honestly returns `false` for `earnings_scout` (F9) and the F1 intraday model so the UI hides them (`:440`). **DATA_PROVIDER defaults to `free` = yfinance Tier-3** (`config.py:77`), not the ₹2k Kite feed. |
| **Watchlist + Alerts** | `watchlist_routes`, `watchlist_live_routes`, `alerts_routes` | `/api/watchlist` CRUD, `/watchlist/live`, `/limits`, `/alerts/preferences`, `/alerts/test` | basic Free (5-symbol cap), unlimited Pro; alert studio Pro | Real. Free 5-symbol cap enforced via tier rank. |
| **Marketplace** | `marketplace_routes` | `/api/marketplace/strategies[/{slug}][/backtest]`, `/deploy`, `/my-strategies`, deployment PUT/DELETE | browse Free / deploy Pro / publish Elite | Browse + deploy real; **creator/publish side is deferred per project decision** — publish tier exists but the creator apply + revenue-share flow is out of v1. |
| **Paper trading** | `paper_routes` | `/api/paper/{portfolio,orders,order,reset,price}`, `/v2/{equity-curve,league,achievements}` | Free (conversion funnel) | Real paper engine on `paper_*` tables. League + achievements live. |
| **Live trades + positions** | `trades_routes`, `positions_routes` | `/api/trades`, `/execute`, `/{id}/approve`, `/{id}/close`, `/kill-switch`; `/api/positions/open`, `/{id}` PUT/close | Auth | Real execution + a kill-switch. `/api/positions` (no suffix) is a hidden alias. |
| **Broker (OAuth)** | `broker_routes` (934 lines) | `/broker/{status,connections,connect,disconnect}`, `/zerodha/auth/{initiate,callback}`, `/angelone/...`, `/upstox/...`, `/positions`, `/holdings`, `/margin` | Auth | **Genuine direct OAuth** — Zerodha initiate builds the real `kite.zerodha.com/connect/login` URL and the callback does the real HMAC token exchange against `api.kite.trade` (`broker_routes.py:487+`). Angel + Upstox flows present. No OpenAlgo, no Alpaca (per locked decisions). |
| **Notifications + comms** | `notifications_routes`, `push_routes`, `telegram_routes`, `whatsapp_routes` | `/api/notifications[/read]`, `/api/push/{vapid-key,subscribe}`, `/telegram/link/*` + `/webhook/{secret}`, `/whatsapp/link/*` + digest toggle | telegram digest Free; whatsapp Pro | Real. Telegram webhook has proper dual-secret verification (URL segment + header, constant-time compare). |
| **Onboarding / referrals / billing** | `onboarding_routes`, `referrals_routes`, `subscription_routes`, `payment_routes` | `/onboarding/quiz`, `/referrals/{status,rotate-code,resolve,attribute}`, `/api/plans`, `/payments/{create-order,verify,webhook,history,subscription-status}` | mostly Free | Real Razorpay integration (create-order, signature verify, webhook). Referrals + onboarding quiz live. |
| **User / auth** | `user_routes`, `auth_routes` | `/api/user/{profile,tier,stats,ui-preferences}`, `/api/auth/{signup,login,refresh,logout,forgot-password,me}` | Free/auth | Real. **Logout is a client-side placeholder beacon** (`auth_routes.py:156`) because JWTs are stateless — expected, not a bug. |
| **Admin** | `admin_routes` + `admin/{users,system,ml,eod,payments,training,observability}.py` | `/admin/users` (suspend/ban/export-csv), `/system/global-kill-switch`, `/ml/{performance,drift,retrain}`, `/training/{trainers,runs,run}`, `/launch-readiness`, `/llm-cost`, `/audit-log`, `/kite/refresh-token` | Admin role (`require_role`) | Real ops console. Role-gated via `admin/_deps.py:181`. Includes the global kill-switch, LLM cost tracking, and training orchestration. |
| **Misc / telemetry** | `dashboard_routes`, `system_routes`, `weekly_review_routes`, `telemetry_routes` | `/api/dashboard/overview`, `/api/system/status`, `/weekly-review/{latest,history,generate}`, `/client-errors`, `/upgrade-intent` | mixed | Dashboard overview is a real parallel aggregation from Supabase. Telemetry captures client errors + upgrade-intent analytics. Weekly review Pro-gated. |

### Bottom line for the founder

- **What's solidly real (most of it):** signals, strategies + the live-deploy backtest gate, screeners, the LLM agents, portfolio doctor, AutoPilot, F&O, paper trading, live trade execution + kill-switch, broker OAuth (real Zerodha/Angel/Upstox), Razorpay billing, and the full admin console. The auth + tier-gating spine is well-designed and consistently applied.
- **What's honestly-not-ready (and says so):** the F9 earnings predictor (503), the F1 intraday model (reported `false`), and a few AI screener sub-forecasts (503 instead of a fake number). This is the no-fallback discipline working as intended — a strength to talk about, not hide.
- **What to fix before it bites you:**
  1. **Per-feature LLM caps are unenforced** (only `chat` is). Five Elite/Pro AI features rely solely on the $20 kill-switch, not per-user ceilings. Low money-risk, but a single account can hog the shared free-model quota.
  2. **SEBI RA number is `PENDING_APPROVAL`** and is served on every dashboard via `/auto-trader/compliance`. The disclaimer text is production-ready; the registration is not.
  3. **DATA_PROVIDER defaults to yfinance (Tier-3)**, not the paid Kite feed — so out-of-the-box data quality is the cheap tier unless `DATA_PROVIDER=kite` is set in env.

---

<a name="aiml-models"></a>

## AI / ML Models — The Trained Brains

This section documents every machine-learning model in the codebase: what it is, what it predicts, how it works, whether it is genuinely in production, which user-facing feature consumes it, its ensemble weight, and — most importantly for a founder — an honest read on whether it actually has an edge. The headline is simple: the *plumbing* is real and disciplined (no fake "AI" stand-ins), but the *measured predictive edge is weak across the board*, and there are two material gaps between what the registry calls "production" and what the running code actually loads.

### How the models fit together (the "swing signal" brain)

When the platform generates a daily swing-trade signal, it does **not** rely on one model. It runs five models on every candidate stock and only emits a signal when at least 3 of the 5 "vote" BUY, the blended confidence clears a threshold, and the forecasted reward-to-risk is good enough. This is the ensemble, and the code that orchestrates it lives in `backend/ai/signals/generator.py:260`. The blend weights are hard-locked in `backend/ai/signals/voters.py:22`.

| Public name | Real model | Type | Predicts | Ensemble weight | PROD in registry? | Loaded at runtime? | Honest edge |
|---|---|---|---|---|---|---|---|
| **Alpha** | Qlib Alpha158 (LightGBM ranker) | Gradient-boosted trees on 158 factors | Cross-sectional rank: which NSE stocks are relatively strongest | 0.20 | Yes — v8, `is_prod=true` | Yes (from B2 cache) | Real but weak. IC ≈ 0.031, ICIR 0.36. `qlib_realmoney_pass: false` |
| **Forecast** | TFT (Temporal Fusion Transformer) | Deep-learning time-series transformer | 5-day price path, p10/p50/p90 quantiles | 0.30 | Yes — v3, `is_prod=true` | **Mismatch** — see caveat | Claimed 68% directional accuracy; served model differs from registry model |
| **Gate** | LightGBM signal gate | 3-class GB classifier (BUY/HOLD/SELL) | Short-term direction of a single stock | 0.30 | **No — v1 is `is_prod=false`** | Yes (disk fallback) | Out-of-sample Sharpe ≈ 0.10, accuracy ≈ 40%. Effectively coin-flip |
| **Regime** | Gaussian HMM (3-state) | Hidden Markov Model | Market state: bull / sideways / bear | 0.10 | Yes — v24, `is_prod=true` | Yes | Reasonable as a context filter; not a return predictor |
| **Mood** | Sentiment classifier | **LLM at runtime** (FinBERT only as fallback) | News tone per stock (−1 to +1) | 0.10 | finbert_india v1 `is_prod=true` (registry) | LLM via OpenRouter by default | Works; but the "PROD model" in the registry is not what runs by default |
| **PatternScope** | breakout_meta_labeler | RandomForest (500 trees, depth 3) | Will a detected chart breakout be profitable? | Not in signal path | Used as Scanner-Lab tag only | Yes (Scanner only) | In-sample ~49%, out-of-sample 35–40% with costs |
| AutoPilot | *(not a model)* | Executor / orchestrator | — | — | — | — | Consumes Alpha + Regime; see note below |

---

### Alpha — Qlib Alpha158 (the stock ranker)

**What it is.** A LightGBM gradient-boosted-tree model trained through Microsoft's real `pyqlib` library on its standard "Alpha158" factor set — 158 hand-engineered price/volume features per stock. Code: `backend/ai/qlib/engine.py`. This is the genuine Qlib pipeline (`qlib.contrib.data.handler.Alpha158`, `qlib.contrib.model.gbdt.LGBModel`), not a home-grown imitation.

**What it does / who it's for.** Every night it scores the entire liquid NSE universe and produces a *relative* ranking — "of all stocks today, which look strongest." Rank #1 → score 1.0, last → 0.0 (`engine.py:333`). It feeds both the swing-signal ensemble (weight 0.20) and the Elite-tier AutoPilot, which buys the top-N ranked names.

**PROD status (verified in Supabase `model_versions`).** v8 is the live prod row (trained 2026-06-04), trained on 480 symbols, 152,575 out-of-sample rows.

**Honest caveats.** This is the most important honesty line in the whole product. The model's own metrics say it is **not good enough to trade real money on**: `rank_ic_mean = 0.0314`, `rank_icir = 0.36`, and the model's stored verdict is literally `qlib_realmoney_pass: false` with the reason *"signal not consistent enough for real-money use. Safe for shadow/display; not safe for autonomous trading."* An IC of 0.03 is a very faint edge (a strong quant signal is 0.05–0.10+). It clears the soft "quality" gate but fails the real-money gate. The earlier narrow-universe versions (v1, v2) had *negative* IC.

---

### Forecast — TFT (the price path predictor)

**What it is.** A Temporal Fusion Transformer — a deep-learning model for time series that outputs not just a single prediction but a *range*: a pessimistic (p10), median (p50), and optimistic (p90) forecast for the next 5 days. Runtime adapter: `TFTPredictor` in `backend/ai/model_registry.py:182`.

**What it does.** It is the model that sets the actual trade levels. The signal generator derives entry, stop-loss, and three targets directly from the TFT quantiles plus an ATR safety band (`generator.py:534`). The stop is the *wider* of TFT-p10 and a 2-ATR floor; target-1 is the *tighter* of TFT-p90 and a 4-ATR ceiling. So even when the rest of the ensemble agrees, TFT decides where you get in and out. Weight in the vote: 0.30 (tied for highest).

**Honest caveats — this is a real disconnect the founder should know about.**
- The registry says PROD tft_swing **v3** with `directional_accuracy: 0.68`, trained with a **NeuralForecast** backend (`max_encoder_length: 60`). The cached artifact is `tft_swing_nf.tar.gz`.
- But the **running code loads a different file and a different framework**: `TFTPredictor` calls `pytorch_forecasting.TemporalFusionTransformer.load_from_checkpoint()` on the on-disk `tft_model.ckpt` (a *pytorch_forecasting* artifact with `max_encoder_length: 120`, per `ml/models/tft_config.json`). There is no `neuralforecast` import anywhere under `backend/ai/`.
- **Net:** the served TFT is the legacy disk checkpoint, not the registry's "PROD v3." The 0.68 directional-accuracy number belongs to a model the live system isn't loading. The number you'd quote and the model you'd ship are not the same artifact. (0.68 directional accuracy is itself only "claimed" — it's the trainer's self-reported metric, not an independently audited result, and directional accuracy is a generous metric.)

---

### Gate — LightGBM Signal Gate (the per-stock direction filter)

**What it is.** A 3-class LightGBM classifier that labels a single stock BUY / HOLD / SELL from ~15–30 technical features. Code: `LGBMGate` in `backend/ai/model_registry.py:37`. It has solid engineering hygiene — it refuses to load if its feature schema doesn't match the artifact, preventing silent "garbage prediction" drift (`model_registry.py:80`).

**What it does.** It is the joint-highest-weighted voter (0.30). Its BUY probability flows straight into the ensemble score, and its verdict counts as one of the agreement votes.

**Honest caveats — two real issues.**
1. **It is not marked production in the registry.** The only row is `lgbm_signal_gate v1` with `is_prod=false`. Yet `SignalGenerator.__init__` *requires* it and loads it from the disk file `ml/models/lgbm_signal_gate.txt`. So a model that the registry does not bless as PROD is nonetheless a mandatory, top-weighted voter in every live signal. The meta sidecar even flags itself as "legacy, pre-PR-169."
2. **Its measured edge is essentially nil.** Registry metrics: cross-validated `sharpe_mean ≈ 0.097`, `accuracy_mean ≈ 0.40`, win-rate ≈ 0.42, with wildly unstable per-fold Sharpes (+5.6, −7.9, −2.7, +4.2, +1.2). For a 3-class problem, ~40% accuracy is barely above chance.

---

### Regime — Gaussian HMM (the market-weather model)

**What it is.** A 3-state Gaussian Hidden Markov Model trained on Nifty returns + India VIX + realized volatility, classifying the market as bull (0) / sideways (1) / bear (2). Code: `ml/regime_detector.py`. States are sorted by mean return so the labels are stable across retrains (`regime_detector.py:111`).

**What it does.** It is a *context gate and position sizer*, not a stock picker. In the ensemble it contributes weight 0.10, and in a bear regime it multiplies the final confidence by 0.6 (`generator.py:79`). In AutoPilot it scales gross exposure (bull 1.0, sideways 0.7, bear 0.3). This is a sensible, well-accepted use of an HMM.

**PROD status.** v24, `is_prod=true` (trained 2026-06-04), 4,011 observations across 5 CV folds.

**Honest caveats.** The metric tracked is log-likelihood, which measures *fit*, not *profitability* — there's no evidence the regime labels improve trade returns, only that the HMM describes the data. There is a hidden hazard: `predict_regime` has a `_default_regime()` that silently returns "bull" on error (`regime_detector.py:201`). The signal generator deliberately refuses to ship signals if regime can't be computed (`generator.py:296`), which is good, but the screener path is less strict.

---

### Mood — Sentiment (the news-tone reader) — and the FinBERT reality

**What it is, honestly.** Despite the registry listing **finbert_india v1** as `is_prod=true` and the generator's comments proudly logging *"FinBERT-India loaded (PROD voter)"*, the code that actually runs **defaults to a general-purpose LLM zero-shot classifier over the OpenRouter gateway**, not FinBERT. See `backend/ai/sentiment/engine.py:52` (`_select_classifier`): FinBERT is used *only* when `USE_FINBERT_FALLBACK=1` is set, or as a last resort if the LLM fails to initialize. The code's own comments justify this (the FinDPO paper shows FinBERT turns negative once trading costs are added; "Vansh180/FinBERT-India-v1 is a 7K-sample hobby model").

**What it does.** Nightly, it pulls ~2 days of news per stock, scores each headline, averages to a −1..+1 sentiment, and contributes weight 0.10 to the ensemble. When a stock has no recent news, it scores a neutral 0.5 — which is the model's honest output for empty input, not a fabricated value.

**Honest caveats.** The "FinBERT-India" branding (in the registry row, the voter name `finbert_india`, and the log lines) is misleading: the production behavior is "an LLM reads headlines," not "a trained finance-specific BERT." The registry row exists but doesn't reflect runtime. This is the single biggest naming-vs-reality gap after the TFT framework mismatch.

---

### PatternScope — BreakoutMetaLabeler (Scanner-Lab only)

**What it is.** A RandomForest (500 trees, max depth 3) that scores a *detected* chart breakout for the probability it'll be profitable. Code: `BreakoutMetaLabeler` in `ml/features/patterns.py:1918`.

**What it does / where.** It is explicitly **not** in the alpha/signal path — the signal generator deletes its loader (`generator.py:135`). It runs only inside the Scanner Lab, where it tags detected patterns with a confidence score above a 0.35 floor (`backend/data/screener/engine.py:364`, `services/chart_patterns/scanner.py:97`).

**Honest caveats.** Per the project's own memory, the once-claimed "63.6% win rate" was retired; reality is ~49% in-sample and 35–40% out-of-sample once costs are included. It is correctly walled off from real trade generation and presented as a Scanner-only confidence tag.

---

### AutoPilot is an executor, not a model

`AutoPilotService` (`backend/trading/autopilot_service.py`) is frequently spoken of as if it were "the AI," but it is **orchestration code, not a trained model**. Daily at 15:45 IST it: resolves the HMM regime, takes the Qlib top-N ranks, converts ranks to Kelly-decayed weights, applies the regime multiplier and a VIX overlay, and caps per-stock exposure. Every "intelligent" decision it makes is borrowed from Alpha (Qlib) and Regime (HMM). There is no AutoPilot model file. (Worth noting: the file's own header still references "regime_hmm v20+" while live PROD is v24 — stale comment, not a functional bug.)

---

### What's genuinely good (don't lose this)

- **No-fallback discipline is real and enforced.** `SignalGenerator.__init__` hard-raises if any required model artifact is missing rather than degrading to heuristics (`generator.py:123`). The earnings predictor raises `ModelNotReadyError` instead of faking output (`ai/earnings/predictor.py`). Trainers write honest self-grading verdicts (`qlib_realmoney_pass`, `*_quality_pass`) into the registry, and the registry shows a graveyard of *failed* models that were correctly retired (RL agents at Sharpe −0.16 to −11.8, momentum_timesfm/chronos at ~49% directional accuracy, vix_tft, chronos2_macro). The team is grading its own homework strictly.
- **A real model registry exists** (`backend/ai/registry/`): B2 object store for bytes + Postgres `model_versions` for the prod/shadow/metrics truth, with atomic promote/demote.

### What a founder must fix or disclose

1. **TFT framework mismatch:** the served model (disk pytorch_forecasting ckpt, encoder 120) is not the registry "PROD v3" (NeuralForecast, encoder 60, 0.68 dir-acc). Pick one and make them match before quoting the 68% number.
2. **LGBM Gate is a required top-weight voter but `is_prod=false`** and self-described as legacy — and its edge is ~coin-flip.
3. **"FinBERT" is an LLM at runtime.** The registry row and the `finbert_india` voter name don't reflect what runs.
4. **No proven live alpha.** The strongest model (Qlib) self-certifies `realmoney_pass: false` at IC 0.03. The honest framing is "decision-support and ranking," not "profitable autonomous trading."
5. **F9 earnings predictor model is unbuilt** (registry `earnings_xgb` v1 = `skipped, insufficient_supabase_data`); the feature correctly serves a 503/"coming soon" rather than faking it.

---

<a name="ai-agents"></a>

## AI Agents — The LLM Reasoning Layer

This section documents the part of Quant X that "thinks in words": the large-language-model (LLM) agents that explain signals, debate trades, audit portfolios, draft strategies, and answer chat. This is **separate** from the 5 statistical/ML models that actually generate swing signals (Alpha/Forecast/Gate/Regime/Mood). The agents here are a **reasoning and narration layer on top** of that data — they explain and structure, they do not (with one gated exception) decide your trades.

### The one-paragraph honest summary

Quant X runs a small, hand-built agent framework ("GraphRunner") that chains LLM calls into reasoning pipelines. There are **3 multi-agent graphs** (Copilot, Portfolio Doctor, Counterpoint Debate) totalling **16 distinct agent nodes**, plus **7 single-shot LLM roles** (strategy generator, scanner thesis, chart vision, F&O advisor, sentiment classifier, daily digest, weekly review). **Every one of them runs on a FREE open-source model** via the OpenRouter gateway, governed by a **hard $20/month spending kill-switch**. The agents are **advisory only** — they never place a live trade on their own. The quality ceiling is set by free models (Llama-3.3-70B, GPT-OSS-120B, Qwen3-Coder, Gemma), the spend ceiling is real and enforced, but the **per-feature usage caps are mostly defined-but-not-enforced** (only chat and the Portfolio Doctor monthly quota are actually metered).

---

### How the runtime works (GraphRunner)

Instead of using LangChain/LangGraph, the codebase ships a ~130-line custom orchestrator (`backend/ai/agents/base.py`). The design is deliberately small for a solo-founder codebase:

- An **Agent** is one unit of work that reads shared state, optionally calls a tool + the LLM, and writes its result back (`base.py:48`).
- A **GraphRunner** runs a list of agents. A plain agent = a sequential step; a list of agents = a parallel "fan-out" run concurrently via `asyncio.gather` (`base.py:101-130`).
- All agents share one `AgentState` object (inputs, scratch memory, tool-call trace, final output) — `state.py`.
- Any agent can `raise EarlyExit` to short-circuit the rest of the graph (e.g. the chat classifier rejecting an off-topic question) (`base.py:42`, `base.py:124`).

Every LLM call goes through a single adapter (`backend/ai/agents/llm.py`) that talks to **OpenRouter only** (OpenAI-compatible HTTP). The adapter handles: per-agent model selection, a free→free→paid fallback chain, real token streaming, JSON-mode output, vision (image) input, and budget enforcement. **There is no Gemini, no Anthropic, no self-hosted model** — the migration to open models is complete (confirmed by `LLM_DEFAULT_MODEL` default and the per-agent map below).

---

### The per-agent model map (what actually runs)

Each role maps to a model via the `AGENT_MODEL_MAP` env JSON (`llm.py:103-117`). The **live `.env` value** assigns every role to a `:free` model:

| Role (code key) | Model assigned (live `.env`) | Paid? |
|---|---|---|
| `classifier` | meta-llama/llama-3.3-70b-instruct **:free** | No |
| `tool_planner` | openai/gpt-oss-120b **:free** | No |
| `responder` | meta-llama/llama-3.3-70b-instruct **:free** | No |
| `doctor` | openai/gpt-oss-120b **:free** | No |
| `debate` | openai/gpt-oss-120b **:free** | No |
| `strategy_generator` | qwen/qwen3-coder **:free** | No |
| `fno` | openai/gpt-oss-120b **:free** | No |
| `scanner_thesis` | meta-llama/llama-3.3-70b-instruct **:free** | No |
| `vision` | google/gemma-4-31b-it **:free** | No |
| `sentiment` | meta-llama/llama-3.3-70b-instruct **:free** | No |

**Honest caveat:** the original "open-LLM strategy" plan called for a tiered brain — a small Qwen3-8B classifier, a big Qwen3-235B "generator where the money is made," DeepSeek for debate, etc. In practice the shipped map uses **three free models** (Llama-3.3-70B, GPT-OSS-120B, Qwen3-Coder) plus Gemma for vision. The premium 235B generator is **not** wired in. So the per-role specialization is real in structure but modest in capability — these are all good-but-free models, not frontier reasoning. (One config nit: `google/gemma-4-31b-it:free` for vision is not in the price card and one fallback entry hardcodes Qwen3-8B; neither affects spend since everything is free.)

---

### The spending kill-switch (this part is real and solid)

`backend/observability/llm_budget.py` is a month-to-date USD meter with a hard cap of **$20/month** (`config.py:249`). How it protects you:

- **Free models cost $0** in the price card (`llm_pricing.py:42-48`), so routine usage **never moves the meter and is never blocked**.
- Only a **paid** call (which would only happen if a free model rate-limited and the code spilled to a paid fallback) is gated. Once month-to-date spend ≥ $20, all paid calls degrade gracefully and the free→paid spill is disabled — free models keep running (`llm.py:81-96`, `llm.py:190-194`).
- The meter reconciles from the `llm_usage_events` table on a 60-second TTL and survives restarts; spend is persisted per call (`llm_budget.py:81-107`).

**Honest caveat:** it is single-instance-accurate. On a multi-instance deploy the cap can briefly overshoot by up to one 60s reconcile window — the code itself flags this as acceptable for a solo deployment. Net: **$20/month is a genuine, enforced ceiling**, and because every role is mapped to free models today, real spend should sit near zero.

---

### Agent 1 — Main Chat / Copilot (the 4-node graph)

- **What it is:** the conversational assistant. A linear 4-stage pipeline: **Classifier → ToolPlanner → ToolCaller → Responder** (`copilot.py:421-429`).
- **What it does:** Classifier decides if your message is finance-related (with a regex fast-path so "explain RSI" or "@TCS" skips an LLM round-trip — `copilot.py:80-90`); ToolPlanner asks the LLM which data tools to call (max 3); ToolCaller fetches **real data** from your portfolio/signals/regime/snapshot via the tool registry; Responder writes the answer in a "numbers-first, no fluff" analyst voice, citing the tool data (`copilot.py:224-232`). It supports **real token streaming** and emits live chart artifacts (price sparkline, regime bars, stat pills) built strictly from real tool output (`copilot.py:289-393`).
- **Tools it can call:** portfolio, watchlist, one signal, today's signals, stock snapshot, current regime, and an options-strategy suggester (`tools.py:145-316`).
- **Who/tier + caps:** all tiers. **Chat is the one feature whose cap is actually enforced** — Free 5/day, Pro 50/day, Elite 200/day, via `AssistantCreditLimiter` (`ai_routes.py:88-120`, `tier_gate.py:182`). Admins bypass.
- **Honest caveats:** runs on **free Llama-3.3-70B**. It is educational/advisory only — the system prompt explicitly forbids it from being "an execution recommendation." The credit limiter **fails open** (if the DB is down, your chat isn't blocked — the $20 switch is the backstop). Tool selection is LLM-planned, so it can occasionally pick the wrong tool or none.

---

### Agent 2 — Portfolio Doctor (4-specialist chain-of-thought)

- **What it is:** a "FinRobot"-style audit graph: **4 specialists run in parallel → 1 synthesizer** (5 nodes total) — `finrobot.py:269-282`.
- **The 4 specialists:** Fundamental (grades ROE/debt/earnings/valuation), Management Tone (reads concall/headlines for hedging + guidance), Promoter Holding (flags falling promoter stake / rising pledges), Peer Comparison (sector percentile rank). The **Synthesizer** blends them into one decision-ready verdict + an Action line (add/hold/trim/exit) and a composite 0-100 score (`finrobot.py:198-263`).
- **Who/tier + caps:** powers F7 Portfolio Doctor (and F5 AI SIP). Tier-gated; Pro gets **1 full run/month**, Elite effectively unlimited — enforced by a **hand-rolled monthly quota** in `portfolio_doctor_routes.py:135-165`, not via the central caps table.
- **Honest caveats:** runs on **free GPT-OSS-120B**. Each specialist gracefully degrades to a neutral output when its input data is missing — so a "report" can be partly hollow if, e.g., no concall transcript or peer metrics were supplied. The composite score uses a **fixed neutral base (0.25 weight) for promoter** rather than the agent's own promoter read (`finrobot.py:241-249`) — a simplification. Recent improvement: fundamentals are now fed from real screener.in data (per project memory), which makes the Fundamental and Peer agents materially more grounded than before.

---

### Agent 3 — Counterpoint Debate (7-agent bull/bear)

- **What it is:** the most elaborate graph — **7 agents across 5 layers**: 3 Analysts (Fundamentals/Technical/Sentiment, parallel) → Manager (distills a briefing) → Bull + Bear Researchers (parallel, debate the brief) → Risk Manager (sizes 0.0–1.0 using VIX + regime) → Trader (final enter/skip/half/wait verdict) — `tradingagents.py:307-325`.
- **What it does:** produces a structured, two-sided argument transcript plus a final decision and a position-size multiplier, persisted to `signal_debates` and surfaced on the signal-detail "Debate" tab (`ai_routes.py:540-588`).
- **Who/tier + caps:** **Elite only**, gated by `RequireFeature("debate")` (HTTP 402 for non-Elite) — `ai_routes.py:544`. A cap of 10/day is *defined* in the caps table but **not enforced** (see below).
- **Honest caveats:** runs on **free GPT-OSS-120B**. Despite the "Risk Manager → size multiplier" and "Trader → enter/skip" framing, this output is **advisory** — it does not place or block a live trade; it's a decision-support transcript. Quality of the analyst layer depends entirely on what fundamentals/news/snapshot the caller passes in (all optional). It is the single most LLM-call-heavy feature (7 sequential+parallel model calls), so it's the most likely to be slow or to hit free-model rate limits.

---

### The single-shot LLM roles (not multi-agent, but still "AI agents")

These are one-call LLM uses, each with its own role/model:

| Role | Surface / what it does | Backed by | Tier | Cap defined | Cap enforced? |
|---|---|---|---|---|---|
| **Strategy generator (NL→DSL)** | Studio: turns plain-English strategy ideas into validated DSL JSON | Qwen3-Coder (free) — `studio.py:185-258` | Pro/Elite | strategy_gen 1/10/30 | **No** |
| **Scanner thesis** | Screener/scanner: writes a short "why this matched" thesis per stock | Llama-3.3-70B (free) — `screener_v2/enrich.py:280-315`, `chart_patterns/explain.py:254-281` | Pro/Elite | scanner_thesis 0/30/100 | **No** |
| **Chart vision** | Stock & signal pages: sends a rendered chart image to a vision LLM for a read | Gemma (free), `complete_vision` — `ai/vision/analyzer.py:88-106` | Pro/Elite | chart_vision 0/20/60 | **No** |
| **F&O advisor** | F&O panel: recommends ONE options structure from a fixed template registry given your view + VIX + regime | GPT-OSS-120B (free) — `fo_strategies_routes.py:843-857` | Elite | fno_advisor 0/0/20 | **No** (feature-gated only) |
| **Sentiment classifier ("Mood")** | Replaces FinBERT at runtime — batched headline → bullish/neutral/bearish JSON | Llama-3.3-70B (free), `complete_sync(role="sentiment")` — `sentiment/llm_classifier.py:191-198` | platform-internal | n/a | n/a |
| **Daily digest / Morning brief** | Push/email intro line for the daily brief | free model — `digest/generator.py:247-275` | Pro+ | n/a | n/a |
| **Weekly review** | Narrative of your trading week | free model — `weekly_review/generator.py:193-221` | Pro+ | n/a | n/a |

**Important note on "Mood":** the user-facing sentiment engine is, by default, **an LLM at runtime, not the FinBERT-India model**. The code deliberately demoted FinBERT (citing the FinDPO paper that LLMs match/beat FinBERT) and routes sentiment through the free Llama model (`llm_classifier.py:4-17`). So "FinBERT" branding/intent is no longer what runs in production sentiment.

---

### Counting the agents

- **Multi-agent graphs:** 3 (Copilot, Doctor, Debate)
- **Distinct graph nodes:** 16 — Copilot 4 (classifier, tool_planner, tool_caller, responder) + Doctor 5 (fundamental, management_tone, promoter_holding, peer_comparison, synthesizer) + Debate 7 (3 analysts, manager, bull, bear, risk_manager, trader)
- **Single-shot LLM roles:** 7 (strategy generator, scanner thesis, chart vision, F&O advisor, sentiment, daily digest, weekly review)
- **Distinct model "roles" in the model map:** 10 keys
- **Models actually used:** ~4 free models (Llama-3.3-70B, GPT-OSS-120B, Qwen3-Coder, Gemma)

---

### The big honest caveats (read these)

1. **Everything runs on free models.** No frontier model is in production. The planned premium "235B generator" is not wired in. Output quality is "competent open-source," not best-in-class.
2. **Per-feature caps are mostly theatre.** `LLM_FEATURE_CAPS` defines daily/monthly limits for chat, strategy_gen, scanner_thesis, chart_vision, debate, fno_advisor, and portfolio_doctor (`tiers.py:129-137`). Code-grep confirms **only two are actually enforced**: `chat` (via the copilot credit limiter) and `portfolio_doctor` (via a separate hand-rolled monthly quota). The rest — strategy generation, scanner thesis, chart vision, F&O advisor, debate — are protected **only** by tier feature-gating (you have the tier or you don't) and the global $20 kill-switch. There is no per-user daily metering on those, so within your tier they're effectively unlimited until the shared free quota or $20 cap is hit.
3. **Agents are advisory, not autonomous.** Across Copilot, Debate, and the F&O advisor the code is explicit: LLMs return suggestions; the user must click Deploy; LLMs never auto-place live trades. The single exception is the **enforced backtest/Sharpe gate** — an LLM-generated strategy can only go live after passing out-of-sample evaluation (project memory: "Strategy GATE built"). That gate, not the LLM, is what protects real money.
4. **Embedded composers now hand off to Main Chat.** The inline agent boxes on feature pages are launchpads, not full chats — they push the follow-up to the single standalone Main Chat orchestrator (`frontend/components/copilot/EmbeddedAgent.tsx:54-60`). So the "embedded domain agent per page" idea is, in shipped reality, a redirect into one central chat.
5. **The $20 ceiling is the most trustworthy guarantee here.** It's enforced, persisted, free-model-aware, and self-correcting. The weakest links are model quality (free-tier) and the unenforced per-feature caps.

---

<a name="automation-data-billing--security"></a>

## Automation, Data & Cross-Cutting Machinery

This section documents the "engine room" of Quant X — the background jobs, market-data plumbing, notifications, alerts, billing, and security that run the product whether or not a user is logged in. None of this is what a user clicks; all of it is what keeps the clickable parts fed, safe, and paid-for. Where something is rule-based, single-instance, or pending external approval, it is flagged plainly.

---

### 1. Scheduler / Background Jobs

**What it is.** One always-on Python scheduler (`APScheduler`, India timezone) that fires ~40 timed jobs across the trading day. It is created and wired up in `backend/platform/scheduler.py:53` (`SchedulerService`) and toggled by the `ENABLE_SCHEDULER` env flag (default on, `config.py:207`). Think of it as the autopilot crew: it generates signals, rebalances portfolios, refreshes data and models, sends digests, and enforces risk — on a clock, every weekday.

**How it helps.** The product can claim "AI runs your watch" because these jobs do the work between market open and close without anyone pressing a button. Every job also writes a row to a `scheduler_job_runs` telemetry table (`scheduler.py:652`) so the admin can see what ran, how long it took, and what failed.

#### Core market-day jobs (Mon–Fri, IST)

| Time (IST) | Job | What it does | Backed by |
|---|---|---|---|
| 06:05 | Kite admin token refresh | Headless TOTP login to Zerodha to renew the daily data token | Rule-based HTTP automation (`kite.py:947`) |
| 06:10 | User broker token refresh | Re-logins each connected Zerodha/Angel account, re-encrypts tokens | Rule-based (`scheduler.py:836`) |
| 06:15 | Subscription lifecycle check | Expires past-due paid subs, sends 3-day renewal reminders | Rule-based |
| 08:15 | Update market regime | Runs the **HMM regime model** on Nifty + VIX, writes `regime_history`, emits a live `REGIME_CHANGE` event | Real ML (Gaussian HMM) |
| 08:30 | Pre-market scan + broadcast | Resets daily kill switches, broadcasts today's active signals | Rule-based dispatch |
| 08:45 | Create trades from signals | Turns signals into `pending`/`approved` trade rows per user's auto-trade mode + risk profile | Rule-based |
| 09:15 | Market-open check | Gap + VIX checks; warns on >2% gap or VIX >25 | Rule-based thresholds |
| 09:30 | Execute pending trades | Places approved/full-auto trades (gated by kill switches) | Rule-based + broker API |
| 09:30 | Options signal scan | Scans option chains for active F&O deployments | Rule-based |
| every 1 min | Price updates | Refreshes LTP + unrealized P&L on open positions | Data layer |
| every 5 min | Position monitor | SL/target/trailing-SL checks, signal lifecycle, daily-loss kill-switch enforcement | Rule-based |
| every 5 min | Watchlist price alerts | Fires `price_alert` when LTP crosses a user threshold (debounced) | Rule-based |
| every 5 min | Intraday LSTM inference (F1) | Scores ~10 large-caps with a Bi-LSTM ONNX model, emits intraday signals | Real ML *if the model artifact exists* (skips with a warning otherwise) |
| every 5 min | Strategy runner position sweep | Real-time exits the moment price hits a stop/target on live `strategy_positions` | Rule-based |
| every 2 min | Live trade reconciler | Polls the broker for fills so "pending" doesn't stick forever | Rule-based |
| every 15 min | Options position monitor | Exit conditions on open options positions | Rule-based |
| 15:30 | Market-close processing | Expires stale signals (>5 trading days), runs signal lifecycle, sends EOD summaries | Rule-based |
| 15:30 | Strategy fan-out runner | Evaluates every live user-strategy across its symbols → signals + positions | Rule-based DSL |
| 15:35 | Broker reconciliation | Syncs DB positions vs actual broker positions (catches external exits) | Rule-based |
| 15:40 | Qlib Alpha158 nightly rank | Cross-sectional alpha ranking, writes `alpha_scores` | Real ML (Qlib) — see caveat |
| 15:45 | EOD scanner → signals | Generates tomorrow's swing signals (the main signal job) | Real ML stack |
| 15:50 | Chronos nightly forecast | Zero-shot price forecast over Nifty 500, writes `forecast_scores` | Real ML *if Chronos installed* |
| 15:50 | AutoPilot daily rebalance (F4) | Elite portfolio rebalance (Qlib ranker + HMM sizing + VIX overlay), idempotent via cron-lock | Real ML, supervised |
| 16:00 | Daily reports | Per-user P&L, win-rate, portfolio history; model performance | Rule-based aggregation |
| 16:15 | AI Stock Ranker picks | Top-15 daily picks → `quantai_picks` | Real ML ranker *if loaded* |
| 16:30 | AutoPilot track record | 30/60/90-day paper+live snapshots for the dashboard | Rule-based |
| 16:30 | FinBERT-India sentiment refresh | News sentiment for Nifty 500 → `news_sentiment` | Real ML *if FinBERT installed* — see caveat |
| 16:45 | Drawdown alerts | Fires `portfolio_drawdown` at −5/−10/−15% with regime context | Rule-based |
| 16:45 | Model drift daily check | Compares prod/shadow models vs rolling perf, can demote on hard drift | Rule-based + ML metrics |
| 17:00 | Earnings predictor scan (F9) | Next-14-day earnings calendar + surprise prediction | **Rule-based** (XGBoost not trained yet) |
| 17:30 | FII/DII daily catch-up | Appends today's FII/DII flows from NSE live API to a parquet cache | Rule-based scrape |
| 17:30 | Evening digest | Post-close summary via Telegram/WhatsApp | Template + optional LLM intro |
| 22:00 | Nightly model refresh | CPU retrain of `regime_hmm` + `qlib_alpha158`, idempotent (18h lookback) | Real ML training |

#### Non-daily / weekly / monthly jobs

| Schedule (IST) | Job | What it does |
|---|---|---|
| Mon 06:30 | Momentum weekly email (F3) | Top-10 momentum picks to Pro+ |
| Mon–Fri 07:30 | Morning digest (F12) | Pre-market brief via Telegram (free) / WhatsApp (Pro) |
| Tue–Sat 07:30 | Strategy discovery digest | Emails overnight discovered strategies to Elite + admin |
| Sun 08:00 | Weekly portfolio review (N10) | LLM-written review for every Pro+ user (rule-based fallback) |
| Sun 02:00 | Model rolling-performance aggregator | Aggregates closed-signal outcomes (powers public /models page) |
| Sun 03:00 | HMM weekly retrain | Re-fits regime model on 10y data; does **not** auto-promote |
| Sat 06:00 | Weekend model check | 30-day accuracy review + CPU retrain |
| Mon–Fri 22:30 | Nightly strategy discovery | Genetic-algorithm + walk-forward strategy search |
| daily 23:00 | Paper portfolio snapshot | Equity curve + league leaderboard data (F11) |
| daily 03:30 | Referral expiry | Expires referrals pending >90 days |
| Last Sun 00:00 | AI SIP monthly rebalance (F5) | Black-Litterman portfolio proposal for Elite |
| 1st of month 02:00 | EarningsScout retrain (F9) | XGBoost retrain *only when enough labeled rows exist* |
| 4 windowed jobs | AutoPilot supervisor (PR-M) | Continuous oversight wrapping the 15:50 rebalance (no LLM) |

**Reliability features that are real.**
- **Retry-on-failure** for the three most critical jobs (regime, pre-market scan, EOD scan): `_run_with_retry` reschedules one retry 5 minutes later, up to 2 attempts (`scheduler.py:705`). A transient data hiccup doesn't dark the whole day.
- **Cron idempotency lock** (`cron_lock.py`): the AutoPilot rebalance, track-record, and drawdown jobs take a per-day `UNIQUE(job_id, run_date)` row in `system_cron_runs`. If the scheduler fires twice (restart, auto-scale), the second firing acquires nothing and exits — so **nobody's portfolio gets placed twice at the broker**. This is genuinely implemented, not aspirational.
- **Telemetry + Sentry/PostHog escalation** on any `failed` job.

**Honest caveats.**
- **Single-instance assumption.** APScheduler runs in one process. If the backend is deployed to more than one worker/instance, the timers fire in *each* process. Most money-touching jobs are protected by `cron_lock`, but the every-1-min/every-5-min monitors and several digest jobs are **not** lock-guarded — they'd duplicate work (and possibly duplicate notifications) on a multi-instance deploy. This is acceptable for a solo-founder single-instance deployment and is the documented assumption, but it is a real ceiling on horizontal scaling.
- **Several "AI" jobs degrade to skip or rule-based.** The intraday LSTM, Chronos forecast, FinBERT sentiment, and AI Stock Ranker jobs all start with a `model_not_ready` / `not loaded` guard and silently skip if the trained artifact or heavy dependency isn't present. That's the no-fallback discipline working as designed, but it means on a fresh/under-provisioned environment these jobs do nothing while still showing green ("skipped") in telemetry.
- **Earnings predictor (F9) is rule-based.** The 17:00 scan runs a rule-based surprise predictor; the XGBoost model is not trained. The calendar and predictions surface works; the "ML" label on it does not yet.
- **Kite token refresh is brittle by nature.** The 06:05 headless TOTP login screen-scrapes Zerodha's login flow (`kite.py:947`). If Zerodha changes their login HTML/flow, auto-refresh fails and an admin must refresh manually that day — the code anticipates this and sends an admin alert, but data quality silently drops to the fallback provider until fixed.

---

### 2. Data Layer

**What it is.** A pluggable market-data provider chosen by the `DATA_PROVIDER` env var, with a three-tier fallback chain underneath. The selection happens in `data/market.py:116` (`MarketDataProvider._get_kite_provider`), which all jobs and routes call through `get_market_data_provider()`.

**The tier chain (per the data-sourcing memo).**

| Tier | Source | When used | Notes |
|---|---|---|---|
| 1 | **Kite Connect** (admin Zerodha account) | `DATA_PROVIDER=kite` | Real-time quotes, historical OHLCV, live NFO option chains with computed Greeks. Rate-limited to 180/min (`kite.py:65`), instrument cache refreshed daily. |
| 2 | **jugaad-data** (free NSE bhavcopy) | Automatic secondary *inside* the Kite provider when the Kite token is expired/invalid (`kite.py:443`) | EOD data only, no intraday. |
| 3 | **yfinance** | `DATA_PROVIDER=free` (**the current default**, `config.py:77`) | Free, no key, but delayed/approximate Indian data, US-style tickers (`.NS`), and aggressive rate-limiting. |

**How it helps.** The same code path serves quotes whether you have a paid Kite subscription or not, so the product runs end-to-end on free data and "upgrades" its accuracy by flipping one env var once the ₹2k/mo Kite admin account is live.

**Specialized sources (all NSE scrapes, in `data/screener/nse_data.py`):**
- **Delivery %** (scanner 34), **Bulk deals** (scanner 35) — from NSE security-wise + bulk CSV endpoints.
- **F&O OI spurts** (scanners for OI spike / long unwinding) — jugaad-data F&O bhavcopy for EOD OI deltas, Kite for live NFO OI when available.
- **FII/DII flows** — the 17:30 job calls NSE's live API (`ml/data/fii_dii_history`) and appends to a forward-cumulative parquet. Per the data memo, this is described as the *only* working free FII/DII source after Moneycontrol's login wall and the NSE archive block.
- **Fundamentals** — `data/fundamentals/screener_in.py` scrapes screener.in company pages (P/E, ROE, ROCE, growth ranges, pros/cons, promoter holding) to feed the Portfolio Doctor. This is the recent fix that stopped the Doctor from grading companies off empty JSON.

**Honest caveats.**
- **The default is the weakest tier.** `DATA_PROVIDER` defaults to `free` (yfinance). Out of the box, every quote, signal, and backtest runs on delayed/approximate data. Real production accuracy depends on the Kite admin account being funded and `DATA_PROVIDER=kite` set — and even then a daily token expiry can silently drop it back toward Tier 2/3.
- **Everything below Tier 1 is an unauthenticated scrape.** jugaad-data, the NSE FII/DII/delivery/bulk endpoints, and screener.in have no API contract. They rate-limit, change HTML, and block IPs. The code defends with backoff, locks, and short failure-TTL caches (`screener_in.py:35`, yfinance circuit-breaker `yfinance.py:158`), but these sources can and do go dark, and when they do the dependent scanners/Doctor return honest-empty rather than wrong numbers.
- **Option Greeks are computed, not vendored.** Kite doesn't return Greeks, so IV is solved via Newton-Raphson and Greeks via Black-Scholes in-process (`kite.py:806`). When live chain data is unavailable the whole chain is *synthetic* (Black-Scholes with VIX-derived IV). Useful for UI, but it is a model output, not market truth.
- **`_fetch_market_data` has a hardcoded simulated fallback** (`scheduler.py:2321`): if the live overview fetch fails, the 09:15 check uses fixed Nifty ≈21,850 / VIX 14.5 numbers. This is a small surface but it means the market-open gap/VIX logic can run on stale dummy values during a data outage.

---

### 3. Notifications

**What it is.** A fan-out layer that pushes events to four channels: **Web Push (VAPID)**, **Telegram**, **WhatsApp**, and **Email**.

| Channel | How | Tier | Status |
|---|---|---|---|
| Web Push | VAPID via `pywebpush` (`push.py:22`) | All | Works if `VAPID_*` keys set; handles expired-subscription 410 cleanup |
| Email | Resend API, branded HTML templates for signals/SL-target/daily summary (`push.py:90`) | Varies by event | Works if `RESEND_API_KEY` set |
| Telegram | Bot + webhook, deep-link onboarding | **Free** (`telegram_digest` = Free in matrix) | Wired; needs `TELEGRAM_BOT_TOKEN` |
| WhatsApp | Gupshup (primary) or Meta Cloud API (`whatsapp.py`) | **Pro** (`whatsapp_digest` = Pro) | **Dormant** — see caveat |

**How it helps.** Users get signals, SL/target hits, regime changes, and the morning/evening digests where they actually are (phone, browser, inbox) instead of having to keep the app open. Telegram-free / WhatsApp-Pro is a deliberate conversion lever.

**Honest caveats.**
- **WhatsApp is built but not live.** `is_configured()` returns False until Gupshup/Meta business verification is done (`whatsapp.py:48`), and every send path is a safe no-op until then. The phone-capture, OTP, opt-in, and scheduler wiring all exist; the actual delivery does not until the BSP approval lands. So "WhatsApp alerts (Pro)" is a sellable feature that currently sends nothing.
- **Each channel silently no-ops if its key/secret is missing.** None of these will crash, but none will deliver either — there is no startup gate that says "you sold push but VAPID is unset."
- **Single-instance digest jobs aren't lock-guarded** (see §1) — on a multi-instance deploy a user could get duplicate morning/evening digests.

---

### 4. Alerts (Alerts Studio)

**What it is.** A per-user **event × channel matrix** at `/api/alerts/*` (`alerts_routes.py`), gated to **Pro** (`RequireFeature("alert_studio")`). Users toggle, for each of ~18 event types, which of the 4 channels fire.

**Events covered** include: new signal, signal triggered, target/SL hit, regime change, Counterpoint debate ready, earnings ahead, weekly review, auto-trade fired, price alert, six F&O-specific events (max-pain shift, OI spike, unprotected position, adjustment recommended, VIX regime change, PCR extreme), portfolio drawdown, and an admin "cron failed" event.

**How it helps.** Instead of a blunt notifications on/off switch, a user routes urgent things (SL hit, unprotected F&O position) to push+WhatsApp+email and noise (new signals) to just Telegram. Sensible defaults are pre-loaded (`DEFAULT_PREFS`, `alerts_routes.py:91`) and a `/test` endpoint fires a real test message per channel.

**Honest caveats.**
- The matrix is the *intent layer*. Whether an alert actually lands still depends on the underlying channel being configured (so any WhatsApp cell is currently inert; see §3).
- The whole studio is **Pro-gated**, so Free users get only the hardcoded defaults, not the matrix.
- It is wired correctly to consult `channels_for_event` before dispatch, but coverage depends on each feature emitter actually calling it — the doc itself notes future emitters "should" consult it.

---

### 5. Billing

**What it is.** Razorpay subscriptions in `api/payment_routes.py`, backing the locked **3-tier model: Free ₹0 / Pro ₹999 / Elite ₹1,999**.

**Flow.** `create-order` → Razorpay order → client pays → `verify` checks the HMAC signature (`payment_routes.py:537`) → `process_successful_payment` flips `user_profiles.tier` and sets subscription dates. A **webhook** (`/payments/webhook`) independently handles `payment.captured`, `payment.failed`, and refunds, verified against `RAZORPAY_WEBHOOK_SECRET`. Both paths are **idempotent** (upsert on unique order ID; "already completed" short-circuit).

**Tier gating mechanics (the real enforcement spine):**
- **`FEATURE_MATRIX`** in `core/tiers.py:53` maps ~50 feature keys → minimum tier (e.g. `auto_trader`→Elite, `signal_unlimited`→Pro, `paper_trading`→Free).
- **`RequireFeature("x")` / `RequireTier(Tier.ELITE)`** FastAPI dependencies (`middleware/tier_gate.py`) resolve the user's tier (60s in-memory cache), and on failure raise a structured **402** that the frontend turns into an upgrade modal. **Admins always bypass.**
- Tier changes fan out: cache bust + PostHog event + a live WebSocket `tier_upgraded` so the UI unlocks features without a reload (`payment_routes.py:382`).
- **Referral credits** (N12) are consumed on payment to extend `subscription_end`.

**How it helps.** It's a working, signature-verified, idempotent payment + entitlement system. The 402-with-upgrade-modal pattern is clean, and tier resolution is centralized so frontend, middleware, and admin all read one source of truth.

**Honest caveats.**
- **No refund surface by design** (locked product rule). The webhook *processes* refunds issued from the Razorpay dashboard so the DB stays in sync, but there is no API or admin button to initiate one. Intentional, but worth knowing legally.
- **LLM per-feature caps are defined but not all enforced.** `LLM_FEATURE_CAPS` (`tiers.py:129`) specifies clean per-tier daily/monthly ceilings (chat 5/50/200, debate 0/0/10, etc.), and chat caps *are* enforced. But per the project notes, the non-chat per-endpoint caps are largely **not wired into the endpoints yet** — the real hard ceiling today is the global $20 kill-switch, not the per-feature numbers shown.
- **`core/security.py` ships a weaker legacy gate.** There are two `get_current_user` implementations: the real one in `app.py` (local HS256 verify) and an older one in `core/security.py` that does a network `supabase.auth.get_user()` and a stale `{free/starter/pro}` hierarchy. As long as routes import from `app.py`/`tier_gate.py` this is harmless, but the dead-but-importable weaker path is a footgun.

---

### 6. Security

**What it is.** Four layers: JWT auth, database RLS, kill-switches, and rate limiting.

**JWT (real and correct).** `app.py:188` decodes the Supabase JWT **locally with HS256 signature + expiry + audience verification** using `SUPABASE_JWT_SECRET`. A forged or wrong-project token is rejected (no network fallback on signature failure). Startup validation (`config.py:372`) **refuses to boot in production** if the JWT secret, broker encryption key, or default `SECRET_KEY` are unsafe. This is genuinely hardened.

**Kill-switches (two layers, both real):**
- **Global halt** — `system_flags.is_globally_halted()` (`system_flags.py:71`), checked before order placement (`scheduler.py:1160`), 15s cached, fail-*open* (a DB blip won't freeze the platform).
- **Per-user kill switch** — auto-activated when a user's daily loss limit is breached (`_enforce_daily_risk_limits`, `scheduler.py:2710`), auto-reset each morning, and re-checked at trade creation/execution.

**Rate limiting.** Per-IP + per-path sliding window (`middleware/rate_limiter.py`), with stricter composite limits on auth, broker-connect, payments, auto-trader, and Copilot endpoints; respects Cloudflare/Vercel forwarded-IP headers; returns proper 429s.

**LLM budget kill-switch.** `observability/llm_budget.py` is a month-to-date spend meter reconciled from `llm_usage_events`. Paid calls are refused once spend hits `LLM_MONTHLY_BUDGET_USD` ($20). Free-tier models cost $0 and are never blocked.

**RLS.** Per project history, RLS is enabled on the 16 user-data tables in live Supabase, with a couple of minor security-definer advisories noted.

**How it helps.** A founder can run live trading for paying users with: forged tokens rejected, a one-click ops "stop everything" switch, per-user loss circuit-breakers, brute-force-resistant auth endpoints, and a hard ceiling that makes it *impossible* to wake up to a surprise LLM bill above $20.

**Honest caveats.**
- **Rate limiting and the LLM meter are single-instance.** Both store counts in process memory. On multi-instance deploys, limits diverge per worker and the $20 cap can briefly overshoot by up to one TTL window before reconcile. Documented and acceptable for a single-instance solo deploy; a real constraint otherwise.
- **Global kill-switch fails open.** A deliberate choice (don't freeze trading on a DB hiccup), but it means a Supabase outage at the wrong moment would *not* halt order placement.
- **The LLM caps shown to users (per-feature) aren't the real enforcement** — the budget kill-switch is. See §5.
- **SEBI RA registration is pending.** `SEBI_RA_REG_NUMBER` defaults to `PENDING_APPROVAL` (`config.py:94`). The code surfaces this in disclaimers and there's a suitability-quiz gate for live AutoPilot, but operating a paid, trade-placing advisory product without the registration is, in the code's own words, "a regulatory grey zone." This is the single biggest non-engineering risk in this domain.

---

<a name="honesty-scorecard"></a>
## Honesty scorecard — whole-app real vs claimed

### ✅ Genuinely strong (real, verified, rare)
- **5 real trained models on disk/B2**, fused into one signal — not prompt wrappers.
- **No-fallback discipline enforced** — `SignalGenerator` hard-raises on a missing model;
  earnings predictor 503s rather than faking; a graveyard of correctly-retired failed
  models (RL, timesfm, chronos2, vix_tft) proves strict self-grading.
- **The Strategy Studio backtest GATE** is real and enforced (out-of-sample, walk-forward
  + holdout, `422 gate_failed` before live) — rare behind an LLM generator.
- **$20/mo LLM kill-switch** real + persisted (single-instance accurate).
- **Custom GraphRunner runtime** (3 graphs), not a LangChain demo. **Gemini 100% removed.**
- **AutoPilot is supervised-only** (RL removed) with 5%/80% caps + −10% drawdown breaker.
- **Honest empty-states** — track record hides 0%/single-trade stats; scanners/Doctor
  return honest-empty when a scrape fails; AI screener forecast endpoints 503 not fake.
- **The frontend is genuinely good** — almost every page is wired to live data.

### ⚠️ Overstated — brand vs reality
- **`/engines` says "Eight engines"; the code defines four** (`lib/engines.ts`).
- **"Mood / FinBERT-India" is an LLM at runtime** (free Llama), not the FinBERT model.
- **Measured edge is weak** and the code says so (Qlib `realmoney_pass:false`, IC ≈ 0.03;
  gate Sharpe ≈ 0.1; HMM confidence ≈ 1.0). Marketing leans on "institutional AI / 12 models."
- **Most screener routes are rule-based technical filters**, not ML — fine, but not "AI."
- **TFT registry-vs-runtime mismatch** — the 0.68 accuracy belongs to an artifact the
  system doesn't actually load (it loads a different, smaller checkpoint).
- **LGBM Gate is `is_prod=false`** in the registry yet is a required 0.30-weight voter.

### 🔧 Broken / placeholder / unbuilt / unenforced
- **`/portfolio/sip` + `/portfolio/rebalance` are skeleton placeholders** — and SIP is
  sold as Elite on the pricing matrix.
- **AutoPilot "Enable" is hardcoded OFF** (`AUTOPILOT_LIVE_TRADING=false`).
- **F9 Earnings predictor 503s** (model never trained); **F3 Momentum has no model**.
- **Per-feature LLM caps defined but not enforced** (only chat + a hand-rolled Doctor quota).
- **WhatsApp notifications fully built but DORMANT** (not business-verified → every send no-ops).
- **`/assistant` is a legacy page that redirects to `/copilot`**; `/engines` "Eight" copy stale.
- **Option Greeks/chain go synthetic** when live chain data is unavailable.

### 🚨 Business / ops / legal risk (non-code)
- **SEBI RA = `PENDING_APPROVAL`** while serving paid signals + an auto-trader. Biggest risk.
- **No proven live alpha anywhere** — the gate proves historical robustness, not profit;
  the daily signal feed bypasses the gate entirely.
- **Data defaults to Tier-3 yfinance** unless `DATA_PROVIDER=kite`; even then a daily 6am
  Kite token expiry (screen-scraped refresh) can silently degrade quality.
- **Single-instance scheduler / rate-limiter / budget meter** — duplicate jobs + cap
  overshoot on any multi-instance deploy (money jobs are `cron_lock`-guarded; monitors aren't).
- **Kill-switch fails OPEN** — a Supabase outage won't halt order placement (by design).

---
<a name="appendix--every-caveat-by-domain"></a>
## Appendix — every caveat, by domain


### User-facing features

- I read all primary page files but only headers (first 40-90 lines) of a few large pages (copilot, assistant, dashboard, scanner, watchlist, settings, track-record, trades, stock/[symbol]); deep sub-component behaviour on those is inferred from imports/comments, not line-by-line verified.
- Tier enforcement: I confirmed watchlist cap and copilot msg caps are server-driven, and admin is is_admin-gated, but I did NOT independently verify backend enforcement of the signals/day cap or per-feature LLM caps — ground truth says per-feature LLM caps are defined-but-not-enforced, which I relied on rather than re-proving.
- /portfolio/sip and /portfolio/rebalance being placeholders is verified from the full page files (skeleton-only, 'Plan 3 wires backend' comments). The pricing matrix selling SIP as Elite is verified at FeatureComparisonMatrix.tsx:77.
- AUTOPILOT_LIVE_TRADING=false is verified at autopilot/page.tsx:58; whether a backend path could still execute AutoPilot trades outside this UI flag is out of frontend scope and not checked.
- The '8 engines vs 4' finding is verified (engines/page.tsx copy vs lib/engines.ts:26). I did not check whether /engines/[slug] handles slugs beyond the 4 defined.
- F&O recommendations being rule-based is taken from the page's own header copy + ground truth, not from reading the backend recommender.
- Stale RL/FinRL-X comments in autopilot/page.tsx are code comments only; the user-visible banner copy is correct. I flagged the contradiction with MEMORY's RL-removed decision but did not trace backend AutoPilot internals.
- F9 earnings 'not trained' and marketplace-creator 'deferred' are from provided ground truth/MEMORY, cross-referenced against the pricing matrix rows — I did not verify model training state from the frontend.
- Route count: I treated /chart-test as a dev harness (not a real user feature) and grouped dynamic routes with their parents; the 'master table' is a feature map, not a 1:1 file listing.
- I did not run the app or take screenshots; all findings are from static source reading, so runtime behaviour (e.g., what an empty SIP actually renders to a logged-in Elite user) is inferred from the JSX.

### Backend API

- Route count: planning docs say 305/43; actual decorator count today is 298 across 43 modules (incl. 12 hidden back-compat aliases) — close but not exact.
- Per-feature LLM caps (strategy_gen, scanner_thesis, chart_vision, debate, fno_advisor) are DEFINED in core/tiers.py:129 but NOT enforced at their endpoints — only the 'chat' cap is enforced (ai_routes.py:97, assistant_routes.py:69). The $20 kill-switch is the only real backstop for those.
- SEBI_RA_REG_NUMBER defaults to 'PENDING_APPROVAL' (config.py:94) yet is served on every dashboard load via /auto-trader/compliance — the legal disclaimer ships with a placeholder registration.
- DATA_PROVIDER defaults to 'free'/yfinance Tier-3 (config.py:77), not the paid Kite feed, unless overridden in env.
- F9 earnings predictor model is not trained: POST /earnings/predict returns 503 model_not_ready (earnings_routes.py:249); /public/models/status reports earnings_scout and the F1 intraday model as false.
- Marketplace publish tier exists in FEATURE_MATRIX but the creator-apply + revenue-share flow is deferred out of v1 per project decision — the publish path is not a full creator surface.
- Logout (/api/auth/logout) is a client-side placeholder beacon because JWTs are stateless — expected behavior, not a stub bug.
- I verified backings by reading representative endpoints per cluster, not all ~300 individually; some lower-traffic endpoints were inferred from their service imports rather than line-by-line.
- AI screener forecast endpoints (nifty-prediction, swing-forecast) correctly 503 instead of faking, but most other screener routes are rule-based technical filters, not ML — which is correct but should not be marketed as 'AI'.

### AI/ML models

- TFT mismatch: registry PROD tft_swing v3 is a NeuralForecast model (tft_swing_nf.tar.gz, encoder=60, dir_acc 0.68) but runtime TFTPredictor (backend/ai/model_registry.py:182) loads pytorch_forecasting from disk tft_model.ckpt (encoder=120). The live model is NOT the registry PROD model; the 0.68 figure belongs to an artifact the system doesn't load.
- LGBM Gate is is_prod=false in model_versions (only v1 exists, marked legacy pre-PR-169) yet SignalGenerator REQUIRES it from disk as the joint-highest-weight (0.30) voter. Its measured edge is near-zero: CV sharpe_mean 0.097, accuracy 0.40, win-rate 0.42, wildly unstable folds.
- 'Mood/FinBERT' is an LLM at runtime by default: sentiment/engine.py:52 selects LLMFinanceClassifier (OpenRouter) unless USE_FINBERT_FALLBACK=1. The finbert_india v1 is_prod=true registry row and the 'finbert_india' voter name + generator log lines ('FinBERT-India loaded (PROD voter)') do NOT match runtime behavior.
- Qlib Alpha158 (PROD v8) measured edge is weak: rank_ic_mean 0.0314, ICIR 0.36, and the model's own metrics carry qlib_realmoney_pass=false ('not safe for autonomous trading'). It clears the soft quality gate but fails the real-money gate.
- qlib_alpha158 has no on-disk artifact in ml/models/; it is served from B2 cache (.model_cache/qlib_alpha158/v4-v6). The cached versions (v4/v5/v6) lag the registry PROD (v8), though resolve() pulls v8 on demand.
- regime_hmm.predict_regime silently defaults to 'bull' on any exception (regime_detector.py:201). SignalGenerator guards against this by refusing to ship if regime can't compute, but the screener path is more permissive.
- AutoPilot is an executor/orchestrator, not a trained model — it consumes Qlib + HMM. Its file header references 'regime_hmm v20+' while live PROD is v24 (stale comment).
- PROD truth verified against live Supabase model_versions: exactly 4 rows are is_prod=true (qlib_alpha158 v8, regime_hmm v24, tft_swing v3, finbert_india v1). lgbm_signal_gate and breakout_meta_labeler are NOT prod-flagged in the registry despite being loaded/used.
- Ensemble weights (voters.py:22) sum to 1.00: LGBM 0.30 + TFT 0.30 + Qlib 0.20 + FinBERT 0.10 + HMM 0.10. These are hard-coded/locked, not learned.
- TFT 0.68 directional accuracy and the regime HMM log-likelihood are trainer-self-reported metrics, not independently audited; directional accuracy is a generous metric and log-likelihood measures fit not profitability.
- F9 earnings_xgb model is unbuilt (registry: skipped/insufficient_supabase_data); feature correctly returns 503. F3 momentum has no passing dedicated model (momentum_timesfm v6 and momentum_chronos both quality_pass=false at ~0.49 dir-acc).
- No-fallback discipline is genuinely enforced (generator.py:123 hard-raises on missing artifacts; earnings predictor raises ModelNotReadyError) and a graveyard of correctly-retired failed models (RL agents, timesfm, chronos2, vix_tft) confirms strict self-grading.

### AI agents

- All 10 agent roles run on FREE OpenRouter models (Llama-3.3-70B, GPT-OSS-120B, Qwen3-Coder, Gemma) per the live .env AGENT_MODEL_MAP — no paid/frontier model in production; planned 235B generator not wired in.
- Per-feature LLM caps (strategy_gen, scanner_thesis, chart_vision, fno_advisor, debate) are DEFINED in core/tiers.py LLM_FEATURE_CAPS but NOT enforced — code-grep shows only 'chat' (credit limiter) and 'portfolio_doctor' (separate hand-rolled monthly quota) are actually metered. The rest rely only on tier feature-gating + the $20 kill-switch.
- User-facing 'Mood'/sentiment is an LLM at runtime (free Llama via complete_sync role='sentiment'), NOT the FinBERT-India model, which was deliberately demoted (sentiment/llm_classifier.py:4-17).
- Agents are advisory-only: Copilot/Debate/F&O explicitly return suggestions and never auto-place live trades; the only path to live execution is the separately-enforced backtest/Sharpe strategy gate, not the LLM itself.
- The $20/month kill-switch is real, persisted, and enforced (llm_budget.py) but is single-instance-accurate — on multi-instance deploys it can overshoot by up to one 60s reconcile window (code acknowledges this).
- Multi-agent 'depth' is structural, not capability: Doctor specialists degrade to neutral stubs when input data is missing, and the Doctor composite score uses a fixed neutral base for promoter rather than the promoter agent's own read.
- Embedded per-page agents are launchpads that redirect to the single Main Chat (EmbeddedAgent.tsx:54-60), so the 'domain agent embedded in every feature' framing is, in shipped reality, a hand-off to one central chat.
- Minor config drift: vision model google/gemma-4-31b-it:free is absent from the price card and one fallback-chain entry references qwen3-8b that isn't in the live map — neither affects spend since all are free, but it signals the model map is hand-maintained.

### Automation / data / billing / security

- Scheduler is single-instance (APScheduler in one process): money-touching jobs are protected by per-day cron_lock, but the every-1/5-min monitors and digest jobs are NOT lock-guarded and would duplicate on a multi-instance deploy.
- DATA_PROVIDER defaults to 'free' (yfinance, Tier-3 delayed/approximate). Real accuracy needs the funded Kite admin account + DATA_PROVIDER=kite, and even then a daily 6am token expiry can silently drop back to fallback.
- Several 'AI' jobs (intraday LSTM, Chronos forecast, FinBERT sentiment, AI Stock Ranker) skip silently with model_not_ready if the artifact/dep is absent — they show green ('skipped') while doing nothing.
- Earnings predictor (F9) is rule-based; the XGBoost model is not trained. The calendar/predictions surface works but the ML label does not.
- WhatsApp notifications are fully built but DORMANT — is_configured() returns False until Gupshup/Meta business verification; every send is a no-op. 'WhatsApp alerts (Pro)' currently delivers nothing.
- All Tier-2/3 data (jugaad-data, NSE FII/DII/delivery/bulk, screener.in) are unauthenticated scrapes with no API contract — they rate-limit, change HTML, and block IPs; on failure dependent scanners/Doctor return honest-empty.
- Option Greeks/IV are computed in-process (Newton-Raphson + Black-Scholes), and when live chain data is unavailable the entire chain is synthetic — model output, not market truth.
- Per-feature LLM caps in FEATURE_MATRIX are largely defined-but-not-enforced at the endpoint level; the real hard ceiling is the global $20 kill-switch (chat caps are enforced).
- Rate limiter and LLM budget meter are single-instance/in-memory: limits diverge per worker and the $20 cap can briefly overshoot by one TTL window on multi-instance deploys.
- Global kill-switch fails OPEN by design (won't freeze trading on a DB hiccup) — a Supabase outage would not halt order placement.
- SEBI RA registration is PENDING_APPROVAL (config default). Operating a paid, trade-placing advisory product without it is, per the code's own comment, a regulatory grey zone — the biggest risk in this domain.
- Kite daily token auto-refresh screen-scrapes Zerodha's login flow; if Zerodha changes it, auto-refresh fails and an admin must refresh manually that day while data quality silently degrades.
- A dead-but-importable weaker get_current_user (network verify + stale free/starter/pro hierarchy) still exists in core/security.py alongside the correct hardened one in app.py.
- _fetch_market_data has a hardcoded simulated Nifty/VIX fallback used during data outages for the 9:15 gap/VIX check.

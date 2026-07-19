# Quant X — Deep End-to-End Audit (Pre-Redesign)
_Generated 2026-06-18 · branch feat/mldl-4engine_

## 0. Executive Summary

Quant X is a large, genuinely-engineered AI trading platform for Indian equities and F&O — Next.js 14 App Router + FastAPI + a real ML stack — and it is far more "real backend" than "Potemkin demo": 62 frontend `page.tsx` routes, ~40 backend routers exposing ~330 endpoints, a 4-voter swing-signal ensemble (Qlib + TFT + LightGBM + HMM) behind a strict no-fallback contract, a custom GraphRunner LLM layer (Copilot/Doctor/Counterpoint) under a hard budget kill-switch, real Razorpay 3-tier billing, real direct-broker OAuth, and ~55 scheduled jobs. Code hygiene is high (zero TODO/FIXME in app source), there is broad test coverage (148 pytest + 23 Playwright specs), and there is a genuinely strong, documented `components/foundation/*` primitive library. The single biggest UI/UX problem is that the app is visibly **two products glued together**: roughly half the surfaces use the new glass/`lg-surface` + token system while the other half still use the legacy `trading-surface` class with hundreds of hardcoded hex literals — and a large set of authenticated pages live outside the `(platform)` route group, so the shell is applied two incompatible ways. The second headline problem is **palette and token drift**: the shipped `lib/tokens.ts` palette (teal-green `#2BD9BC` / bg `#0B0E15`) does **not** match the stated "AI Trading OS" intent (`#00E6A7` / `#080B12`), and three or four competing color sources (`tokens.ts`, `lib/models.ts` hex, the chart's own palette, an orphaned "Deep Space neon" Tailwind theme) mean "green" is literally a different green across pages. The third is **content/trust correctness**: marketing pages break in the app's default light theme (white-on-white), advertise contradictory model counts (12 / 8 / 6 / 5 / 4) and Copilot caps (three different numbers), leak internal model names and a "SEBI Registered" claim that contradicts "registration PENDING", and a stale "5 AI voter" description sits on the public `/regime` and `/models` trust surfaces. Despite this, the bones are excellent and the backend is a frozen, well-typed contract. **Redesign-readiness verdict: GREEN to proceed, but only after a structural+token "phase 0"** — consolidate the shell, reconcile to one palette + one token set, fix the dead `success`/`danger` tokens, and run a content/honesty pass. Done in that order, the bulk of the redesign becomes a high-ROI token-driven re-skin of an already-capable system, not a rebuild.

---

## 1. Architecture at a Glance

**Stack:** Next.js 14.1 App Router + React 18 + TypeScript + Tailwind 3.4 + Radix + shadcn-style (cva + tailwind-merge) + framer-motion + recharts + lightweight-charts + SWR + Supabase (auth only). Backend = FastAPI flattened to `backend/` (no `src.`). ML under `ml/`.

**Three-tier split:**
- **Frontend** (`frontend/`): 62 page routes, a typed API client (`lib/api.ts`, ~262 methods / ~241 paths), `components/foundation/*` (33 primitives) + ~25 feature dirs.
- **Backend** (`backend/`): single FastAPI app (`api/app.py`) registering ~40 routers via guarded `try/except include_router`; `services/*` (business logic), `data/*` (market+brokers+screener), `ai/*` (signal ensemble + LLM agents + registry), `trading/*` (executors + AutoPilot).
- **ML** (`ml/`): canonical 9-stage training spine + trainers for 5 models (lgbm_signal_gate, momentum_lambdarank, qlib_alpha158, regime_hmm, tft_swing) + strong eval tooling (SPA/PBO/purged-CV/drift).

**Request flow (FE→BE→AI):** Component → `api.*` method → single `request<T>()` wrapper (attaches Supabase JWT as Bearer, sanitizes query, throws structured `ApiError`) → FastAPI router → `get_current_user` (local HS256 verify) → service layer → (for AI) GraphRunner / SignalGenerator / model registry (B2 + Postgres `model_versions`) → response. Realtime rides a bespoke `useWebSocket` (`/ws`, JWT via subprotocol) + 2 SSE streams (Copilot chat, patterns scan). No direct Supabase DB access from app code — backend is the sole data authority.

---

## 2. Frontend — Information Architecture & Shell

The shell **structure** matches the stated intent and is genuinely good: header-less desktop (`Topbar` is `lg:hidden`; `Sidebar` carries brand/nav/recent-chats/footer/collapse), `Cmd+K` palette, dual-mode (Managed vs Pro) cleanly routed through a single chokepoint (`UiModeContext` → `NavList` → `NAV`/`MANAGED_NAV`), and a polished ChatGPT-style recent-chats list. Auth is defence-in-depth (middleware cookie gate + `AuthContext` session→cookie bridge + `ClientAuthGate`), with documented dev bypasses.

The **application** of the shell is the single biggest structural debt:
- Only the ~20 routes inside `app/(platform)/` inherit `AppShell` from a route-group `layout.tsx`. A large set of authenticated pages (`signals`, `signals/[id]`, `portfolio`, `portfolio/doctor`, `watchlist`, `trades`, `stocks`, `markets`, `settings`, `stock/[symbol]`) live at app root and each **manually imports `<AppShell>`**. Consequence: AppShell unmounts/remounts on cross-section nav (sidebar state, scroll, recent-chats SWR all reset), and `AutopilotStickyStop` + `CopilotProvider` + `SystemHaltBanner` (mounted only in `(platform)/layout`) **silently don't render** on those pages.
- **Four** distinct nav chromes coexist: AppShell sidebar (app), `LightNavbar`+Footer (landing/pricing), a hand-rolled inline mini-navbar duplicated 3× (`/regime`, `/models`, `/track-record`), and a custom full-screen terminal header on `/stock/[symbol]`.
- `/markets` is **public** (in middleware `PUBLIC_PREFIXES`) yet wraps itself in the **private** `AppShell` — logged-out visitors see the authenticated sidebar + a 401'ing recent-chats fetch.
- `/fo-strategies` is **both a live page and a 301-redirect target** in middleware → unreachable by URL.
- Two competing authenticated "homes": `/copilot` (Main Chat) vs `/dashboard` (Command Center) — different entry points pull users to different homes.
- No route-level tier gating (auth-only); tier is enforced only by in-page upsell CTAs. The admin console is a separate, un-restyled shell on legacy tokens.

---

## 3. Frontend — Design System (tokens, Tailwind, globals.css, theming)

This is the layer that drives the entire redesign, so it gets the most depth.

**The good news — it is genuinely re-themeable.** Theming is CSS-variable-driven through ~25 variables. The strategy is *inverted single-source*: `:root` in `globals.css` **is** the dark theme; `html.light` overrides surfaces/text to a cool off-white light theme. `next-themes` is wired (`attribute="class"`, `defaultTheme="light"`, `storageKey="quantx.theme"`). There are **zero Tailwind `dark:` variants** and **no `html.dark` selector**, so `darkMode:'class'` in `tailwind.config` is dead config — re-theming happens purely by editing the variable blocks. Token adoption is strong (`text-primary` 1145×, `text-d-text-primary` 767×, border tokens 692×, `bg-wrap` 313×). Adopting a new palette is, in principle, a ~30-line change to the `:root`/`html.light` variable blocks + `tokens.ts`.

**The bad news — many surfaces bypass the variables, so a token swap alone won't re-skin them:**

| Liability | Detail | Severity |
|---|---|---|
| Intent palette unimplemented | Brief says `#00E6A7`/`#080B12`/`#00FF9D`/`#FF5B7F`; live values are `#2BD9BC`/`#0B0E15`/`#16C995`/`#F23645`. Intent hexes appear only twice (color-mix fallbacks in `MarketProfileCard`). | high |
| `lib/models.ts` parallel color source | Per-engine hex (`#05B878`, `#FF5947`, `#FEB113`…) drives ~225 inline-style hex literals; its bull/bear (`#05B878`/`#FF5947`) ≠ token up/down (`#16C995`/`#F23645`). Hex-guard doesn't catch inline styles. | high |
| `globals.css` is 2,528 lines, 4+ eras | deep-space-neon → LuxAlgo glass → flat trading-surfaces → Liquid Glass 2.0. Dead/duplicate utilities (`pulse-ring` defined twice, ~13 gradient-text variants). | high |
| 230 `text-white` + `bg-white/[x]` | Kept light-safe only by ~60 lines of `html.light … !important` overrides — brittle, fights specificity. | high |
| Orphaned 2025 palette in `tailwind.config` | `space`/`neon`/`tv` color families + `gradient-space/nebula/aurora` + ~30 keyframes coexist with token colors → competing vocabularies. | medium |
| Dead `success`/`danger` tokens | `tailwind.config` maps `success`/`danger` to `var(--success)`/`var(--danger)` which are **never defined** → `text-success`/`text-danger` render with no color across Managed beginner mode + several stock/strategy cards (live visual bug). | high |
| Two parallel scale systems | `tokens.ts` type 11–34px (dense fintech) vs `tailwind.config` `0.75rem–6rem` + `display-*` (marketing). | medium |
| Hardcoded hex in provider/CSS | `ThemedToaster` (`#1C1E29`…), glass/glow utilities in `globals.css` (`#4FECCD`, `#8D5CFF`…) bypass tokens. | medium |
| Multiple token namespaces | landing `bg-main`/`d-text-*`, pricing `d-bg`/`d-bg-card`, auth `l-*`, plus undefined `dot-*` tokens (broken `StockAvatar` colors) and `bg-background`/`text-accent` (undefined, on auth callback). | medium |

The newest `.lg-*` Liquid Glass 2.0 layer (`lg-ambient`/`lg-surface`/`lg-bar`/`lg-ring`/`glow-ai`) is clean, token-driven via `color-mix`, with an `@supports` fallback — the model the rest of the system should follow. Aesthetic critique: the implemented look (aurora blobs, gradient teal→purple text, neon glow, glass) is the generic 2024–25 "AI fintech" cliché — heavy LuxAlgo, little of the Cursor/Vercel restraint the DNA target wants.

---

## 4. Frontend — Component Library

Two clear quality tiers.

**Tier 1 — `components/foundation/*` (33 exports): excellent and the redesign substrate.** Button/Card/Badge/Input/NumericInput/Select/Dialog/ConfirmDialog/Sheet/Tabs/Tooltip/Popover/Toast/DataTable/StatCard/Sparkline/ChangeBadge/EmptyState/PageHeader/Skeleton/UsageMeter/Reveal/Verdict/DisclaimerFooter. Radix-backed overlays/forms, `cva` in Sheet, full loading/empty/error/focus state coverage, strong a11y, documented (README + JSDoc). Standouts: `DataTable` (generic, sortable, sticky, keyboard, skeleton/empty/error slots), `NumericInput` (onWheel-blur guard, formatters, clamp), `ChangeBadge` (Indian lakh/crore + auto tone), `Verdict` (BUY/HOLD/SELL single source), `DisclaimerFooter` (SEBI invariant).

**Tier 1.5 — copilot GenUI layer:** `EmbeddedAgent` (STATIC/LIVE streaming choreography, honest error+Retry) + `artifacts.tsx` (ChipRow/ArtifactCard/Bars/StatPills/Gauge/ActionRow) — the AI-forward differentiator, theme-aware. (Caveat: some chip/action buttons are decorative no-ops; `EmbeddedAgent` hardcodes `#8B5CF6` once.)

**Tier 2 — feature dirs (~104 files): mixed.** Newer dirs (scanner 11 files, signals) correctly compose foundation + `lg-*`. Large swathes bypass it: **125 raw `<button>` instances**, the "never inline a button" mandate is ~30% adopted. `managed/*`, `stock/*`, `landing/*`, shell internals roll bespoke styles. Charting is fragmented across **three** approaches (TradingView lightweight-charts with its own non-token PALETTE, recharts ×3, hand-rolled SVG sparklines + redundant `Spark` vs `Sparkline`). `landing/*` is a separate dated design system (hardcoded hex + stale route links). `ui/PillTabs` duplicates foundation `Tabs`; `ui/` and `shared/` are near-empty stray dirs. framer-motion is barely used (4 files; most motion is CSS).

---

## 5. UI/UX Quality Assessment (by surface cluster)

### Marketing / Onboarding — **C (onboarding A-; marketing D)**
Onboarding (risk-quiz → mode-choice → broker-connect → complete) is the cleanest, fully-tokenized, best-built area in the whole app. Marketing is data-rich (live hero chart, regime timeline, track-record sparkline) but **breaks in the default light theme** (white-on-white), shows contradictory model counts and Copilot caps, leaks internal engine names in the comparison matrix, has a stale/broken signup (Elite ₹2,499 vs locked ₹1,999, "5 signals/day" vs 1/day, a no-op plan-selection step, fabricated "5,000+ traders"/"73% win rate"), an off-spec cyan `#00F0FF` CTA, ~7 retired-route links, and a "SEBI Registered" vs "PENDING" contradiction.
**Top fixes:** force/own a theme policy for marketing; one source of truth for prices/caps/counts; rebuild signup; reconcile links; fix legal copy.

### Core Platform (signals/scanner/stock/markets/stocks/watchlist/dashboard/home) — **B-**
Feature-dense, mostly live on real data, good empty states, strong dashboard `MorningBriefing` streaming, one shared `CategorySignalsPage` for all 4 horizons. Held back by: three+ competing color systems; route-level `loading.tsx` skeletons that don't match real layouts (visible jump); `/stock/[symbol]` hardcoding `theme="dark"` (breaks light mode on the most chart-centric page) and hiding **~18 AI panels** in one ungrouped drawer firehose; `DataTable` not used on `/stocks`/`watchlist` (hand-rolled lists); per-card sparkline fetch storm.
**Top fixes:** collapse to one palette; fix theme propagation + group the stock-terminal drawer; re-author route skeletons; consolidate lists onto `DataTable`.

### AI / Trading — **C+ (split 50/50)**
The "new generation" (copilot, portfolio, deployed strategies, trades, strategies, inbox, fno hub, risk) is genuinely good — glass, Reveal, foundation, tokenized recharts, SWR, honest-empty, **best-in-class money-action safety friction** (AutoPilot ConfirmDialog/kill-switch, mobile sticky-stop, strategy live-deploy typed-name + ack + naked-short detection). The "old generation" (autopilot, regime, models, paper-trading, portfolio/doctor, settings, referrals) uses legacy `trading-surface` + hardcoded hex + `bg-primary text-black` + native `confirm()`. Plus: `/engines` is a dead stub ("Plan 3 wires this"); `fo-strategies` is a 2,543-line monolith; "Ask AI" does a full-page `window.location.href` reload; 3 ad-hoc markdown renderers; two semantic color vocabularies (up/down vs success/danger).
**Top fixes:** migrate the 18 legacy files onto `lg-surface`/foundation (mechanical); preserve all safety UX; finish-or-cut `/engines`; replace `confirm()` and hard reloads.

### Admin — **C (utilitarian, off-palette but appropriate)**
~2,894 lines across 9 pages, reads real registry/rolling data, role-gated. Uses raw Tailwind palette (purple/cyan/rose, `text-white`, `glass-card`) on a separate legacy shell — a global token swap won't reach it. Internal-only, so real model names are acceptable here.
**Top fix:** include it explicitly in the token migration or accept it stays utilitarian.

---

## 6. Frontend ↔ Backend Contract (must-preserve)

`frontend/lib/api.ts` (3,611 lines, ~262 methods over ~241 paths) **is** the contract and is frozen ("backend untouched"). Everything funnels through one `request<T>()` (Bearer JWT via `getAuthToken()`, query sanitization, structured `ApiError` carrying `.status` + tier-gate `.detail`). No direct Supabase DB access. SWR (36 hooks, global `SWRConfig`) + 119 imperative call sites; zustand installed but **unused**.

**Must-preserve plumbing:** `AuthContext` (session→cookie bridge for middleware), `UiModeContext` (server `ui_preferences.ui_mode` is source of truth), `useTier()` (soft gates) + 402/403/422 `ApiError` shapes (hard gates), `useWebSocket` subprotocol auth, the 2 SSE streams, `middleware.ts` retired-route 301 map + `PUBLIC_PREFIXES`, and raw-fetch bypass endpoints (auth/signup, push, admin/verify, telemetry). **Pure, reuse-verbatim logic:** `tradePlan.ts` (position sizing), `utils.formatPercent/asPercent` (ratio-vs-percent heuristic), `stockHref`.

**Contract risks:** response types are FE-only (no OpenAPI codegen → silent drift); pervasive `Record<string,any>` at high-traffic endpoints (market/portfolio/screener/admin); duplicate/stale `SubscriptionTier` (`free|starter|pro` vs `free|pro|elite`); three quota-signaling conventions (402 vs 429 vs in-body flags); a per-request async `getSession()`; two parallel chat contracts (assistant vs copilot); hardcoded hex in `models.ts`/`tierUpsell.ts`/`tokens.ts` diverging from each other and the spec.

---

## 7. Backend — API Surface

Single FastAPI app, ~40 routers registered behind guarded `try/except include_router` (resilient boot, but a broken module silently vanishes from the surface with only a log line). Two prefix conventions coexist (full `/api/...` in decorators with no prefix, vs short prefix + `include prefix="/api"`), producing three shared-prefix overlaps (`/api/strategies`, `/api/watchlist`, `/api/portfolio`) where route precedence depends on registration order.

Auth is local HS256 verify (`get_current_user`) with Supabase fallback; `get_user_profile` fails **closed** (503, no Pro fallback). `/ws` uses subprotocol JWT (no token-in-URL). Money paths are correctly defended: `trades/execute` consults the global kill-switch first; strategy `transition`→live enforces both a tier gate (403) **and** the OOS quality gate (422 `gate_failed`). Tier gating is centralized in `middleware/tier_gate.py` (structured 402 payload) but applied in only ~11 of ~43 route files — others hand-roll checks. `screener_routes.py` is a 140KB / ~95-endpoint monolith. 12 deprecated alias routes are hidden from OpenAPI but kept functional. **Real bug:** frontend posts `/api/telemetry/*` but backend mounts `/api/client-errors/*` → silent 404s (masked by an early-return in `reportUpgradeIntent`).

---

## 8. Backend — Services, Data Layer & AI/ML Subsystems (live vs dead)

**Core signal brain (live):** 4-voter ensemble in `ai/signals/generator.py` — Qlib Alpha158 ("Alpha", 0.20) + TFT 5-day quantile (0.30, no public brand) + LightGBM gate ("Gate", 0.30) + 3-state Gaussian HMM ("Regime", 0.10), strict no-fallback (`__init__` raises if any artifact missing). Min-agreement 3, confidence≥40, RR≥1.5, bear regime ×0.6.

**A second, divergent signal architecture coexists:** per-style engines in `ai/signals/engines/` — but **only `momentum.py` exists** (serves trained `momentum_lambdarank` via `/api/signals/momentum`); swing/positional/intraday engines and trainers are **not built**. The "4-engine" framing is **1/4 built**. The two architectures use different data loaders and output schemas (`GeneratedSignal` vs `StyleSignal`) and are unreconciled.

**Honesty issues to carry into copy:** "Mood/FinBERT" actually runs an **OpenRouter LLM zero-shot classifier** by default (FinBERT only with `USE_FINBERT_FALLBACK=1`); earnings predictor has **no trained model** (calendar/rule-based, 503 on predict); measured edge is **weak** (Qlib `realmoney_pass:false`/IC≈0.03, LGBM Sharpe≈0.1, HMM confidence structurally ~1.0). On-disk artifacts diverge from registry PROD records (TFT checkpoint hidden=32 on disk vs hidden=128 in registry; LGBM is legacy 15-feature; lgbm loads via disk fallback despite not being `is_prod`).

**Genuinely strong & live:** model registry (B2 + Postgres `model_versions` + disk fallback), the **strategy-promotion gate** (`ai/strategy/evaluation.py`, OOS walk-forward — the one path to live money), the LLM layer (custom GraphRunner, no LangChain, free→free→paid fallback, $50/mo persisted kill-switch), the brand-firewall map (`core/public_models.py` `public_label()`), AutoPilot orchestrator (Qlib top-N → Kelly → regime × VIX × caps), ~55-job scheduler, strong `ml/eval` rigor (SPA/PBO/purged-CV/drift). **Dead:** F5 AI SIP routes removed. **Data inconsistency:** serving provider (`data/market.py`) selects `free`/`kite`; ML loader (`ml/data/data_loader.py`) selects `free`/`truedata` and raises on `kite` — TrueData is trial-only and unwired into serving (train/serve skew footgun).

`PUBLIC_MODELS` defines 10 names (Alpha/Mood/Regime/Forecast/Intraday/Gate/AutoPilot/PatternScope/InsightAI/Counterpoint) vs the locked 5-engine set (Alpha/Mood/Regime/AutoPilot/Counterpoint) — brand sprawl to reconcile. `docs/QUANTX_AI_SYSTEMS_EXPLAINED.md` remains an accurate honest audit (one stale point: momentum now has a model).

---

## 9. Feature Inventory

| Feature | Route | Backend | AI/Model | Tier | Status |
|---|---|---|---|---|---|
| Signals hub | `/signals` | `signals_routes /today` | 4-voter ensemble | Free 1/day, Pro+ ∞ | live |
| Swing/Intraday/Positional signals | `/signals/{swing,intraday,positional}` | `signals_routes` | ensemble re-sliced by hold-tag | tiered | live (one stack, not 4 models) |
| Momentum signals | `/signals/momentum` | `/api/signals/momentum` | momentum_lambdarank (real) | tiered | live (new 06-17) |
| Signal detail + Counterpoint debate | `/signals/[id]` | `ai_routes /debate/signal/{id}` | 7-agent debate (advisory) | debate=Elite | live |
| Stocks browser | `/stocks` | `screener.getLivePrices` | — | Free | live (capped ~48/250) |
| Stock terminal | `/stock/[symbol]` | `dossier_routes`, `vision_routes` | Dossier/Vision/~18 panels | Vision Pro+ | partial (dark hardcoded, drawer overload) |
| Markets desk | `/markets` | `market`/`screener`/`news` | Mood agent (LLM) | public/auth-gated agent | live |
| Regime | `/regime` | `publicTrust.regimeHistory` | HMM | public | partial (stale "5-voter" copy) |
| Watchlist | `/watchlist` | `watchlist_live_routes` | consensus + sentiment | Free 5-cap, Pro+ ∞ | live |
| Strategies (Library/Builder/Deployed/Discovered) | `/strategies*` | `strategies_routes` | NL→DSL + OOS gate | build Free, deploy Pro+ | live (standout) |
| F&O Desk | `/fno` | `screener` OI feeds | rule-based + LLM advisor | Elite | live |
| F&O Strategy Lab | `/fo-strategies` | `fo_strategies_routes` | rule-based recs + LLM | Elite | live (2,543-line monolith; redirect conflict) |
| Portfolio | `/portfolio` | `portfolio_routes`, `positions` | — | Free | live |
| Portfolio Doctor | `/portfolio/doctor` | `portfolio_doctor_routes` | 5-agent FinRobot CoT | Pro 1/mo, Elite ∞ | live |
| AI SIP / Rebalance | (deleted) | — | — | — | dead |
| AutoPilot | `/autopilot` | `auto_trader_routes` | Qlib+HMM+VIX+Kelly orchestrator | Pro/Elite | live (live-off removed 06-12) |
| Paper trading | `/paper-trading` | `paper_routes /v2/*` | — | Free | live |
| Main Chat / Copilot | `/copilot` | `ai_routes /copilot/chat(+stream)` | 4-node graph, SSE | caps 5/50/200 | live (flagship) |
| Engines | `/engines(+[slug])` | `lib/engines.ts` (4) | — | — | partial/stub |
| Models (public) | `/models` | `publicTrust.models` | per-model accuracy | public | partial (brand-leak risk) |
| Track record (public) | `/track-record` | `publicTrust` | — | public | live |
| Managed Home/Activity/Risk | `/home`,`/activity`,`/risk` | `managed_routes /overview` | deterministic (0 LLM) | beginner mode | live |
| Dashboard | `/dashboard` | `dashboard_routes /overview` | MorningBriefing stream | Pro mode | live |
| Scanner/QuantScan | `/scanner` | `screener_routes` (~95) | rule-based + AI sub-endpoints | mostly Pro | live |
| Earnings (F9) | (signal/stock pages) | `earnings_routes` | no model (calendar) | — | stub (503 honest) |
| WhatsApp digest | settings | `whatsapp_routes` | — | Pro | partial (dormant until BSP) |
| Broker OAuth | settings/onboarding | `broker_routes` | — | all | live |
| Admin suite | `/admin/*` | `admin_routes + admin/*` | real registry/rolling | admin | live |

---

## 10. Tech Health & Risks — Consolidated Issues

| # | Severity | Issue | Area |
|---|---|---|---|
| 1 | **critical** | Marketing pages render white-on-white in the app's default light theme (dark-only `text-white` + `defaultTheme='light'`, no `forcedTheme`) | marketing |
| 2 | **critical** | Brand-firewall leaks + contradictory model counts (12/8/6/5/4) shown to users; comparison matrix prints internal engine refs | marketing/AI |
| 3 | **critical** | Signup stale + structurally broken (₹2,499 vs ₹1,999, 5/day vs 1/day, no-op plan step, fabricated social proof) | marketing |
| 4 | high | Shell applied two incompatible ways — `(platform)` layout vs per-page `<AppShell>` (remount jank, missing global FAB/halt-banner) | shell |
| 5 | high | `/fo-strategies` is both a live page and a 301-redirect target → unreachable | shell |
| 6 | high | Two coexisting design generations (~18 `lg-surface` vs ~18 `trading-surface`) — app looks like two products | components/AI |
| 7 | high | Intent palette (`#00E6A7`/`#080B12`) NOT implemented; live is `#2BD9BC`/`#0B0E15` | design-system |
| 8 | high | Three+ competing color systems + ~225 inline-hex literals from `lib/models.ts`; bull/bear ≠ token up/down | design-system |
| 9 | high | Dead `success`/`danger` tokens render with NO color (Managed home, Fusion verdict, several cards) | components |
| 10 | high | `globals.css` 2,528 lines, 4+ eras, dead/duplicate utilities; 230 `text-white` patched by `!important` block | design-system |
| 11 | high | `/regime` + `/models` describe a stale "5 AI voter" ensemble (now 4 voters, Mood standalone) | AI/content |
| 12 | high | `/engines(/[slug])` are dead stubs contradicting the live engine model | AI |
| 13 | high | "4-engine" style program only 1/4 built (momentum only); two divergent signal architectures unreconciled | backend/AI |
| 14 | high | Telemetry path mismatch `/api/telemetry/*` (FE) vs `/api/client-errors/*` (BE) → silent 404s | contract |
| 15 | high | Route-level `loading.tsx` skeletons don't match real layouts → hydration jump | core |
| 16 | high | `/stock/[symbol]` hardcodes `theme="dark"` + ~18-panel ungrouped AI drawer | core |
| 17 | medium | Mood runs an LLM, not FinBERT; earnings has no model — copy honesty risk | backend |
| 18 | medium | On-disk artifacts diverge from registry PROD (TFT/LGBM); lgbm not `is_prod` but required | backend |
| 19 | medium | Inconsistent tier gating (~11/43 files use `tier_gate`); two prefix conventions + shared-prefix overlaps | backend |
| 20 | medium | `screener_routes.py` 140KB/~95 endpoints; `fo-strategies` 2,543-line monolith; `settings` 1,138-line monolith with 6× duplicated toggles | backend/FE |
| 21 | medium | `Record<string,any>` + FE-only types (no codegen) → silent contract drift; duplicate `SubscriptionTier` | contract |
| 22 | medium | Charting fragmented (3 approaches, non-token PALETTE); `Spark` vs `Sparkline` redundant | components |
| 23 | medium | Native `confirm()` on destructive actions; "Ask AI" = full-page `window.location.href` reload; 3 markdown renderers | AI |
| 24 | medium | Silent error-swallowing hides failures as empty states (trades, compare) | AI/core |
| 25 | medium | SEBI "Registered" vs "PENDING" contradiction + visible `<PENDING>` placeholder | marketing/legal |
| 26 | medium | Conflicting Copilot caps advertised on 3 surfaces (5/50, 5/150/400, 5/50/200) | marketing |
| 27 | low | `zustand` dead dep; orphan/undiscoverable routes; dead `href="#"` socials; WhatsApp dormant; per-feature LLM caps unenforced; two build paths (Dockerfile vs nixpacks); stale comments | mixed |

---

## 11. Reusable Assets (keep) vs Replace (rebuild)

**KEEP (re-skin / reuse verbatim):**
- `components/foundation/*` (33 primitives) — the design substrate; restyle, don't rebuild. `DataTable` is the canonical list contract.
- `components/copilot/EmbeddedAgent` + `artifacts.tsx` + `CopilotProvider` — the GenUI/streaming engine for all AI surfaces.
- `components/shell/*` (AppShell, Sidebar, NavList, Topbar, MobileDrawer, CommandPalette) + `nav.ts` + `UiModeContext`/`ModePanel` — IA + dual-mode chokepoint; preserve structure.
- The CSS-variable token contract in `globals.css` + `.lg-*` Liquid Glass layer + `lib/tokens.ts` scales (spacing/radius/motion/z) — the re-theme surface.
- `lib/api.ts` (frozen contract), `useTier`/`useBrokerStatus`, `ApiError`/`handleApiError`, `AuthContext`/`UiModeContext`, `useWebSocket`, `tradePlan.ts`, `utils.cn/formatPercent`, `stockHref`, telemetry/AB harness, `types/strategies.ts` (correct `EngineName`).
- Safety/friction UX: AutoPilot ConfirmDialog/kill-switch, `AutopilotStickyStop`, strategy live-deploy modal (typed-name+ack+naked-short).
- Reference implementations: `app/portfolio/page.tsx`, `strategies/deployed/page.tsx`, `app/copilot/page.tsx`, `signals/SignalCard.tsx`, `CategorySignalsPage`.
- Onboarding engine, Razorpay checkout, `LightweightChart` logic, `middleware.ts` redirect map + CSP, brand-firewall mappers, `BrokerLock`/`CopilotQuotaModal`, Playwright suite.
- Backend: `tier_gate.py`, `evaluation.py` OOS gate, model registry, `public_models.public_label()`, `risk_engine.derive_levels`, scheduler job graph, `managed_overview`.

**REPLACE / REBUILD:**
- `globals.css` (rewrite, don't extend as era #5) — extract variable blocks + `.lg-*`, delete deep-space-neon + redundant gradient/lux/glass utilities.
- Orphaned `space`/`neon`/`tv` palette + duplicate shadcn hsl vars + dead `darkMode:'class'` in `tailwind.config`.
- `lib/models.ts` hex (reconcile to tokens or delete the hex field); the 230 `text-white` usages + `!important` override block.
- `landing/*` (separate dated system, stale links) — its own redesign track.
- `/engines(/[slug])` stubs — finish with real data or cut.
- The 18 legacy `trading-surface` pages — migrate onto `lg-surface`/foundation.
- Monoliths: split `fo-strategies` (2,543 lines), `settings` (1,138 lines, shared Toggle), `screener_routes.py`.
- Stock-terminal AI drawer — regroup into tabbed sections.

---

## 12. Redesign Readiness — Verdict & Recommended Sequencing

**Verdict: GREEN, conditional on a structural "Phase 0" before any visual polish.** The backend is real and frozen, the IA intent is already implemented, and a strong foundation library exists — so the redesign is fundamentally a **token-driven re-theme + consistency migration**, not a rebuild. But three pre-conditions must be cleared first or the redesign inherits silent bugs (dead tokens, white-on-white marketing, two-product split) and the token swap will only re-skin half the app.

**Phase 0 — Foundations (highest leverage, lowest visual risk; do FIRST):**
1. **Decide the palette.** Adopt `#00E6A7`/`#080B12`/etc. (then change ~3 CSS vars + `tokens.ts`) **or** formally retire the brief palette and keep what ships. This single decision gates all visual work.
2. **Fix the token system:** define-or-rename `--success`/`--danger` (kills the no-color bug on the beginner home); define/remove `dot-*`; prune the orphaned Deep-Space/neon palette + duplicate shadcn vars; reconcile `lib/models.ts` hex into tokens; align one green/one red through `--color-up`/`--color-down`; replace `text-white` with a semantic token and delete the `!important` block.
3. **Consolidate the shell.** Move every authenticated app-root page into `(platform)/` (or a shared `(app)/` group) so `AppShell`/`AutopilotStickyStop`/`CopilotProvider`/`SystemHaltBanner` apply once. Resolve `/fo-strategies` redirect-vs-page conflict. Pick ONE home (`/copilot` vs `/dashboard`).
4. **Content/honesty pass + the cheap correctness fixes:** one source of truth for prices/caps/model-count (drive from `FEATURE_MATRIX` + `publicTrust`); rewrite `/regime`+`/models` to the 4-voter reality; fix telemetry path mismatch; fix marketing theme; fix SEBI copy; re-author route skeletons.

**Phase 1 — Re-skin via tokens (high ROI):** With Phase 0 done, flip the variable blocks and verify visually. The foundation kit + `lg-*` re-theme automatically. **Highest-visual-ROI surfaces:** the AI/trading cluster (migrate the 18 `trading-surface` legacy files onto `lg-surface`/foundation — mostly mechanical) and the marketing funnel (separate track, FinStocks aesthetic).

**Phase 2 — Targeted rebuilds:** stock-terminal drawer regrouping; decompose `fo-strategies`/`settings`; consolidate lists onto `DataTable`; one charting contract + theme wiring; one markdown renderer; replace `confirm()`/hard-reloads.

**Safe to change:** all styling/skin via tokens, component internals, page layout/IA, marketing copy.
**Risky (coordinate):** API response shapes (no codegen), prefix conventions, anything touching the auth cookie bridge or WebSocket handshake.
**MUST NOT touch:** the `lib/api.ts` contract + path strings, money-path guards (kill-switch, OOS gate 422, role checks), the brand firewall (`public_label`), the no-fallback/honest-empty discipline, and the safety-friction UX on money-moving actions. Do not surface swing/positional/intraday as if engines exist (only momentum is real), and do not amplify performance claims the models' own metadata doesn't support.
# Quant X Pricing & Packaging Plan — 2026-06-12

**Goal:** Lock the launch pricing + per-tier feature packaging now that dual-mode
(Managed/Pro) is shipped, resolving the one open tension: AutoPilot is the
engine managed-mode beginners need, but it sits on the most expensive tier.

**Price points are NOT reopened.** Free ₹0 / Pro ₹999 / Elite ₹1,999 (locked
2026-05-28, live in Razorpay + `subscription_plans`: ₹999/₹2,699/₹9,599 and
₹1,999/₹5,399/₹19,199 monthly/quarterly/yearly — yearly = 20% off, ex-GST).
This plan changes WHAT each tier contains, not what it costs.

---

## 1. Current state (audited live, 2026-06-12)

**Enforced today (verified in code):**
- 1 swing signal/day Free (signals_routes), watchlist 5 symbols Free (PR-U, 402 + structured payload)
- Per-feature LLM caps via `enforce_llm_cap` middleware across 7 route files
- $50/mo LLM hard kill-switch (UsageMeter) — caps below are abuse ceilings, not cost exposure
- AutoPilot / F&O / debate Elite-gated via `RequireFeature`

**Honesty bug:** pricing page sells "Copilot 150 messages/day" on Pro; backend
`LLM_FEATURE_CAPS["chat"]` is 50. Elite page says "unlimited"; backend ceiling 200.

**Stale marketing:** the pricing page predates the entire AIL v2 + feature
buildout — Earnings Preview, News/Watchlist digests, Trade Review, Coach,
Setup Finder, Options Copilot, NL screener, Risk center, and Managed mode
itself are all missing from the feature lists.

## 2. Unit economics

| Item | ₹/month |
|---|---|
| LLM (hard cap) | ~4,300 ($50) |
| Kite Connect data | 2,000 |
| NSE EOD/Derived licence (Path A, post-launch) | 17,500–35,000 |
| Infra (Railway/Vercel/Supabase/B2) | ~5,000–10,000 |
| **Fixed floor** | **~29k (pre-licence ~11k)** |

Breakeven pre-licence: **~11 Pro subs**. With licence: **~30 Pro / 15 Elite**.
Marginal cost per user ≈ ₹0 (free-model LLM routing + per-user broker OAuth
data) → gross margin ~95%+ above the floor. Price discrimination must come
from packaging, not cost.

## 3. Competitor anchors (India)

Streak ₹690/1,400/2,800 · Sensibull ₹800 · Tradetron ₹500–5,000 ·
smallcase managers ₹100–300 · LuxAlgo ~₹3,400 ($40) · Intellectia ~₹1,700 ($20).
**Pro ₹999 sits exactly in the Streak-mid / Sensibull band; Elite ₹1,999 is
below Streak-top.** The price points are defensible — no change.

## 4. THE decision: where does AutoPilot live?

### Option A — AutoPilot Lite on Pro, uncapped on Elite ★ RECOMMENDED
Pro gains **AutoPilot Lite**: live auto-trading on the user's own broker,
**capped at ₹2,00,000 deployed capital, max 8 concurrent positions, equity
only**. Elite keeps **AutoPilot Unlimited** (no capital cap, 15 positions,
F&O allowed, RL exit manager).

- Beginner math works: ₹999/mo on ₹2L = 0.5%/mo — credible vs returns; ₹1,999 on ₹1L (24%/yr) never works.
- The price-discrimination axis becomes **capital** — exactly how AutoPilot's value scales. Anyone with >₹2L self-selects into Elite; nobody downgrades who shouldn't.
- Managed-mode upsell card points at Pro (mass market) instead of Elite (power users).
- Enforcement is one knob: `max_deployed_capital` in auto_trader config, tier-resolved (same pattern as `allow_fno`).

### Option B — keep Elite-only (status quo)
Managed mode stays an Elite funnel. Zero build, but beginners — the entire
point of managed mode — hit a ₹1,999 wall and churn. The new journey
underperforms.

### Option C — 4th "Auto" plan (₹1,499 between Pro/Elite)
Cleanest segmentation on paper; in practice reopens the locked 3-tier model,
touches the Tier enum across ~40 call sites, Razorpay plan seeds, billing
webhooks, and every upsell surface. Not worth it at launch.

### Free-tier activation hook (pairs with A): Paper AutoPilot
Free users get AutoPilot running on their **₹10L virtual paper portfolio** —
watch the AI pick, size and exit with fake money, then one click to go live
on Pro. The daily 15:50 IST cron already computes picks for Elite; extending
the emit path to write `execution_mode='paper'` trades for opted-in Free
users is a small, deterministic build (no broker, no new models). This is
the single strongest conversion device available to us: the product demos
itself for free, honestly.

## 5. Packaging v2 — full per-tier matrix

| | **Free ₹0** | **Pro ₹999/mo** | **Elite ₹1,999/mo** |
|---|---|---|---|
| **Positioning** | Try the AI risk-free | Trade with the AI (or let it trade ≤₹2L) | Full automation, full depth |
| Managed mode (beginner shell) | ✅ + **Paper AutoPilot** | ✅ AutoPilot Lite — live, ≤₹2L, 8 pos, equity | ✅ AutoPilot Unlimited + F&O + RL exits |
| Swing signals | 1/day | Unlimited + intraday | Unlimited + intraday |
| Momentum weekly Top-10 | — | ✅ | ✅ |
| Scanner Lab (50+ scanners, patterns, Setup Finder) | — | ✅ + AI thesis 30/day | ✅ + AI thesis 100/day |
| NL screener (plain-English scans) | 1/day | 10/day | 30/day |
| Copilot chat | 5 msgs/day | **150/day** (raise backend 50→150) | Unlimited fair-use (400/day ceiling; raise 200→400) |
| Strategy Studio (NL→DSL gen) | 1/day | 10/day | 30/day |
| Backtests + walk-forward gate | paper deploy only | ✅ live deploy via gate | ✅ live deploy via gate |
| Portfolio Doctor | 1/month | 10/month | 60/month |
| Watchlist | 5 symbols | Unlimited + digest 10/day | Unlimited + digest 30/day |
| Earnings Preview / News Digest / Market Explainer | view-only drivers | ✅ full narratives | ✅ full narratives |
| Trade Review + Coach + Risk center | ✅ (deterministic — free everywhere) | ✅ | ✅ |
| Chart vision (image → AI read) | — | 20/day | 60/day |
| Counterpoint debate | — | — | 10/day |
| F&O strategies + Options Copilot | — | — | ✅ (advisor 20/day) |
| WhatsApp digest + Alerts Studio | Telegram only | ✅ | ✅ |
| Weekly Review | — | ✅ | ✅ |
| Marketplace | browse | deploy | publish |
| Kill switch / honesty surfaces | ✅ always | ✅ | ✅ |

Deliberate keeps: deterministic agent layers (drivers, Trade Review, Coach,
Risk center) stay free at all tiers — they cost ₹0 to serve and are the trust
engine. LLM narratives are what's metered.

## 6. Implementation checklist (after founder locks Option A/B/C)

1. **tiers.py**: `auto_trader` → PRO; new `auto_trader_unlimited` → ELITE; new
   `AUTO_TRADER_TIER_LIMITS = {PRO: {max_deployed_capital: 200_000, max_concurrent_positions: 8, allow_fno: False}, ELITE: {...uncapped}}`.
2. **auto_trader executor/supervisor**: enforce deployed-capital cap at order
   emit (sum open position values + proposed > cap → skip + honest log line in
   managed Activity feed: "Skipped X — Pro capital limit reached").
3. **Paper AutoPilot (Free)**: scheduler emit path writes `execution_mode='paper'`
   trades into the paper portfolio for Free/Pro users who toggled it on;
   managed overview already reads honest state.
4. **LLM caps**: chat 5/**150**/**400**; pricing page prints real numbers.
5. **Pricing page rewrite**: v2 matrix above; managed-mode row first; FAQ
   entries for "What is AutoPilot Lite?" + capital cap; keep no-refund stance.
6. **Upsell surfaces**: managed Home AutopilotCard → Pro CTA (Free) /
   Elite CTA only when hitting the ₹2L cap; `tierUpsell.ts` copy.
7. **DB**: `subscription_plans.features` JSON refresh (prices unchanged — no
   Razorpay re-wiring).
8. Tests: tier-limit resolution, capital-cap skip path, paper-emit path.

## 7. Compliance line (one paragraph, not a blocker)

AutoPilot executes on the user's own broker account under their consent +
kill switch — we never custody funds (locked copy rule). Before scaling:
SEBI's 2025 retail-algo framework pushes algo platforms toward
broker-empanelment/registration; budget a legal review of the AutoPilot
ToS + per-broker algo registration in the go-live checklist. Percent-of-AUM
pricing was rejected partly for this reason — flat SaaS fees keep us clearly
on the software side of the line.

## 8. Launch levers (optional, founder's call)

- **Founding 100**: first 100 paid subs lock −30% for life (₹699/₹1,399) — urgency without touching list prices.
- **Referrals (already built, N12)**: +1 free month per converted referral.
- Yearly 20% off stays as-is (already live in DB).

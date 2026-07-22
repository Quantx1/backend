# Quant X — "FintechX" Design System v4 (2026-07-14)

Adapted 1:1 from the licensed FintechX Framer template (user's project copy).
Supersedes "Violet Minimal" v3. Premium *blue* fintech, light-first, photographic
warmth, pill geometry. Reference screenshots (READ the one for your section):
`/private/tmp/claude-501/-Users-rishi-Downloads-Swing-AI-Final/f7cba593-96bb-435a-803f-0b7b32ea0f53/scratchpad/fintechx/*.jpeg`

## 1. Tokens (exact values — globals.css is the only place hexes live)

| Token | LIGHT (native) | DARK (derived) |
|---|---|---|
| main (page) | `#EDF1F4` | `#0D0D0E` |
| wrap (cards/sidebar) | `#FFFFFF` | `#151517` |
| wrap-hover / hover / surface-2 | `#F4F7F9` | `#1E1E21` |
| line (hairline) | `#DDE5ED` | `#29292D` |
| wrap-line (L4 border) | `#C8D4DE` | `#3B3B40` |
| light (ink) | `#1D1D1D` | `#F7F7F8` |
| desc (body) | `#4D585F` | `#D3D3D7` |
| muted | `#5F6B75` | `#96969E` |
| primary (FILL, white ink on it) | `#406AE4` | `#406AE4` |
| primary-hover | `#3055C2` | `#3055C2` |
| primary-text (INK; text-primary/text-accent/ai/signature) | `#3459C9` | `#8FB0FF` |
| cyan (live) | `#2563EB` | `#5290F4` |
| up (P&L only) | `#0A6B50` | `#10B981` |
| down (P&L only) | `#B81C22` | `#F5808C` |
| warning/highlight/orange | `#9A4D00` | `#F0A94F` |
| chart-primary/accent | `#406AE4` | `#8FB0FF` |
| chart-secondary | `#5F6B75` | `#96969E` |
| chart-tooltip-bg | `#FFFFFF` | `#1E1E21` |
| chart-grid | `rgba(29,29,29,0.06)` | `rgba(255,255,255,0.06)` |

Validated (WCAG): white on #406AE4 = 4.77, on hover 6.55; L primary-text 6.14/white,
5.39 on 10% tint; L up 4.77 & down 4.62 on their 20% self-tints; D muted 6.6,
D primary-text 9.1, D up tint20 5.7, D down tint20 5.69. Keep the fill-vs-ink split
exactly as v3 did (textColor.primary/accent → `--rgb-primary-text`). Sync ALL of:
hex tokens, `--rgb-*` triplets, shadcn channel vars (`--primary: 64 106 228`,
`--ring`, `--accent`, destructive = down, radius **0.75rem**), `--success/--danger/
--warning`, `lib/tokens.ts`, and `frontend/scripts/validate-theme.mjs` (update its
D/L objects; must print ALL PASS).

Gradients (the ONE family — glossy blue):
- `--gradient-signature` / `--gradient-cta`: `linear-gradient(180deg, #5290F4 0%, #406AE4 55%, #3055C2 100%)` (both modes; contents wear white).
- `--gradient-text`: light `linear-gradient(110deg, #3055C2 0%, #406AE4 100%)`, dark `linear-gradient(110deg, #AFC6FF 0%, #7FA3FF 100%)`.
- CTA glow shadow recipe: `0 8px 24px -8px rgba(58,119,229,0.5), inset 0 1px 0 rgba(255,255,255,0.35)` (+1px lighter top edge = the template's glossy bevel).

## 2. Typography

- **Display**: Bricolage Grotesque **600** (`--font-display`). Ramp: XL 96px/1.0/−1.4px
  (hero), H1 72/1.2/−1.4, H2 48/1.2/−1, H3 40/1.2/−1, H4 32/1.2/0, H5 28, H6 24.
- **Body/UI**: Inter (`--font-sans`) — Medium 500 default. Body 18/1.3, Lead 20/1.3,
  SM 16/1.3, XS 14/1.3; SemiBold 600 for nav (16) and buttons (18/14).
- **Numerics**: Geist Mono stays (`--font-mono`, tabular).
- Load in `frontend/app/layout.tsx` via next/font/google: `Bricolage_Grotesque`
  (`variable: '--font-display'`, weights 600,700) + `Inter` (`variable: '--font-sans'`).
  REMOVE Plus Jakarta Sans. Keep `--font-geist-mono` loader untouched. In
  globals.css `:root`, change `--font-display: var(--font-sans)` alias → nothing
  (the layout now sets --font-display directly); keep `--font-mono` alias.
  tailwind fontFamily: display → `var(--font-display)` first.

## 3. Geometry & elevation

- Pills (9999px) for ALL buttons/CTAs/nav chips. Cards 16–24px (`rounded-2xl`/`rounded-3xl`).
  Inputs 12px. `--radius: 0.75rem`.
- Light mode owns soft shadows (template): cards `0 1px 2px rgba(29,29,29,0.04), 0 8px 24px -12px rgba(29,29,29,0.12)`; borders #DDE5ED. Dark = borders only.
- Photographic sections: light sky/nature imagery with white cards floating on top.

## 4. Component recipes (match screenshots)

- **Navbar** (01-hero.jpeg): floating white pill bar (max-w ~1100, rounded-full,
  white bg, subtle shadow), logo left, center links (Inter 600 16 #4D585F), right a
  BLACK pill button (`#1D1D1D`, white text) with circular arrow chip.
- **Primary CTA**: glossy blue pill (gradient-cta + glow recipe + white 600 18px
  label + trailing white circle w/ arrow icon). Secondary: white pill, ink label.
- **Cards**: white, rounded-2xl/3xl, #DDE5ED hairline, generous 24–32px padding,
  eyebrow (Pre Title: 14 600 uppercase #4D585F or blue), H3/H4 Bricolage, body Inter.
- **Badges/chips**: pill, tinted (`bg-primary/10 text-primary` etc.).
- **Stat items**: huge Inter Display Bold numbers (Text XL 86/0.72) or Bricolage; label 16/500 slate.

## 5. Landing page blueprint (app/page.tsx) — section order (template-true)

Build each as `frontend/components/landing/v4/<Name>.tsx` ('use client', framer-motion
whileInView reveals, token classes ONLY, no raw hexes — CI blocks them). Copy is for
Quant X: AI swing-trading for NSE/BSE, 5 engines (Alpha·Mood·Regime·AutoPilot·
Counterpoint), backtest-gated signals, broker OAuth (Zerodha/Upstox/Angel One),
paper trading, ₹0/₹999/₹1,999 pricing. SEBI-SAFE COPY: never promise returns, never
"we manage your money" (say "AutoPilot on your own broker account"), include
educational/risk disclaimer in footer. NEVER use real model names (TFT/Qlib/FinBERT/
HMM) — public engine names only. Never reference competitor brand names.

1. `HeroV4` (01-hero.jpeg + 01-herob.jpeg) — sky backdrop (CSS gradient `#BFE0F8→#E8F4FD`
   + `/v4/hero-sky.webp` if present in public/v4/), floating navbar, centered XL
   Bricolage headline "The AI trading desk **[AI chip]** for India" style w/ inline
   glossy AI icon chip, lead line, glossy-blue CTA "Start free" + white "View live demo",
   trust row (★ rating · SEBI-aware design · Backtest-gated signals), then a floating
   white dashboard frame at bottom (use `/v4/app-dashboard.png` if present, else a
   clean HTML mock with our tokens).
2. `ComparisonV4` (02-comparison*.jpeg) — "Trading alone vs with Quant X" before/after
   split card: left muted "Manual" list, right blue-tinted "With Quant X" list.
3. `FeaturesV4` (03-features*.jpeg) — eyebrow + H2 + 3-col feature cards (Signals w/
   explainable thesis, NL Screener, Options/F&O copilot) with small UI vignettes.
4. `OverviewV4` (04-overview*.jpeg) — photographic band; H2 "One command center for
   your book"; large floating product frame (`/v4/app-markets.png` fallback mock) +
   4 overview mini-cards (Portfolio, AI Copilot, Scanner, AutoPilot).
5. `StepsV4` (05-steps.jpeg) — "How it works" 3 steps: Connect broker (OAuth) →
   AI screens & backtests → You approve every trade. Step switcher pills.
6. `SecurityV4` (06-security.jpeg) — slim trust band: broker-OAuth only (we never
   hold funds), kill-switch, encrypted keys, backtest gate — 4 items w/ icons.
7. `UseCasesV4` (07-usecases*.jpeg) — persona cards (Swing trader · F&O trader ·
   Long-term investor · Busy professional) + logo/symbol ticker strip (reuse
   `animate-marquee`).
8. `IntegrationsV4` (08-integrations*.jpeg) — broker & data integrations grid:
   Zerodha, Upstox, Angel One, NSE/BSE EOD, Supabase-synced watchlists etc.
   (SymbolLogo/monogram tiles, no fake logos).
9. `StatsV4` (09-stats*.jpeg) — big-number stats (2,385 NSE stocks covered · 5 AI
   engines · 810+ automated tests · 100% trades backtest-gated). Honest numbers only.
10. `TestimonialsV4` (10-tail-a.jpeg) — photographic band + testimonial ticker
    (placeholder Indian trader personas, clearly generic names).
11. `PricingV4` (11-tail-b.jpeg) — 3 pill-tabbed plans ₹0 / ₹999 / ₹1,999 (feature
    lists from app/pricing/page.tsx; CTA to /pricing).
12. `FaqsV4` (12-tail-c.jpeg) — accordion (6 Qs: is it advice? broker safety? refunds
    (none — trial tier instead)? data? cancel? paper trading?).
13. `CtaFooterV4` (13-footer.jpeg) — big CTA band + DARK footer (template "Night"
    style: #0D0D0E bg, white ink, columns: Product/Company/Legal, socials,
    SEBI/educational disclaimer line, © Quant X).

Assembly: rewrite `frontend/app/page.tsx` to render exactly these in order (keep
existing metadata export pattern). Old `components/landing/*` (non-v4) stay on disk
but unreferenced from page.tsx. Landing is LIGHT-pinned: wrap page root in
`<div className="light-page">`? NO — instead force light tokens by rendering inside
`<div data-theme="light">`… simplest reliable: page root `className="light-landing"`
and add to globals.css: `.light-landing { /* copy of html.light var overrides */ }`
— the Retheme agent must add this class (duplicate the html.light var block under
`.light-landing` selector) so the marketing page is ALWAYS light regardless of app
theme, exactly like the template. App (platform) pages keep the dual-mode toggle.

## 6. Rules

- No raw hexes outside globals.css (pre-commit lint). Use token classes
  (`bg-main bg-wrap border-line text-d-text-* text-primary bg-primary text-up …`).
- Alpha tints work (`bg-primary/10`) — tokens are channel-based.
- Icons: Lucide via `@/lib/icons` ONLY (import { X } from '@/lib/icons'), 1.5px, sm sizes.
- Motion: framer-motion `whileInView` fade/rise (0.5s, ease [0.22,1,0.36,1], stagger
  60ms), marquee via `.animate-marquee`; honor reduced-motion.
- Images referenced from `/v4/...` must gracefully no-op if missing (use bg fallback).
- Run `npx tsc --noEmit` before declaring done. Never `git add/commit`.

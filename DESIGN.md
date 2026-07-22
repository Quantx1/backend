# Quant X — Design System (DESIGN.md)

> **"FintechX"** · v4 · 2026-07-15
> The single source of truth for Quant X's visual language, adapted 1:1 from
> the licensed FintechX Framer template. Supersedes "Violet Minimal" v3.
> AI coding agents: read this before generating any UI. Runtime tokens live in
> `frontend/app/globals.css` (CSS variables) and `frontend/lib/tokens.ts` (TS
> mirror); Tailwind semantic classes resolve to them via
> `frontend/tailwind.config.ts`. **Never inline a hex — always use a token.**
> Full spec: `docs/FINTECHX-SYSTEM.md` + measured specimens in
> the scratchpad `fintechx/SPECIMENS.md`.

## Overview

Quant X is an AI swing-trading platform for the Indian market (NSE/BSE). The
FintechX register is **premium blue fintech: light-first, photographic warmth,
pill geometry, flat tinted tiles** — a confident consumer-grade surface over a
professional quant-desk core.

- **Light (native)** — soft blue-grey page (`#EDF1F4`), pure-white cards,
  royal-blue accent, sky/nature photographic bands with white cards floating
  on top. Marketing pages are pinned light.
- **Dark (derived)** — near-black terminal (`#0D0D0E`) with the same royal
  blue; borders carry depth. App (platform) pages keep the dual-mode toggle.

Key characteristics:

- **One accent.** Royal blue `#406AE4` is the only brand hue. It signals the
  product and "AI" alike — there is no separate AI color family.
- **Flat tinted tiles, not shadowed cards.** The template's section card is a
  page-tint fill (`#EDF1F4`) at 20–30px radius with **no border and no
  shadow** — a recessed tile inside a white section (`.tile-tint`).
- **Pill geometry.** Every button, CTA, nav chip, and badge is a full pill
  (9999px). The signature CTA is a glossy blue pill (`.cta-gloss`).
- **Green/red are P&L semantics only** — profit/loss, up/down, buy/sell.
- **Frosted white glass** for floating chrome only (navbars, docks) —
  `.lg-surface` / `.glass-pill`, wrap-tinted and blurred, never dark smoke.
- **Text is never the series color.** Values, labels, legends wear ink
  tokens; a colored mark beside them carries the semantics.

## Colors

Every pair below is WCAG-validated (script-checked via
`frontend/scripts/validate-theme.mjs`, not eyeballed).

### Brand & Accent — the fill-vs-ink rule (load-bearing)

- **Blue fill** (`--color-primary` — `#406AE4` both modes): the ONE brand
  accent as a **fill** — white ink sits on it (4.77:1). Buttons, active
  states, focus rings, brand marks.
- **Blue hover** (`--color-primary-hover` — `#3055C2` both modes): pressed/
  hover state of the fill. Always darker, never lighter (white ink 6.55:1).
- **Blue ink** (`--color-primary-text` — light `#3459C9` / dark `#8FB0FF`):
  the accent as **text**. Tailwind's `text-primary` / `text-accent` / `text-ai`
  resolve HERE, not to the raw fill — the fill only reaches ~3:1 as text, so
  this split is load-bearing. 6.14:1 light (5.39 on its 10% tint) / 9.1:1 dark.
- **Live blue** (`--color-cyan` — light `#2563EB` / dark `#5290F4`): rare
  secondary for "live/streaming" energy. Not chrome; use sparingly.

### Surfaces

Five layers, light-native with dark derivations:

| Layer | Token | Light | Dark |
|---|---|---|---|
| L0 page canvas | `--color-main` / `bg-main` | `#EDF1F4` | `#0D0D0E` |
| L1 cards / sidebar | `--color-wrap` / `bg-wrap` | `#FFFFFF` | `#151517` |
| L2 hover / elevated | `--color-wrap-hover` / `bg-wrap-hover` | `#F4F7F9` | `#1E1E21` |
| L3 hairline borders | `--color-line` / `border-line` | `#DDE5ED` | `#29292D` |
| L4 accent borders | `--color-wrap-line` / `border-wrap-line` | `#C8D4DE` | `#3B3B40` |

The tile inversion is the template's signature: **white sections carry
page-tint tiles** (`.tile-tint` = `bg-main`), and tinted sections carry white
cards (`bg-wrap`).

### Text

| Role | Token | Light | Dark |
|---|---|---|---|
| Primary ink | `--color-light` / `text-d-text-primary` | `#1D1D1D` | `#F7F7F8` |
| Secondary (body) | `--color-desc` / `text-d-text-secondary` | `#4D585F` | `#D3D3D7` |
| Muted | `--color-muted` / `text-d-text-muted` | `#5F6B75` | `#96969E` |

### Financial Semantics (P&L ONLY)

| Role | Token | Light | Dark |
|---|---|---|---|
| Up / profit / long | `--color-up` / `text-up` | `#0A6B50` | `#10B981` |
| Down / loss / short | `--color-down` / `text-down` | `#B81C22` | `#F5808C` |
| True caution (rare) | `--color-warning` / `text-warning` | `#9A4D00` | `#F0A94F` |

Tint chips follow the `bg-up/20 text-up` pattern — up 4.77 / down 4.62 on
their own 20% self-tints in light; 5.7 / 5.69 in dark.

### Charts

Canvas-agnostic chart tokens (Recharts inline styles):
`--chart-primary` (light `#406AE4` / dark `#8FB0FF`), `--chart-secondary`
(= muted), `--chart-tooltip-bg` (= wrap/L2), `--chart-grid` (6% ink). Series
that mean profit/loss use `--color-up`/`--color-down`; everything else stays
in the blue/slate family.

### Brand Gradient

Exactly **one** gradient family — glossy blue, never a rainbow:

- Fill: `--gradient-signature` / `--gradient-cta` =
  `linear-gradient(110deg, #3B82F6 0%, #406AE4 100%)` (the EXACT measured
  template gradient — 110deg two-stop; identical both modes; contents wear
  `.text-on-signature` = white).
- Text: `--gradient-text` — light `#3055C2 → #406AE4`, dark
  `#AFC6FF → #7FA3FF` (each stop AA against its canvas).

### Validated contrast matrix (the ship gate)

Ink/desc/muted on every surface, white on both fills (≥4.5), blue ink on
canvas and its 10% tint (≥4.5), up/down/warning on canvas, cards, and their
tints, fills vs canvas ≥3:1 (non-text UI), hairlines visible. Re-run
`node scripts/validate-theme.mjs` (must print ALL PASS) whenever a token
value changes.

## Typography

- **Display**: Bricolage Grotesque 600 (`--font-display`, loaded in
  `app/layout.tsx` via next/font). Ramp: XL 96px/1.0/−1.4px (hero),
  H1 72/1.2/−1.4, H2 48/1.2/−1, H3 40/1.2/−1, H4 32/1.2/0, H5 28, H6 24.
- **Body/UI**: Inter (`--font-sans`) — Medium 500 default register. Body
  18/1.3, Lead 20/1.3, SM 16/1.3, XS 14/1.3; SemiBold 600 for nav (16) and
  buttons (18/14). Plus Jakarta Sans is REMOVED.
- **Numerics**: Geist Mono (`--font-mono`, `tabular-nums`) — every price,
  percent, and count goes through `.numeric` / `.num-display` / `MONO`.
- **Stat items**: huge bold numbers (up to 86px/0.72) with 16/500 slate labels.

Principles: display headings 600 (never 800+), tracking tightens as size
grows, body stays 400–500. Emphasis comes from weight and ink level — never
from color.

## Layout

- **Spacing**: 4px base grid. Tile padding 30–40px (template card register);
  dense data cards 16–24px. Marketing band rhythm 100–200px vertical at
  1440w; content max-width ≈ 1200px; centered eyebrow pill above H2 48
  Bricolage with lead 20 slate.
- **Shell**: fixed left sidebar (L1) · fluid main pane (L0) · right utility
  rail. Content max-width: none on data pages (full-bleed tables),
  `max-w-3xl` for chat/composer surfaces.
- **Photographic bands**: light sky/nature imagery (`/v4/*.png`) with white
  cards floating on top — marketing only, never in the platform shell.

## Elevation & Depth

| Level | Treatment |
|---|---|
| Section tile (template default) | `.tile-tint` / `.tile-tint-lg` — `bg-main` fill, 20/30px radius, **no border, no shadow** (flat recessed) |
| White card on tinted band | `bg-wrap` + optional `border-line` hairline |
| Product/dashboard frames | white, 16–24px radius, `#DDE5ED` hairline, `.shadow-card-float` (soft shadow allowed HERE only) |
| Floating chrome (navbars, docks, dropdowns) | `.lg-surface` — frosted white glass: `color-mix(wrap 72%)` + `blur(20px) saturate(1.4)`, wrap-tint border, `0 8px 32px -12px rgba(29,29,29,.14)` + white inner top edge |
| Floating pill bar | `.glass-pill` — same glass recipe + 9999px radius (the template's navbar) |
| Signature CTA | `.cta-gloss` — EXACT measured bevel: `inset 4px 4px 8px rgba(255,255,255,.3), inset -4px -4px 8px rgba(255,255,255,.3), 0 8px 16px rgba(58,119,229,.5)` on `.bg-gradient-cta` |

Both glass utilities keep an `@supports` fallback (flat `bg-wrap` + `border-line`
where `color-mix()` is unsupported). Colored glows and neon rings stay retired;
`.glow-ai` is the sole token-driven accent halo (rare, hero/AI only).

## Shapes

| Radius | Use |
|---|---|
| 12px | Inputs (`--radius: 0.75rem`) |
| 16–24px | Cards, product frames (`rounded-2xl`/`rounded-3xl`) |
| 20px | `.tile-tint` — standard section tile |
| 24–30px | `.tile-tint-lg`, stats/security/FAQ tiles (`rounded-[30px]`) |
| 9999px | ALL buttons/CTAs/nav chips/badges, `.glass-pill` (`rounded-full`) |

Corners are generous and soft — never sharp brutalism, never blob-round
beyond the pill controls.

## Components

- **Primary CTA** — glossy blue pill: `bg-gradient-cta` + `.cta-gloss` +
  white Inter 600 18px label + trailing 28px white circle with blue arrow
  icon. Height ~47px, padding 12px 30px (reserve ~54px right with arrow chip).
  The only high-emphasis element on any view.
- **Secondary button** — white pill (`bg-wrap`), ink label; ghost: no border,
  muted ink, L2 hover. Nav CTA variant: BLACK pill (`#1D1D1D` token ink
  color as fill), white text, circular arrow chip.
- **Navbar** — floating `.glass-pill` bar (max-w ~1100), logo left, center
  links Inter 600 16 `text-d-text-secondary`, black pill button right.
- **Tiles/cards** — `.tile-tint` (feature 20px pad `40px 40px 0`, vignette
  bleeds to bottom edge; stats 30px radius pad 30; security pad `30px 60px`;
  FAQ pad 40). Pricing wrapper: tile 24px radius pad 10 with inner white
  card. Comparison "after" card: black, 24px, white ink. Eyebrow 14/600
  uppercase slate or blue, H3/H4 Bricolage, body Inter.
- **Chips / badges** — pill, tinted: `bg-primary/10 text-primary` (blue ink),
  `bg-up/20 text-up`, etc., or neutral `bg-surface-2 text-d-text-secondary`.
- **Inputs** — L1 fill, L3 border, 12px radius, focus: 2px accent ring
  (`--ring`). Composer keeps `.input-animated-wrapper`.
- **Tables** — hairline row dividers only (no zebra), muted uppercase
  headers, mono numerics right-aligned, row hover = L2. Never place
  illustrations inside dense data tables.
- **AI moments** — `.ai-shimmer` thinking states, `text-ai` ink (= blue ink).
  AI chrome is the same royal blue as everything else.
- **Illustrations** — `public/v4/illus/*.png` (flat-vector, light `#EDF1F4`
  ground): empty states, hero/upsell cards, onboarding moments ONLY; render
  via next/image with explicit sizes + `rounded-2xl`, ideally inside tinted
  tiles.
- **Status** — dots + label, never color alone: up-green (live), warning
  (degraded), down-red (halted).
- **Icons** — Lucide via `@/lib/icons` only, 1.5px stroke, sm sizes; feature
  hues via `lib/feature-colors.ts`.

## Do's and Don'ts

**Do**
- Pull every colour from a token; add new tokens to `globals.css` first.
- Put white ink on blue fills (`text-primary-foreground`), blue ink on
  canvases (`text-primary` → `--color-primary-text`).
- Use `.tile-tint` for section cards (flat, borderless) and reserve shadows
  for product frames and floating glass chrome.
- Keep green/red exclusively for market/P&L meaning.
- Verify both modes when touching a platform surface (`html.light` flips
  everything); marketing stays light-pinned.
- Use `.numeric`/`MONO` for every number.

**Don't**

- Don't inline hexes (CI blocks raw hex in Tailwind arbitrary values) —
  the CTA gradient comes from `--gradient-cta`, never a literal.
- Don't put borders or drop shadows on `.tile-tint` tiles — flatness is the
  register.
- Don't introduce a second accent hue, gradient family, or colored icon set
  beyond `lib/feature-colors.ts`.
- Don't use `text-primary` on a solid blue fill (blue-on-blue) or pair
  `bg-primary` with any text color other than white.
- Don't add neon glows, dark glass tints, navy surfaces, violet or
  emerald/teal chrome — those are retired systems (Violet Minimal, xAI mono,
  LuxAlgo teal, AI-Trading-OS emerald).
- Don't put real model names (TFT/Qlib/FinBERT/HMM) in UI copy — public
  engine names only. SEBI-safe copy: never promise returns, never "we manage
  your money".

## Responsive Behavior

| Breakpoint | Shell |
|---|---|
| <768px | Drawer nav, 44px touch targets, 16px input font (iOS zoom guard), safe-area utilities |
| 768–1023px | Drawer nav, 2-col tile grids |
| ≥1024px (`lg`) | Sidebar + right rail visible, 3–4-col grids |

Tables scroll inside `overflow-x-auto` containers — the page never scrolls
horizontally. Marquees (`.animate-marquee`) pause on hover and collapse under
`prefers-reduced-motion`; so does `.animate-sky-drift` (hero parallax).

## Iteration Guide

1. Change token **values** only in `frontend/app/globals.css` (both `:root`
   and `html.light`/`.light-page`), then mirror in `frontend/lib/tokens.ts`.
2. Re-run the contrast gate: `node scripts/validate-theme.mjs` — it must
   print ALL PASS before shipping.
3. `text-primary` ≠ `bg-primary` ink: if you re-hue the accent, re-derive
   `--color-primary-text` for BOTH modes (fill-vs-ink rule — current values
   fill `#406AE4`, ink `#3459C9` light / `#8FB0FF` dark).
4. Canvas-rendered charts (lightweight-charts) and Satori OG images cannot
   read CSS vars — update their literals when tokens change
   (`components/charts/LightweightChart.tsx`, `app/opengraph-image.tsx`,
   `app/global-error.tsx`, landing mockups).
5. Never edit `lib/icons.tsx` by hand — regenerate via
   `scripts/gen-icons.mjs` (Solar set).
6. Glass/tile utilities live in `globals.css`: `.lg-surface`, `.glass-pill`,
   `.tile-tint`, `.tile-tint-lg`, `.cta-gloss`, `.shadow-card-float` — reuse
   them; don't re-derive the recipes inline.

## Known Gaps

- Some platform surfaces still carry v3 card chrome (border + shadow on
  resting cards) — migrate to `.tile-tint` opportunistically as pages are
  touched.
- Dark-mode derivations of the photographic marketing bands are undefined —
  marketing is light-pinned by design; revisit only if a dark landing ships.
- Categorical chart palettes beyond 2 series are undefined — extend from the
  blue/slate family + validate with the dataviz palette script before use.

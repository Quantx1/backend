# Quant X — xAI Design Language Redesign

_Design spec · 2026-06-18 · supersedes the "AI Trading OS" emerald-teal + glassmorphism palette_

## 1. Context & Goal

Quant X is a mature AI trading platform (Next.js 14.1 App Router + FastAPI + ML). The pre-redesign audit (`docs/QUANTX_DEEP_AUDIT_2026_06_18.md`) found the app is visually "two products glued together" across 4 design eras, with palette/token drift and a generic "2024-25 AI-fintech glow" look.

This redesign replaces the entire visual language with an **interpretation of xAI's design language** (per `DESIGN-x.ai.md`): engineered restraint — a near-black canvas, white outline pills, a weight-400 geometric sans paired with an uppercase tracked mono, hairline-bordered charcoal cards, and no glass/gradients/shadows on the main surface. The goal is one coherent, distinctive, austere-but-beautiful system across all 62 routes — built as a **token-driven re-theme + consistency migration on top of a frozen backend**, not a rebuild.

This spec covers the **design system + the Copilot north-star surface**. The full roll-out across remaining routes is sequenced here but executed in waves after north-star sign-off.

## 2. Locked Decisions (from brainstorm)

1. **Strategy:** Role-assigned component libraries, all binding to **one** token system (no per-page library mixing).
2. **Design language:** xAI (`DESIGN-x.ai.md`) — near-black `#0a0a0a`, white outline pills, Inter (Universal Sans substitute) weight 400, Geist Mono uppercase labels, 8px charcoal hairline cards, no glass/shadow/gradient on main surface.
3. **Theme:** **Dark-only.** Light mode is removed (xAI is dark-canvas only; the audit showed our light theme was half-broken on marketing).
4. **Semantic color:** **Minimal duotone** — monochrome everywhere except financial data, which uses a single restrained green (up) / red (down) pair. Color never touches chrome.
5. **Sequencing:** Build the design system → re-skin `components/foundation/*` → live `/preview-design` kitchen-sink → **Copilot north-star end-to-end → user approval** → roll across routes in waves.
6. **Flagship north-star:** `/copilot` (Main Chat) — the most xAI-spiritual surface.
7. **Library roles:** shadcn = primary system; 21st/Magic = bespoke generation; Aceternity + Magic UI = a curated, mono-restyled **delight layer** (allowlist in §7); HeroUI = reference-only (Tailwind v4 blocker, deferred).
8. **"Streamlined" UX** — minimal, restrained interaction (not the Streamlit framework; we keep Next.js).

## 3. Token System

One source of truth, dark-only. `globals.css` is rewritten around xAI CSS variables; `lib/tokens.ts` and `tailwind.config.ts` are reconciled to the same set.

### 3.1 Color

| Role | Token | Value |
|---|---|---|
| Canvas (only page surface) | `--canvas` | `#0a0a0a` |
| Canvas soft (hover/tooltip) | `--canvas-soft` | `#1a1c20` |
| Canvas card | `--canvas-card` | `#191919` |
| Canvas mid (nested/code) | `--canvas-mid` | `#363a3f` |
| Hairline (all borders/dividers) | `--hairline` | `#212327` |
| Ink (primary text) | `--ink` | `#ffffff` |
| Body (secondary text) | `--body` | `#dadbdf` |
| Mute (captions/fine print) | `--mute` | `#7d8187` |
| Accent — sunset (rare/illustration) | `--accent-sunset` | `#ff7a17` |
| Accent — dusk | `--accent-dusk` | `#7c3aed` |
| Accent — twilight | `--accent-twilight` | `#c4b5fd` |
| Accent — breeze | `--accent-breeze` | `#a0c3ec` |
| **Semantic — up/bull** (financial only) | `--up` | `#3FB950` |
| **Semantic — down/bear** (financial only) | `--down` | `#F85149` |

- Accents appear **rarely** — product illustration / a single AI-moment, never as UI chrome.
- Duotone `--up`/`--down` are GitHub-dark register (calm, legible on near-black, **not neon**). Tunable on first review of real data screens.
- **The dead `--success`/`--danger` tokens are deleted** (the audit found they were referenced but never defined → no-color bug). All consumers move to `--up`/`--down`.

### 3.2 Typography

Two faces. **Weight 400 only** — the brand never bolds; size + negative tracking carry emphasis.

- **Display/body/button/link:** Inter (`next/font/google`), Universal Sans substitute, negative tracking per the display ladder below (≈`-0.025em` at 96px easing toward neutral at body sizes) — Inter approximates Universal Sans' gathered, precise look.
- **Eyebrows / labels / metric counters:** Geist Mono (`geist` npm package), **UPPERCASE**, +tracking (1.2-1.4px).

| Token | Size / LH / Tracking | Use |
|---|---|---|
| `display-xl` | 96 / 96 / -2.4px | Hero |
| `display-lg` | 72 / 72 / -1.8px | Sub-hero |
| `display-md` | 48 / 48 / -1.2px | Section headline |
| `display-sm` | 32 / 36 / -0.6px | Card-cluster heading |
| `display-xs` | 20 / 28 / 0 | Inline display |
| `body-lg / md / sm` | 18 / 16 / 14 | Body copy |
| `caption-mono` | 14 / 20 / 1.4px | Section eyebrow (Geist Mono UPPER) |
| `caption-mono-sm` | 12 / 16 / 1.2px | Small mono label |
| `button-md` | 14 / 20 / 0 | Button label |

### 3.3 Spacing / Radius / Elevation

- **Spacing** (4px base): `xxs` 2 · `xs` 4 · `sm` 8 · `md` 12 · `lg` 16 · `xl` 24 · `2xl` 32 · `3xl` 48 · `4xl` 64. Section bands 64px desktop; card interior 24px.
- **Radius:** `none` 0 (full-bleed bands) · `sm` 8px (cards) · `pill` 9999px (**every interactive element**).
- **Elevation:** flat + hairline only. **No shadows, no glass, no gradients** on the main surface. Hairline borders carry all elevation.

### 3.4 Re-skin mechanism (high leverage)

The audit found token adoption is already strong (`text-primary` 1145×, border tokens 692×, `bg-wrap` 313×, `text-d-text-primary` 767×). **Remapping the CSS-variable definitions re-skins most token-using surfaces automatically.** `:root` becomes the xAI dark theme; `html.light` and all `dark:` variants are removed (`darkMode:'class'` was already dead config). Legacy `trading-surface` + hardcoded-hex pages are migrated by hand in the roll-out waves.

## 4. Fonts

- Inter via `next/font/google` (weight 400; 500 only if an a11y legibility need is proven — default 400).
- Geist Mono via the `geist` package.
- Wired in `app/layout.tsx`; old `components/shell/appFont.ts` reconciled. CSS variables `--font-sans` / `--font-mono` drive Tailwind `fontFamily`.

## 5. Primitive Re-skin (`components/foundation/*` → xAI)

The 33 foundation primitives keep their **API**; only their **skin** changes. shadcn is the pattern source (installed via `npx shadcn@latest`, owned in-repo, restyled to tokens).

| Primitive | xAI treatment |
|---|---|
| `Button` | Default = outline pill (translucent-white border). Primary = rare white-filled pill. `scale(0.97)` on `:active`. |
| `Card` / `StatCard` | 8px `#191919` rect, 1px `#212327` hairline, no shadow. |
| `Input` / `NumericInput` / `Select` | `canvas-soft` fill, hairline, 8px radius, mono placeholder. |
| `Badge` / `ChangeBadge` / `Verdict` | Mono by default; **duotone only** for direction / P&L / BUY-SELL. |
| `Tabs` | Mono-caps labels; pill or underline indicator. |
| `DataTable` | `caption-mono` header (UPPER), `body-sm` rows, hairline row borders. Becomes the canonical list contract (replaces hand-rolled `/stocks`,`/watchlist` lists). |
| `Dialog` / `Sheet` / `Popover` / `Tooltip` | Hairline charcoal surface, mono labels; modals stay centered, popovers/tooltips scale from trigger origin. |
| `Skeleton` / `EmptyState` / `PageHeader` | Mono; empty states host subtle `dot-pattern`. |
| **New: `EyebrowMono`** | The uppercase tracked Geist Mono label above every section — the xAI signature. |

All primitives are verified together in a new **`/preview-design` kitchen-sink route** (dev-only) so the skin can change without touching consumers.

## 6. Library Roles (concrete)

| Library | Role | Scope | Install |
|---|---|---|---|
| **shadcn** | The system — primitive patterns re-skinned to xAI | whole cockpit | `npx shadcn@latest` |
| **21st/Magic** | Generate bespoke components themed to xAI tokens | per-need | MCP `21st_magic_component_builder` |
| **Aceternity** | Curated mono-restyled delight (allowlist §7) | marketing + select app moments | `npx shadcn@latest add https://ui.aceternity.com/registry/<x>.json` |
| **Magic UI** | Curated mono-restyled delight (allowlist §7) | marketing + select app moments | shadcn registry |
| **HeroUI** | Reference-only (needs Tailwind v4) | none yet | deferred |

## 7. The Delight Layer (Aceternity + Magic UI allowlist)

**Principle:** every effect is (a) restyled to monochrome (color only ever on financial numbers), (b) fired only at frequency-appropriate moments (Emil's framework: rare/hero/empty/first-time get delight; 100+/day actions and dense live data do not), and (c) authored per `emil-design-eng` and gated by `review-animations` before merge, with `prefers-reduced-motion` honored.

### ✅ ALLOW — marketing AND select app moments (mono)
- **number-ticker** (Magic) — big metrics on *first paint* only, never per live tick.
- **marquee** (Magic) — index/watchlist ticker-tape (mono chrome + duotone numbers), linear, pause-on-hover.
- **blur-fade / text-animate** (Magic) — section/message enter, <300ms, staggered.
- **typing-animation / text-generate-effect** (both) — Copilot streamed-response reveal.
- **terminal** (Magic) — Copilot tool-call / "thinking" trace (pure mono, very Grok).
- **animated-list** (Magic) — activity feed · inbox · notifications.
- **dot-pattern / grid-pattern / noise-texture** (Magic) — low-opacity white-on-black texture on hero + empty states.
- **animated-circular-progress-bar** (Magic) — risk/health gauges (mono).
- **scroll-progress / progressive-blur** (Magic) — long pages.
- **placeholder-and-vanish-input** (Aceternity) — Copilot composer.
- **confetti** (Magic) — rare celebration (first paper trade / first deploy), white/mono.
- **animated-tooltip / loader / animated-tabs** (both) — app-wide, hairline-mono.

### ⚠️ MARKETING-ONLY (heavier; mono-restyled)
`hero` · `hero-highlight` · `text-hover-effect` (the documented x.ai effect) · `bento-grid` · `feature-section` · `timeline` · `expandable/focus cards` · `resizable-navbar` · `world-map`/`globe` (mono dots) · `avatar-circles` · `magic-card` (white spotlight, feature cards only) · device mockups.

### 🚫 BANNED everywhere (intrinsically neon/gradient/glow)
`aurora-text` · `animated-gradient-text` · `neon-gradient-card` · `rainbow-button` · `border-beam`/`animated-beam`/`background-beams` · `meteors` · `light-rays` · `retro-grid` · `warp-background` · `sparkles-text` · `shimmer-button`/`shine-border` · `pulsating-button` · `cool-mode` · `glare-hover` · `backlight`.

## 8. Motion Discipline

- Authored per **`emil-design-eng`**: `ease-out`/custom curves (never `ease-in` on UI), <300ms UI, pill `:active` scale(0.97), `scale(0.95)+opacity` entrances (never `scale(0)`), transitions over keyframes for interruptible UI, GPU-only props, asymmetric enter/exit.
- Every animation **blocked-or-approved by `review-animations`** before merge.
- `prefers-reduced-motion`: drop movement, keep opacity/comprehension cues. Hover effects gated behind `@media (hover:hover) and (pointer:fine)`.

## 9. North-Star: `/copilot` rebuilt in xAI

**Backend & data flow untouched** — same SSE stream (`/copilot/chat/stream`), same `lib/api.ts` contract, same `EmbeddedAgent` / `artifacts.tsx` engine, re-skinned only.

- **Empty state:** centered `display-md` Inter-400 headline (tight tracking), a `caption-mono` eyebrow, a row of **suggested-prompt outline pills**, subtle `dot-pattern` background.
- **Composer:** single hairline pill input on `canvas-soft`, mono placeholder, white-filled send pill; `placeholder-and-vanish-input` flourish on submit.
- **Message stream:** user right / assistant left; mono-caps role eyebrows; **one** markdown renderer (replaces the 3 ad-hoc renderers the audit found), restyled to xAI; assistant tokens reveal via `typing-animation`; messages enter via `blur-fade`.
- **Tool trace:** rendered as a mono `terminal` block.
- **GenUI artifacts** (sparkline / regime bars / stat cards): mono chrome, **duotone only on numbers**; charts restyled to the xAI palette (eliminating the non-token chart `PALETTE`).

## 10. Architecture & Isolation

- Tokens: `globals.css` (rewritten) + `tailwind.config.ts` + `lib/tokens.ts`, one set (mono + duotone). Fonts in `app/layout.tsx`.
- Light mode removed; `next-themes` simplified to dark-only (or removed if no toggle remains).
- Each primitive independently restyled + visible in `/preview-design`.
- **Folded-in structural wins** (audit Phase 0, only where touched): delete dead `--success`/`--danger`; one chart contract; one markdown renderer.
- Existing `lib/models.ts` inline hex reconciled to tokens (or its `color` field removed).

## 11. Hard Guardrails — DO NOT TOUCH

- `lib/api.ts` paths & response contract (frozen — backend untouched).
- Money-path guards: AutoPilot kill-switch, strategy OOS gate (422 `gate_failed`), role/tier checks.
- Brand firewall (`public_label()` — no real model names in UI).
- No-fallback / honest-empty discipline.
- Safety-friction UX on money-moving actions (ConfirmDialog, typed-name + ack, sticky-stop).
- **Every existing feature keeps its route** (redesign preserves features; folded ≠ deleted).
- Do not surface swing/positional/intraday as if they are separate live engines (only momentum is built); do not amplify performance claims beyond model metadata.

## 12. Build Sequence

1. **Design system** — fonts (Inter + Geist Mono), token rewrite (`globals.css` / tailwind / tokens.ts), shadcn init, delete dead tokens, remove light mode.
2. **Foundation re-skin** + `/preview-design` kitchen-sink → quick visual check.
3. **`/copilot` north-star** end-to-end → **user approves the look.**
4. **Roll-out waves (post-approval):** cockpit core (signals/scanner/stock/markets/watchlist/dashboard/home) → AI/trading (autopilot/strategies/portfolio/paper/fno/regime) → marketing (landing/pricing, restrained Aceternity/Magic) → admin → settings.
   - Each wave folds in the relevant audit fixes (shell consolidation into one route group, route-skeleton re-authoring, list→DataTable, stock-terminal drawer regroup, monolith splits) **only where the surface is already being touched.**

## 13. Verification

- Playwright before/after screenshots per surface (live visual check).
- `review-animations` gate on all motion.
- `vercel:react-best-practices` + `/code-review` on the code.
- Brand-firewall test stays green; existing Playwright suite stays green; no `lib/api.ts` contract change.

## 14. Risks / Constraints

- **HeroUI deferred** — requires Tailwind v4 (app is on 3.4). Revisit as a separate migration if first-class HeroUI is wanted later.
- **Aceternity/Magic restraint** — their default look fights xAI; strictly the §7 allowlist, mono-restyled. Flashier effects only ever in marketing.
- **Duotone discipline** — color must stay on financial data only; chrome leakage reverts us to a generic look.
- **Light-mode removal** — confirm no user/legal requirement for a light theme before deleting it.
- **Scope** — only the design system + Copilot north-star are committed here; full roll-out is sequenced but each wave is its own implementation step.

## 15. Open Questions (resolve during build)

- Exact duotone greens/reds — tune against real signal/P&L screens.
- Whether to keep a theme toggle at all (dark-only ⇒ likely remove the Settings → Appearance toggle).
- `framer-motion` vs CSS/WAAPI per animation — decided case-by-case via `emil-design-eng`.

## References
- Audit: `docs/QUANTX_DEEP_AUDIT_2026_06_18.md`
- xAI tokens: `DESIGN-x.ai.md`
- Memory: pre-redesign audit, AI Trading OS design (superseded palette), visual-style-commitment, redesign-preserves-features.

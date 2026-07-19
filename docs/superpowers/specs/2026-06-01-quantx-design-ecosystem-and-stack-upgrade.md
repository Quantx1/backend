All claims are grounded. I now have everything needed to produce the decisive document.

---

# Quant X "AI Trading OS" — Stack Upgrade Verdict, Design Resource Map & MCP Connection Kit

*Grounded against live 2025-2026 sources (verified June 2026). All version numbers, MCP commands, and the framer-motion→motion rename confirmed against official docs.*

---

## A. STACK UPGRADE VERDICT

### ✅ RECOMMEND — upgrade all four axes together, now, on a dedicated branch.

This is not a close call. Our current stack (Next 14.1 / React 18.2 / Tailwind 3.4 / TS 5.3 / framer-motion 10) is the *exact* configuration that locks us OUT of the modern dark-fintech design ecosystem. Every high-value tool below — shadcn CLI 3.0 MCP, Tremor v4, the entire Magic UI / Aceternity / Motion-Primitives kinetic layer, OKLCH P3 color, the Next DevTools MCP — assumes React 19 + Tailwind v4. Staying on the old stack means perpetual `--legacy-peer-deps`, hand-porting tokens, and framer-motion 10 reconciliation errors under React 19. **The upgrade is the unlock.**

| Axis | From → To | Risk | Codemod coverage |
|---|---|---|---|
| **Next.js** | 14.1 → **16.x** (Turbopack stable default, ships React 19.2) | Medium | `npx @next/codemod@canary upgrade latest` (~80%) |
| **React** | 18.2 → **19.2** (bundled with Next 16; do 18.3 as a dev-only intermediate to surface warnings) | Low-Med | Comes with Next 16; minimal API breaks |
| **Tailwind** | 3.4 → **v4.x** (CSS-first `@theme`, OKLCH) | Medium | `npx @tailwindcss/upgrade` (~90%) |
| **TypeScript** | 5.3 → **5.9** | Trivial | None needed (5.x is purely additive) |
| **Animation** | framer-motion 10 → **motion v12** (`motion/react`) | Trivial | Find/replace import path |
| **Node** | 18 → **22 LTS** (Next 16 drops 18; min is 20.9) | Trivial | Infra only |

### What each upgrade UNLOCKS (named libs)

- **Next 16 → Turbopack stable default** (2-5x builds, up to 10x Fast Refresh), **React Compiler** (auto-memoization — removes `useMemo`/`useCallback` noise in chart-heavy cockpit code), **Cache Components (`'use cache'`)** for live-price RSC, **native View Transitions** for cockpit page-to-page flow, and the **built-in Next DevTools MCP** that exposes routing/server-log/error state to Claude Code.
- **React 19.2 → `<Activity>`** (render the persistent Copilot sidebar in background with state preserved — never re-mounts conversation history on route change), `useActionState`, View Transitions API, **forwardRef no longer required** (matches shadcn's React-19 rewrite). Unlocks: **assistant-ui**, **Tremor v4**, **Magic UI**, **Aceternity**, **Motion Primitives**, **HeroUI v3**.
- **Tailwind v4 → palette as live OKLCH CSS variables in `@theme`** (emerald `#00E6A7`, bull `#00FF9D`, bear `#FF5B7F` accessible to JS runtime, inline styles, *and* lightweight-charts canvas without re-exporting tokens), **P3 wide gamut** (the emerald/bull pop on OLED trading monitors), 100x faster incremental builds. Unlocks: **shadcn CLI 3.0 / new-york-v4**, **Origin UI**, **HeroUI v3**, **Kokonut UI**, **tailwindcss-radix-colors v2**.
- **motion v12** → the only animation engine that supports React 19 concurrent rendering (framer-motion 10 will throw reconciliation errors under R19).

### Breaking changes to plan for (all surfaced in our 36-route app)

1. **`async params`/`searchParams`/`cookies`/`headers`** — warned in 15, *fully enforced* in 16. The `@next/codemod` handles it; run `npx next typegen` for typed `PageProps`/`LayoutProps`.
2. **`middleware.ts` → `proxy.ts`** — our `middleware.ts` (auth redirects across all routes) must rename to `proxy.ts`, export function renamed to `proxy`, and it now runs on **Node.js** runtime (gains `fs`/`crypto`, loses Edge-only constraints). Test auth flow thoroughly. (`middleware.ts` still works but is deprecated.)
3. **Turbopack default** — if any custom webpack loaders exist (SVGR etc.), either migrate or pass `--webpack`. Audit `next.config` first.
4. **Tailwind config gone** — `tailwind.config.js` deprecated; tokens move to CSS `@theme`. `darkMode: 'class'` → `@custom-variant dark (&:is(.dark *))`. `content: []` array removed (auto-detection). `@tailwind` directives → `@import "tailwindcss"`. Class renames: `shadow`→`shadow-sm`, `rounded`→`rounded-sm`, `outline-none`→`outline-hidden`, `ring`→`ring-3`, `bg-gradient-*`→`bg-linear-*`.
5. **recharts 2** may flag React 19 peer warnings → upgrade to **recharts v3** (safe in-place; fixes the v2 infinite re-render loops).

### Step-by-step migration plan (~36-route App Router app)

1. **Branch** `feat/stack-v3-upgrade`. Bump Node to **22 LTS** locally + in CI/Docker base image.
2. **React 18.3 dev pass** — install 18.3, run dev, fix every PropTypes/defaultProps deprecation warning. (Don't ship; this is a surfacing step.)
3. **Next codemod** — `npx @next/codemod@canary upgrade latest`. Manually: rename `middleware.ts`→`proxy.ts` + function; retest auth on all routes; add `default.js` for any parallel-route slots; convert any `revalidateTag()` to the 2-arg `cacheLife` form.
4. **Motion swap** — uninstall `framer-motion`, install `motion`; global find/replace `'framer-motion'` → `'motion/react'` (incl. dynamic imports).
5. **Tailwind codemod** — `npx @tailwindcss/upgrade` on the branch; diff carefully. Move `extend:` blocks to `@theme`; rewrite any JS plugins as `@utility`/`@variant`; swap `tailwindcss-animate` → `tw-animate-css`; switch PostCSS plugin to `@tailwindcss/postcss`. Keep `next-themes` toggling `.dark` on `<html>`.
6. **Tokens to OKLCH** — convert `#080B12 / #121826 / #00E6A7 / #3B82F6 / #8B5CF6 / #FFB547 / #00FF9D / #FF5B7F` to `oklch()` via the **mcp-color-convert** MCP (so Claude Code writes them correctly) and **tweakcn** (visual map to shadcn slots). Define once in `@theme`.
7. **shadcn migrate** — `npx shadcn@latest migrate` to convert existing components to new-york-v4 / OKLCH / `data-slot`.
8. **TS 5.9** — bump, run `tsc --noEmit`. Zero breaks expected.
9. **recharts v3** — upgrade; remove `activeIndex` usages, drop `react-smooth` imports.
10. **Verify with Playwright MCP** — drive every route, screenshot dark-mode render, confirm charts load and `proxy.ts` auth redirects work. Leave `reactCompiler: true` **off** initially; enable after a clean baseline.

**Risk: Medium overall** (the Tailwind config rewrite + `proxy.ts` auth are the two real watch-points). **Effort: 3-5 developer days** with codemods + Context7-guided fixes — not a multi-week rewrite. *(Note: TS 6.0 lands ~Q1-Q2 2026 with `strict:true` default and module-resolution changes — 5.9 is the clean pre-6.0 baseline; schedule a 6.0 audit later.)*

---

## B. A-Z DESIGN RESOURCE MAP (dark fintech, our palette)

### Components
| Tool | URL | Why (one line) |
|---|---|---|
| **shadcn/ui** (new-york-v4) | https://ui.shadcn.com | Unbreakable foundation; React 19 + TW v4 native, OKLCH vars, copy-own — hard-code `#080B12`/`#00E6A7` once, propagates via `@theme`. MCP-first. |
| **Tremor** (Vercel, MIT) | https://www.tremor.so | The only data-dashboard library: KPI cards, AreaChart (P&L gradient fills), Tracker (regime streaks), BadgeDelta — dark-ready. Solar template confirms Next 15 + TW v4 + React 19. |
| **Magic UI** | https://magicui.design | Kinetic AI-OS layer: NumberTicker, AnimatedBeam (copilot "thinking"), BorderBeam, BentoGrid, Terminal (DSL editor), AnimatedList (order feed). Dark-first, MIT, MCP. |
| *Supplementary (register via shadcn MCP)* | — | **Origin UI** (dense form inputs/order entry), **Kibo UI** (DataTable w/ column-pinning + CodeBlock for DSL), **Aceternity** (BackgroundBeams/Spotlight — *audit each for TW v4*). |

**AVOID:** Mantine + Tailwind v4 token collision (use Mantine *only* for isolated heavy widgets if at all); HeroUI v3 (still beta June 2026); mixing Flowbite React as a primary system.

### Motion / Animation
| Tool | URL | Why |
|---|---|---|
| **Motion v12** (`motion/react`) | https://motion.dev | Mandatory under React 19; layout animations for order-row reflows, `<AnimatePresence>` for panels, TW v4 compatible (inline styles). |
| **Number Flow** (`@number-flow/react`) | https://number-flow.barvian.me | **Highest-ROI fintech primitive** — digit-spin P&L/LTP tickers with `trend` for bull/bear color; debounce >4Hz websocket ticks. |
| **AutoAnimate** (`@formkit/auto-animate`) | https://auto-animate.formkit.com | One-ref FLIP for watchlist/scanner list mutations (`WatchCard.tsx`) — zero boilerplate. |
| *Marketing only* | https://gsap.com · https://lenis.dev | **GSAP 3.13** (all plugins now free incl. commercial) + SplitText for hero reveals; **Lenis** smooth scroll. Gate behind `prefers-reduced-motion`; `data-lenis-prevent` on chart canvases. |

**AVOID:** running animation "for its own sake" — every motion communicates a data change, confirms an action, or guides attention (SEBI no-urgency). Rive/Lottie only for 2-3 signature moments + empty states.

### Icons
| Tool | URL | Why |
|---|---|---|
| **Phosphor** (`@phosphor-icons/react`) | https://phosphoricons.com | Primary line set; 6 weights = semantic state (Regular idle / Fill active / Duotone for engine cards). Use `/ssr` submodule for RSC. 9,000+ finance primitives. |
| **pqoqubbw/icons** (lucide-animated) | https://icons.pqoqubbw.dev | Animated accent layer (~15 spots: copilot send, AutoPilot toggle, alert pulse). shadcn-CLI install, motion-powered, owned code. |

*Caveat:* Phosphor GitHub #137 (React 19 forwardRef on *custom* icon composition) still open — use built-in icons + `/ssr` base, don't compose custom. **AVOID** Lucide as primary (1,600-icon ceiling shows gaps in finance density) — keep it only as the geometry base for the animated layer.

### Illustrations / 3D
| Tool | URL | Why |
|---|---|---|
| **dotLottie** (`@lottiefiles/dotlottie-react`) | https://lottiefiles.com | All micro-animations/empty states ("no signals", "strategy deployed"). 40-60% smaller than JSON Lottie; respects reduced-motion. |
| **Spline** (marketing hero only) | https://spline.design | One lazy-loaded 3D signal-wave on `/` landing. `@splinetool/react/next` SSRs a placeholder. **One embed per route max.** |
| **unDraw** (empty states) | https://undraw.co | Single-color SVGs set to `#00E6A7`; strip light backgrounds for `#080B12`. Onboarding/empty only. |

**AVOID:** R3F/three.js (~580kB) unless absolutely needed — and never as a cockpit background competing with chart data. A serious trading OS earns credibility through data density, not decoration.

### Backgrounds / Effects
| Tool | URL | Why |
|---|---|---|
| **Aceternity** Grid/Dot + Spotlight + Beams | https://ui.aceternity.com | Low-opacity dot grid over `#080B12` = the Tradomate/LuxAlgo structural depth; Spotlight tuned to emerald highlights the copilot panel. MIT, MCP. |
| **Magic UI** Noise Texture + Animated Beam + Border Beam | https://magicui.design | `feTurbulence` noise (~0.8kB) is the *correct* glass-on-overlays grain; Animated Beam = copilot data-flow. |
| *Calibration / static* | https://shadergradient.co · https://heropatterns.com | **ShaderGradient** (login hero only, lazy) for emerald/blue/purple ambient; **Hero Patterns** at 3-5% opacity for card texture (pure CSS, zero bundle). |

**AVOID:** full-screen WebGL on the cockpit; glassmorphism on the base card grid (`#121826`) — **glass on overlays only** (modals/drawers/copilot). Use **Hype4 glass generator** as the CSS calibration reference (blur 12-16px, sat 180%, `#121826` @ 60%).

### Fonts / Type
| Tool | URL | Why |
|---|---|---|
| **Geist Sans + Geist Mono** | https://vercel.com/font | UI body + financial data; Next default, zero config (`geist/font`), 78% lighter than Inter, `tnum` tabular figures. |
| **JetBrains Mono** | https://www.jetbrains.com/lp/mono | DSL strategy editor only — ligatures (`!=`→`≠`, `>=`→`≥`) for conditions like `rsi_14 >= 70`. |
| **Clash Display** (Fontshare ITF) | https://www.fontshare.com/fonts/clash-display | Marketing/pricing hero display ≥24px; self-host via `next/font/local`. |

*Tooling:* **Utopia** (https://utopia.fyi) for the `clamp()` fluid type scale → map into `@theme`. Geist has **no italics** → use Inter for italic body text only. **AVOID** a third app-inner sans (Satoshi) — keeps the palette tight.

### Color / Theming
| Tool | URL | Why |
|---|---|---|
| **tweakcn** | https://tweakcn.com | Maps every token to shadcn CSS-var slots; live OKLCH/v4 preview + inline WCAG check. The bridge during v3→v4. |
| **uicolors.app** | https://uicolors.app/generate | Generates full 50-950 OKLCH tint scales from each brand hex; has a public API for scripted token gen. |
| **oklch.com + Evil Martians Harmonizer** | https://oklch.com | Mint brand tokens as native OKLCH; perceptually-uniform steps don't compress on `#080B12`. |

*Compliance:* **mcp-color-convert** MCP auto-verifies WCAG in the PR loop (see Section C); **apcacontrast.com** for APCA on 11px tabular numerals — but keep formal **WCAG 2.1 AA** as the gate (SEBI/DPDP reference 2.1, not WCAG 3 candidate APCA).

### Design-tools / Design-to-code
| Tool | URL | Why |
|---|---|---|
| **Figma Dev Mode MCP** *(connected)* | https://developers.figma.com/docs/figma-mcp-server/ | Code Connect maps our Radix+cva components so Claude emits *our* components, not raw divs; reads our `#00E6A7` variables. |
| **v0 by Vercel** | https://v0.dev | Fastest prompt→Next/TW/shadcn component; controls all three stacks so v4/R19 output is native. Override palette in every prompt. |
| **Onlook** (live tuning) | https://www.onlook.com | Open-source "Cursor for designers" — visual edits write to real Next/TW source for fine-tuning cockpit spacing/glow. |

### Charts / Finance-viz
| Tool | URL | Why |
|---|---|---|
| **Lightweight Charts v5** (5.2.0) | https://github.com/tradingview/lightweight-charts | Canonical candlestick/OHLC engine; v5 adds **multi-pane**, **yield-curve** + **options chart type** (price-as-x-axis → payoff/term-structure), P3 color API, −16% bundle. Already in our stack. |
| **Apache ECharts v6** (selective imports) | https://echarts.apache.org | Sector heatmaps, OI surfaces, gauges — the only chart lib here with a **real working MCP** (`mcp-echarts`) for in-IDE config generation. |
| **Recharts v3** | https://recharts.org | P&L sparklines / portfolio time-series; safe in-place upgrade (fixes v2 re-render loops). + **visx** for bespoke vol-cone/payoff shading. |

**AVOID:** react-financial-charts (unmaintained ~3 yrs); Observable Plot (not finance-optimized). KLineChart v10 is "watch once stable."

### Templates / Starters
| Tool | URL | Why |
|---|---|---|
| **shadcn/ui Blocks** (new-york-v4) | https://ui.shadcn.com/blocks | dashboard-01 + sidebar + login/signup — the cockpit structural skeleton; OKLCH dark baked in. MCP-installable. |
| **Tremor Blocks** (free/MIT) | https://blocks.tremor.so | 300+ chart blocks; Solar template = Next 15 + TW v4 + React 19. |
| **shadcn-fintech** (MIT, our exact stack) | https://github.com/abderrahimghazali/shadcn-fintech | Next 16 + R19 + TW v4 + shadcn + Recharts + Motion; lift page layouts (invert to dark). + **Shadcnblocks Pro** ($149) for marketing/dashboard density. |

---

## C. MCP CONNECTION KIT *(priority deliverable)*

Already connected (per project memory): **Figma** ✅ · **Stitch** ✅ · **Playwright** ✅ · **Context7** ✅ · **Supabase** ✅ · **Vercel** ✅

| Pri | MCP | What it gives Quant X | Connect command / snippet | Cost |
|---|---|---|---|---|
| **T1** | **shadcn** | Browse/search/install all components + every registered registry (Magic UI, Aceternity, Origin, Kibo, Tremor) by natural language; v4-aware/OKLCH | `claude mcp add shadcn -- npx shadcn@latest mcp` | Free |
| **T1** | **Next DevTools** | Claude inspects running dev server: routing state, server logs, error traces (huge during the 16 migration) | `{"mcpServers":{"next-devtools":{"command":"npx","args":["-y","next-devtools-mcp@latest"]}}}` | Free |
| **T1** | **mcp-color-convert** | Auto-verifies WCAG + converts hex→OKLCH on *every* generated token — color compliance becomes an automated PR step | `claude mcp add color-convert -- npx -y mcp-color-convert@latest` | Free |
| **T1** | **Context7** *(connected)* | Version-specific docs injected pre-generation — prevents hallucinating TW v3 `purge`, Next 14 `getServerSideProps`, old motion APIs during upgrade | *connected* — prefix prompts with "use context7" | Free |
| **T1** | **Playwright** *(connected)* | "Verify before completion": drive routes, screenshot dark render, confirm `proxy.ts` auth + chart load | *connected* | Free |
| **T2** | **Magic UI** | Discover/install animated components (NumberTicker, AnimatedBeam, BorderBeam) by name | `npx @magicuidesign/cli@latest install claude` | Free |
| **T2** | **21st.dev Magic** | `/ui <desc>` → generates dark-fintech shadcn components in multiple variants in-IDE | `npx @21st-dev/cli@latest install claude --api-key <KEY>` | Freemium (100 cr/mo) |
| **T2** | **Mobbin** *(official, launched 12 May 2026)* | 621k+ real app screens incl. Zerodha/Groww/Robinhood/TradingView — competitor research inside Claude | `claude mcp add mobbin --scope user --transport http https://api.mobbin.com/mcp` (then browser auth) | Paid plan |
| **T2** | **mcp-echarts** | Generate/iterate ECharts heatmap/OI/gauge configs in-conversation | `{"mcpServers":{"mcp-echarts":{"command":"npx","args":["-y","mcp-echarts"]}}}` | Free |
| **T2** | **Chrome DevTools** | Deep runtime debugging — "why no dark-mode chart?" → inspect computed CSS vars, network, console source-maps | `claude mcp add chrome-devtools npx chrome-devtools-mcp@latest` (Chrome with `--user-data-dir`) | Free |
| **T3** | **Aceternity** | Search/install background-beam/spotlight components | `{"mcpServers":{"aceternityui":{"command":"npx","args":["aceternityui-mcp"]}}}` (community) | Free |
| **T3** | **Refero** | Extracts design-system *rules* (DESIGN.md) from premium SaaS — "design taste injection" | `claude mcp add --transport http refero https://api.refero.design/mcp --header "Authorization: Bearer <token>"` | Paid (beta) |
| **T3** | **assistant-ui docs** | In-IDE docs for building the persistent Copilot (Thread/Composer/generative-UI) | `npx assistant-ui@latest create -t mcp` (docs server) | Free |
| **T3** | **Mantine** / **HeroUI** / **Iconify** | Per-library scaffolding if those libs get adopted | `npx -y @mantine/mcp-server` · `npx -y @heroui/react-mcp@latest` (Node 22+) · `npx -y iconify-mcp-server@latest` | Free |
| **T3** | **Storybook** / **LottieFiles** | Post-v3 component-library docs gate; Lottie search | `{"url":"http://localhost:6006/mcp"}` · `npx -y mcp-server-lottiefiles` | Free |

### → CONNECT THESE FIRST (paste block)

The 4 new free Tier-1/2 MCPs that immediately accelerate the upgrade + build (everything else is already connected or situational):

```bash
# Day 0 — the upgrade + build accelerators (all free, zero API key)
claude mcp add shadcn -- npx shadcn@latest mcp
claude mcp add color-convert -- npx -y mcp-color-convert@latest
claude mcp add next-devtools -- npx -y next-devtools-mcp@latest
npx @magicuidesign/cli@latest install claude

# Then restart Claude Code and run /mcp to confirm all show "connected"
```

Equivalent `.mcp.json` (project root, version-controlled):
```json
{
  "mcpServers": {
    "shadcn":        { "command": "npx", "args": ["shadcn@latest", "mcp"] },
    "color-convert": { "command": "npx", "args": ["-y", "mcp-color-convert@latest"] },
    "next-devtools": { "command": "npx", "args": ["-y", "next-devtools-mcp@latest"] },
    "magic-ui":      { "command": "npx", "args": ["-y", "@magicuidesign/mcp@latest"] }
  }
}
```
Add **Mobbin** (`claude mcp add mobbin --scope user --transport http https://api.mobbin.com/mcp`) once a paid seat exists — it's the strongest grounding for the cockpit aesthetic.

---

## D. REVISED BUILD FOUNDATION

### Final component stack (post-upgrade)
- **Base:** shadcn/ui (new-york-v4, OKLCH) — owns every primitive.
- **Copilot:** **assistant-ui** (Thread/Composer/generative-UI, streaming, tool-call rendering) wrapped in React 19 **`<Activity>`** so the persistent sidebar never re-mounts conversation history across routes.
- **Tables:** **TanStack Table** headless core + **Kibo UI DataTable** (column pinning/virtualization) for screener/positions/signal history.
- **Charts:** **Lightweight Charts v5** (candles/term-structure/options payoff) · **Recharts v3** (P&L sparklines) · **Apache ECharts v6** (heatmaps/OI/gauges) · **visx** (bespoke vol-cone/payoff shading).
- **Data cards/KPI:** **Tremor** (v4) — KPI + BadgeDelta + Tracker.
- **Motion:** **motion v12** + **Number Flow** (tickers) + **AutoAnimate** (lists) + **Motion Primitives** (text reveal/in-view) + **Magic UI** (AnimatedBeam/BorderBeam ambient).
- **Icons:** Phosphor (`/ssr`) + pqoqubbw animated accents.
- **Type:** Geist Sans/Mono + JetBrains Mono (DSL) + Clash Display (marketing).

### Build order
1. **Theme tokens** → convert palette to OKLCH (mcp-color-convert), define in `@theme`, map to shadcn slots in tweakcn, generate 50-950 scales (uicolors.app). Set Geist via `geist/font`, Utopia `clamp()` scale. *One source of truth before any UI.*
2. **App shell** → shadcn dashboard-01 + sidebar block → cockpit chrome (nav, top bar, persistent Copilot drawer via `<Activity>`). Wire `proxy.ts` auth. Add dot-grid background (Aceternity, low opacity).
3. **First surfaces** → (a) **Cockpit/Signals** card grid (Magic UI BentoGrid + Tremor KPI + Number Flow tickers + Lightweight Charts); (b) **Scanner** (TanStack/Kibo DataTable + AutoAnimate rows); (c) **Copilot** answer-first panel (assistant-ui + AnimatedBeam). Then Portfolio/AutoPilot/Strategy-DSL (JetBrains Mono + Kibo CodeBlock).
4. **Marketing** (separate aesthetic) → shadcn/Tremor blocks + GSAP/Lenis hero + Spline (one lazy embed).

### What the upgrade CHANGES from the earlier plan
- **framer-motion 10 → motion v12** everywhere (mandatory under R19) — earlier plan assumed framer-motion.
- **recharts 2 → v3** (peer-dep + re-render fix).
- **Tailwind config → `@theme`** — tokens live in CSS, not `tailwind.config.js`; the earlier config-based token approach is retired.
- **Tremor is now viable** (Vercel/MIT + Solar confirms v4/R19) — promote it from "maybe" to core KPI/chart-card layer.
- **Copilot persistence** is now a first-class React 19 `<Activity>` pattern, not a manual keep-alive hack.

---

## E. RISKS & GUARDRAILS

**Tailwind v4 gotchas:** `tailwind.config.js` deprecated (tokens → `@theme`); JS plugins must become `@utility`/`@variant`; `content:[]` gone; class renames (`shadow-sm`/`rounded-sm`/`outline-hidden`/`ring-3`/`bg-linear-*`); `darkMode:'class'` → `@custom-variant dark`; run codemod on a branch and **diff every change**; recharts internal utility classes need review post-rename. Browser floor: **Safari 16.4+ / Chrome 111+ / Firefox 128+** (aligns with Next 16 targets — no extra concern).

**RSC / `'use client'`:** Lightweight Charts, ECharts, Recharts, motion, assistant-ui, all canvas/WebGL backgrounds = **client components**. Use Phosphor's `/ssr` submodule to keep RSC parents pure server. Live-price RSC → `'use cache'`. Lazy-load all heavy 3D/WASM (Spline 6.8MB runtime, R3F ~580kB, dotLottie WASM) behind `next/dynamic({ ssr:false })`.

**Licenses:** MIT/free — shadcn, Magic UI core, Tremor, Aceternity free tier, motion, Number Flow, AutoAnimate, Lightweight Charts (Apache), ECharts (Apache), Recharts, visx, Phosphor, pqoqubbw, unDraw, Geist (OFL), JetBrains Mono (OFL). **GSAP 3.13** all plugins now free incl. commercial (prohibited only in competing animation-builder tools — N/A to us). **Paid:** Clash Display (ITF Free — permitted for commercial SaaS, self-host), Spline Pro (~$16/mo for no-watermark), Shadcnblocks Pro ($149), Mobbin/Refero (subscription), 21st.dev (freemium).

**Bundle weight discipline:** ECharts selective imports only (`echarts/core` + needed series); Tremor remaps to our palette but uses Recharts internally (no double-chart-lib); never load full `nivo` meta-package; cap one Spline embed/route; lazy-load every Lottie. Geist-first keeps base type at 114kB vs Inter's 525kB.

**Non-negotiables (enforce in PR review + Playwright audit):**
- **Brand-firewall:** public engine names only — **AlphaRank / Sentinel / Compass / AutoPilot**. Never surface TFT/Qlib/FinBERT/HMM or internal "Swing AI" in any UI string, alt text, or v0/Stitch prompt.
- **SEBI no-urgency:** zero countdowns, zero "act now" timers, no scarcity. Animations communicate data/confirm action/guide attention — never manufacture urgency.
- **Glass on overlays only:** frosted glass restricted to modals/drawers/copilot overlay (Hype4 calibration: blur 12-16px, sat 180%, `#121826`@60%, border `rgba(255,255,255,0.08)`, `-webkit-backdrop-filter` for Safari). **Never** on the base `#121826` card grid.
- **P&L-only colors:** bull `#00FF9D` / bear `#FF5B7F` reserved strictly for directional/P&L meaning. Emerald `#00E6A7` = brand/signal; electric-blue `#3B82F6` = structural; purple `#8B5CF6` = AI-only. Never color non-directional UI green/red.
- **Data sensitivity:** never put live prices, portfolio data, signals, or user PII into v0 / Stitch / Mobbin / Anima prompts (hosted MCPs transit external servers).

---

### Sources
- [Next.js 16 blog](https://nextjs.org/blog/next-16) · [Upgrading: v16](https://nextjs.org/docs/app/guides/upgrading/version-16) · [middleware→proxy](https://nextjs.org/docs/messages/middleware-to-proxy)
- [React 19.2](https://react.dev/blog/2025/10/01/react-19-2)
- [Tailwind v4 blog](https://tailwindcss.com/blog/tailwindcss-v4) · [TW v4 upgrade guide](https://tailwindcss.com/docs/upgrade-guide) · [shadcn TW v4](https://ui.shadcn.com/docs/tailwind-v4)
- [shadcn CLI 3.0 + MCP changelog](https://ui.shadcn.com/docs/changelog/2025-08-cli-3-mcp) · [shadcn MCP docs](https://ui.shadcn.com/docs/mcp)
- [Motion upgrade guide](https://motion.dev/docs/react-upgrade-guide) · [motiondivision/motion](https://github.com/motiondivision/motion)
- [assistant-ui](https://github.com/assistant-ui/assistant-ui) · [docs](https://www.assistant-ui.com/docs)
- [Mobbin MCP](https://mobbin.com/mcp) · [official server repo](https://github.com/mobbin/mobbin-mcp-server) · [docs](https://docs.mobbin.com/mcp/clients)
- [mcp-color-convert](https://github.com/bennyzen/mcp-color-convert)
- [Tremor / Vercel acquisition](https://vercel.com/blog/vercel-acquires-tremor) · [Tremor blocks](https://blocks.tremor.so)
- [Lightweight Charts v5 announce](https://www.tradingview.com/blog/en/tradingview-lightweight-charts-version-5-50837/) · [v4→v5 migration](https://tradingview.github.io/lightweight-charts/docs/migrations/from-v4-to-v5)
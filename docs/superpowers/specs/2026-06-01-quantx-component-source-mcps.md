All the data I need is in the task itself — this is a synthesis/compilation task, not a research task. Let me produce the document directly.

# Quant X — UI-Component-Source MCP Servers (definitive list)

Stack target: **Next 16 / React 19 / Tailwind v4 / shadcn**. Dark AI-trading-OS cockpit. All commands below are verified from the scout data.

---

## 1. Ranked master table — REAL component-source MCPs

Tier key: **T1** = connect now (free or near-free, directly serves this stack) · **T2** = useful (worth adding for specific surfaces) · **T3** = situational (paid/niche/heavier setup).

| Tier | Name | What it gives | Connect command | Off/Comm | Cost |
|---|---|---|---|---|---|
| **T1** | **shadcn MCP** (meta-registry) | Browse/search/install components, blocks, templates from the core registry **and any namespaced third-party registry** in `components.json`. ~7 tools. One MCP fronts Magic UI, Aceternity, Origin, Kibo, ReUI, Cult, Kokonut, Skiper, ShadcnSpace. | `pnpm dlx shadcn@latest mcp init --client claude` | Official | Free (MIT) |
| **T1** | **21st.dev Magic** | `/ui <prompt>` → production-ready React/shadcn/Tailwind components from a curated lib (dark mode, data-viz, trading patterns). Writes files directly. SVG logo search + site-clone (Pro). | `npx @21st-dev/cli@latest install claude --api-key <YOUR_21ST_DEV_API_KEY>` | Official | Freemium (100 free credits/mo; Pro $20/mo) |
| **T1** | **Magic UI MCP** | Official MCP for magicui.design: 50+ animated React/Tailwind/Framer-Motion components — beams, orbits, particles, sparkles, number tickers, marquee, terminal, bento, device mocks. Marketing + cockpit hero / data-viz flourishes. | `npx @magicuidesign/cli@latest install claude` | Official | Free (MIT) |
| **T1** | **HeroUI MCP** (`@heroui/react-mcp`) | 6 tools: list/docs/source `.tsx`/styles/theme tokens. **HeroUI v3 architected ground-up on Tailwind v4 + React 19** — best native fit for the stack. Light/dark token export. | `claude mcp add heroui-react -- npx -y @heroui/react-mcp@latest` | Official | Free |
| **T2** | **ReUI MCP** (via shadcn MCP) | 1000+ components / 68 categories shown in real dashboard layouts; 17 primitives not in core shadcn — **Data Grid, Kanban, Filters, Sortable, Timeline, Stepper, Tree**. Strong for data-heavy trading dashboards. | (use shadcn MCP) `pnpm dlx shadcn@latest mcp init` | Official | Free (Pro tier exists) |
| **T2** | **Aceternity UI MCP** (`aceternityui-mcp`) | Community MCP proxying the Aceternity registry: 3D card, spotlight, background-beams, moving borders, wavy/aurora backgrounds, tracing beams. High-drama cockpit hero / signal-card backgrounds. | `claude mcp add aceternityui -- npx aceternityui-mcp` | Community | Free (MIT) |
| **T2** | **Eldora UI MCP** (`@eldoraui/mcp`) | Official MCP: getUIComponents/getButtons/getBackgrounds. Animated marquee, card-flip-hover, text anim, device mockups, bento grids. Same client support as Magic UI. | `npx @eldoraui/cli@latest install claude` | Official | Free (MIT) |
| **T2** | **Mantine MCP** (`@mantine/mcp-server`) | 4 tools over all Mantine v9 components + hooks. **v9 requires React 19.2+**; coexists with Tailwind v4 via separate CSS layers. Released Mar 2026. | `claude mcp add mantine -- npx -y @mantine/mcp-server` | Official | Free |
| **T2** | **Material UI MCP** (`@mui/mcp`) | 6 tools over 50+ MUI components + **MUI X data-grid/charts**. React 19 support active; Tailwind v4 layered via CSS layer ordering. Good if you want a heavyweight data-grid. | `claude mcp add mui-mcp -- npx -y @mui/mcp@latest` | Official | Free |
| **T2** | **ShadcnSpace MCP** | Extended block registry beyond core shadcn. 5 tools (listBlocks/listComponents/getBlockInstall/searchBlocks/listInstalledBlocks). Compose full page layouts from prebuilt TS blocks. | `npx shadcnspace-cli install claude` | Community | Freemium |
| **T2** | **Cult UI MCP** (via shadcn MCP) | Cult UI registry through shadcn MCP: Dynamic Island, dock, texture cards, spinning text, direction-aware hover, bg-animate (~50 free). Cockpit micro-interactions / autopilot status. | (use shadcn MCP) `pnpm dlx shadcn@latest mcp init --client claude` | Official | Free tier (Pro license for blocks) |
| **T2** | **Animate UI MCP** (via shadcn MCP) | shadcn-registry-compatible animated lib (`@animate-ui` namespace): sliding numbers, fade transitions, spring effects. Tasteful Motion animations. | (use shadcn MCP) `pnpm dlx shadcn@latest mcp init` | Community | Free (MIT) |
| **T3** | **v0 by Vercel** (official MCP) | Generates full Next.js + React + Tailwind + shadcn views/layouts from text or image. Scaffold entire cockpit views. Requires paid v0 plan + API key. | (`.mcp.json` only) `npx mcp-remote https://mcp.v0.dev --header "Authorization: Bearer ${V0_API_KEY}"` | Official | Paid (free $5/mo credits; API needs paid plan) |
| **T3** | **Lovable MCP** | 40+ tools: create/remix/deploy full Lovable projects, agent UI iteration, pull code diffs, SQL. Spin up standalone prototypes then pull code back. | `claude mcp add --transport http lovable "https://mcp.lovable.dev"` then `/mcp auth` | Official | Paid (Pro ~$25/mo; no free MCP) |
| **T3** | **Chakra UI MCP** (`@chakra-ui/react-mcp`) | list/props/example/design-tokens + v2→v3 migration. Chakra v3 is **Panda CSS-based, not Tailwind-native** — off-stack; include only if you adopt Chakra. | `claude mcp add chakra-ui -- npx -y @chakra-ui/react-mcp` | Official | Free |
| **T3** | **Flowbite MCP** (`flowbite-mcp`) | 60+ Tailwind-native components (HTML/React/Svelte), Figma→code (needs `FIGMA_ACCESS_TOKEN`), theme generator. React 19 compat unconfirmed; better for utility/HTML output. | `claude mcp add flowbite -- npx -y flowbite-mcp` | Official | Free |
| **T3** | **Radix UI MCP** (`@gianpieropuleo/radix-mcp-server`) | Radix Themes + Primitives + Colors source from GitHub (rate-limited; add `GITHUB_TOKEN`). Primitives are unstyled → fully Tailwind-composable. | `claude mcp add radix-ui -- npx -y @gianpieropuleo/radix-mcp-server@latest` | Community | Free |
| **T3** | **Superdesign MCP** | 5 tools (generate/iterate/extract_system/list/gallery). Returns structured specs for Claude's own LLM — dark-theme component iteration **without burning 21st.dev credits**. Needs local clone + build. | `git clone …/superdesign-mcp-claude-code && npm i && npm run build` then `claude mcp add --scope user superdesign /abs/path/dist/index.js` | Community | Free (MIT) |
| **T3** | **Mantine Community MCP** (`@hakxel/mantine-ui-server`) | Docs lookup + component gen + theme config for Mantine. **Superseded by official `@mantine/mcp-server`** — listed for completeness only; prefer official. | `claude mcp add mantine-community -- npx @hakxel/mantine-ui-server` | Community | Free |

### Inspiration / asset MCPs (support component work, not component sources themselves — keep handy)
| Tier | Name | Gives | Connect | Cost |
|---|---|---|---|---|
| T2 | **Iconify MCP** | 200k+ icons / 200+ sets with exact import strings (kills icon hallucination). | `claude mcp add iconify -- npx -y iconify-mcp-server@latest` | Free |
| T2 | **svgl MCP** | 660+ brand/framework SVG logos — broker marks (Zerodha/Upstox/Angel), exchanges, fintech. | `claude mcp add svgl -- npx -y @modelcontextprotocol/server-svgl` | Free |
| T3 | **LottieFiles Creator MCP** | Build/edit Lottie animations by prompt — signal-load spinners, autopilot status, alert micro-interactions. | `claude mcp add lottiefiles-creator -- npx -y @lottiefiles/creator-mcp@latest` | Freemium |
| T3 | **Figma Dev Mode MCP** | Figma frame → React/CSS via Code Connect mapping. | `claude mcp add --transport http figma https://mcp.figma.com/mcp` | Free (beta) / paid Figma seat |
| T3 | **Mobbin / Refero MCP** | 600k+ / 150k+ real shipped-app screens for dark-fintech reference patterns (paid). | Mobbin: `claude mcp add mobbin --scope user --transport http https://api.mobbin.com/mcp` · Refero: `claude mcp add --transport http refero https://api.refero.design/mcp --header "Authorization: Bearer <token>"` | Paid |

### Dropped — NO real outbound MCP (use registry / copy-in / browser only)
- **Bolt.new / bolt.diy** — consumes MCP, never exposes one. No way to pull components into Claude Code.
- **Polymet.ai** — browser-only AI designer; GitHub push handoff, no MCP.
- **tldraw make-real** — web app, outputs HTML not React; its MCP only draws diagrams, no component code.
- **Onlook** — visual React editor; consumes MCPs, doesn't expose one.
- **Park UI** — Panda CSS (off-stack), no MCP, copy-source only.
- **Magic UI / Aceternity / Origin UI / Kibo UI / Tremor / ReUI / Cult UI / Kokonut / Skiper as libraries** — install **through the shadcn MCP** or `shadcn add` (see §2). (Magic UI, Aceternity, Cult, ReUI, Eldora additionally have/are reachable via dedicated MCPs above.)

---

## 2. The shadcn meta-registry play — one MCP, many catalogs

Add these namespaces to `frontend/components.json` under `"registries"`, then the **single shadcn MCP** (or `shadcn add`) can pull from all of them by natural language. This is the highest-leverage move for this stack.

```jsonc
// frontend/components.json
{
  "registries": {
    "@magicui":   "https://magicui.design/r/{name}.json",
    "@aceternity":"https://ui.aceternity.com/registry/{name}.json",
    "@originui":  "https://originui.com/r/{name}.json",
    "@kibo-ui":   "https://www.kibo-ui.com/r/{name}.json",
    "@reui":      "https://reui.io/r/{style}/{name}.json",
    "@cult-ui":   "https://cult-ui.com/r/{name}.json",
    "@kokonutui": "https://kokonutui.com/r/{name}.json"
  }
}
```

| Library | Registry add URL / namespace | Install example | Best for Quant X | Stack notes | Cost |
|---|---|---|---|---|---|
| **Magic UI** | `@magicui` → `https://magicui.design/r/{name}.json` | `npx shadcn@latest add @magicui/globe` (or `add "https://magicui.design/r/bento-grid"`) | Marketing pages, cockpit hero, number-ticker P&L | Framer Motion | Free (MIT) |
| **Aceternity UI** | `@aceternity` → `https://ui.aceternity.com/registry/{name}.json` | `npx shadcn@latest add @aceternity/3d-marquee` | Cockpit backgrounds, signal cards | Tailwind + Framer Motion | Freemium |
| **Origin UI** | `@originui` → `https://originui.com/r/{name}.json` | `npx shadcn@latest add @originui/comp-01` | **Strategy-builder forms, settings** (advanced inputs, multi-selects, date pickers, command palettes) | **Built for Tailwind v4** | Free (631+) |
| **Kibo UI** | `@kibo-ui` → `https://www.kibo-ui.com/r/{name}.json` | `npx shadcn@latest add @kibo-ui/gantt` | **Strategy builder (Kanban/timeline/Gantt)**, signal detail (diff viewer, code block) | shadcn-composable | Free (MIT) |
| **ReUI** | `@reui` → `https://reui.io/r/{style}/{name}.json` | `pnpm dlx shadcn@latest add @reui/filters` | **Dashboard scaffolding, KPI blocks, data grid, filters** | Radix + Base UI variants | Free (1000+) |
| **Cult UI** | `@cult-ui` → `https://cult-ui.com/r/{name}.json` | `npx shadcn@beta add @cult-ui/texture-card` | Cockpit micro-interactions, autopilot status (Dynamic Island/dock) | Framer Motion | Free (MIT) |
| **Kokonut UI** | `@kokonutui` → `https://kokonutui.com/r/{name}.json` | `npx shadcn@latest add @kokonutui/particle-button` | Signal cards, scanner animations | **Requires Tailwind v4** + Motion | Free |
| **Skiper UI** | per-component (no clean namespace) | `npx shadcn add @skiper-ui/skiper40` or `add "https://shadcnregistry.com/r/skiper-ui/skiper54"` | Scroll-driven hero sections | Tailwind + Radix | Freemium (24 free / 54 paid) |
| **Tremor** | **NOT a shadcn registry — npm package** | `npm i @tremor/react` (or copy-paste from tremor.so) | **Portfolio charts, P&L sparklines, regime tracker, KPI cards** (Area/Bar/Donut/Line/Spark/Scatter, BadgeDelta) | **Tailwind v4 required**; Apache 2.0, Vercel-owned | Free |

> Note: Tremor is the one charting pick here that is **not** installable through the shadcn MCP — install it as the `@tremor/react` npm package separately.

---

## 3. Connect-now paste block

Already connected (given earlier) — **do not re-run**: `shadcn`, `magic-ui`.

Recommended additions for the Quant X dark cockpit, in priority order:

```bash
# T1 — native Tailwind v4 + React 19 component framework (closest stack fit)
claude mcp add heroui-react -- npx -y @heroui/react-mcp@latest

# T2 — animated effects for hero/cockpit backgrounds
claude mcp add aceternityui -- npx aceternityui-mcp
npx @eldoraui/cli@latest install claude

# T2 — data-heavy dashboard primitives (Kanban/DataGrid/Timeline/Filters) via shadcn registries
#   no extra MCP — add namespaces to frontend/components.json (see §2),
#   then drive installs through the already-connected shadcn MCP.

# T2 — extended page-layout blocks
npx shadcnspace-cli install claude

# Asset MCPs — icons + broker/exchange logos (kills hallucinated imports)
claude mcp add iconify -- npx -y iconify-mcp-server@latest
claude mcp add svgl    -- npx -y @modelcontextprotocol/server-svgl

# T3 — full-view scaffolding (PAID; only if a v0 plan is purchased)
#   add to .mcp.json (no clean one-liner); needs V0_API_KEY in shell profile:
#   { "mcpServers": { "v0": { "command": "npx",
#     "args": ["mcp-remote","https://mcp.v0.dev","--header","Authorization: Bearer ${V0_API_KEY}"] } } }

# T3 — free design iteration without burning 21st.dev credits (local build required)
#   git clone https://github.com/jonthebeef/superdesign-mcp-claude-code
#   cd superdesign-mcp-claude-code && npm install && npm run build
#   claude mcp add --scope user superdesign /absolute/path/to/superdesign-mcp-claude-code/dist/index.js
```

Mantine and MUI MCPs are deliberately left out of the connect-now block — they're off the primary shadcn/Tailwind path (add only if you intentionally pull in a Mantine/MUI data-grid):

```bash
# Optional, only if adopting that framework:
# claude mcp add mantine -- npx -y @mantine/mcp-server          # Mantine v9 (React 19.2+)
# claude mcp add mui-mcp -- npx -y @mui/mcp@latest              # MUI X data-grid/charts
```

---

## 4. Notes — Tailwind v4 / React 19, reliability, dark-fintech picks

**Tailwind v4 / React 19 compatibility flags**
- **Native fit (no friction):** **HeroUI v3** (built ground-up on Tailwind v4 static CSS + React 19), **Origin UI** (built for Tailwind v4), **Kokonut UI** (requires Tailwind v4 + Motion), **Tremor** (Tailwind v4 required). These are first-class for this stack.
- **shadcn / 21st.dev / Magic UI / Aceternity / Cult / ReUI / Eldora / Animate UI** — all shadcn+Tailwind based, fine on v4. Magic UI / Aceternity / Cult / Kokonut lean hard on Framer Motion; budget for that bundle weight on the cockpit.
- **React 19 caveats:** Mantine needs **v9 (React 19.2+)** — pin v9. MUI React 19 support is active but tracked-in-progress; Flowbite React 19 compat is **unconfirmed in docs**. Treat those three as "verify before relying on."
- **Off-stack (Panda CSS, not Tailwind):** **Chakra v3** and **Park UI** use Panda CSS; **Radix Themes** ship their own CSS tokens. Only Radix **Primitives** (unstyled) are cleanly Tailwind-composable. Skip Chakra/Park unless you deliberately change CSS engines.

**Community-MCP reliability cautions**
- `aceternityui-mcp`, `iconify-mcp-server`, `svgl`, `superdesign` are community packages — pin versions and expect occasional registry/format drift; they're convenience proxies over public registries, so the `shadcn add <URL>` fallback always works if an MCP breaks.
- **Radix UI MCP** fetches source from GitHub and is rate-limited (60 req/hr unauthenticated) — set `GITHUB_TOKEN` for 5000/hr if you use it.
- **Mantine Community MCP** is superseded — always prefer the official `@mantine/mcp-server`.
- Prefer **official MCPs over community ones** wherever both exist (shadcn, Magic UI, HeroUI, Eldora are official; Aceternity's only MCP is community).

**Prefer for dark fintech (decisive picks)**
- **Charts / P&L / KPIs:** **Tremor** (sparklines, BadgeDelta, regime tracker) — npm, not via shadcn MCP.
- **Data-heavy dashboards (positions, scanner, strategy lists):** **ReUI** (DataGrid, Filters, Kanban, Timeline, Stepper) + **Kibo UI** (Gantt, Kanban, diff viewer, code block).
- **Forms (strategy builder, settings, auth):** **Origin UI** (Tailwind v4 advanced inputs, command palette) — pairs with **HeroUI** for native v4 primitives.
- **Hero / marketing / cockpit drama:** **Magic UI** (number tickers, beams, bento) + **Aceternity** (spotlight, background-beams, aurora) + **Cult UI** (Dynamic Island / autopilot status micro-interactions).
- **Generation workhorses:** **21st.dev Magic** for one-shot dark trading components (watch credit burn → use **Superdesign** free for iteration); **v0** only if a paid plan exists, for full-view scaffolds.
- **Assets:** **Iconify** (consistent dark-theme icons) + **svgl** (Zerodha/Upstox/Angel + exchange logos).

**Bottom line:** Connect **shadcn MCP + 21st.dev Magic + Magic UI MCP + HeroUI MCP** as the core four; wire the seven shadcn registry namespaces (§2) so that one MCP unlocks Magic UI / Aceternity / Origin / Kibo / ReUI / Cult / Kokonut; add **Iconify + svgl** for assets; install **Tremor** via npm for charts. Everything else is situational.
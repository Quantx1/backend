-- PR-F: Strategy registry table per v2 design spec §7.3 + §7.5
--
-- One row per saved strategy (template or user-created). Lifecycle:
--   draft   → only authored, never run
--   backtest → run through ml/backtest/engine.py on demand
--   paper   → scheduler runs every N minutes, executes against
--             paper_trades table (no broker)
--   live    → scheduler runs every N minutes, executes against
--             trades table via TradeExecutionService (real broker)
--   paused  → live or paper, suspended without losing config
--
-- The state machine transitions are enforced in code
-- (backend/ai/strategy/registry.py), NOT via DB triggers — keep the
-- DB dumb so we can audit + tune transition rules in Python.
--
-- RLS: SELECT/INSERT/UPDATE/DELETE on own rows; SUPER_ADMIN full access.

BEGIN;

CREATE TABLE IF NOT EXISTS public.user_strategies (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    -- Identity + display
    name              TEXT NOT NULL,
    description       TEXT,
    template_slug     TEXT,                 -- non-null if derived from a strategy_catalog row
    -- The DSL document, stored verbatim as JSONB. Pydantic owns the schema.
    dsl               JSONB NOT NULL,
    -- Lifecycle state machine
    status            TEXT NOT NULL DEFAULT 'draft'
                      CHECK (status IN ('draft', 'backtest', 'paper', 'live', 'paused', 'archived')),
    -- Capital allocation when running paper/live
    capital_allocated NUMERIC(15, 2) DEFAULT 0 CHECK (capital_allocated >= 0),
    -- Last backtest summary (denormalized so /strategies list is one query)
    last_backtest     JSONB DEFAULT '{}'::jsonb,
    -- Running statistics (updated by scheduler tick)
    runtime_stats     JSONB DEFAULT '{}'::jsonb,
    -- Ownership origin: user-created via Studio, or seeded from library
    source            TEXT NOT NULL DEFAULT 'user'
                      CHECK (source IN ('user', 'studio', 'template')),
    -- Audit timestamps
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at       TIMESTAMPTZ,
    deployed_at       TIMESTAMPTZ,
    paused_at         TIMESTAMPTZ,
    archived_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS user_strategies_user_id_idx
    ON public.user_strategies (user_id);

CREATE INDEX IF NOT EXISTS user_strategies_status_idx
    ON public.user_strategies (status)
    WHERE status IN ('paper', 'live');

-- Updated-at trigger so we don't rely on app code remembering
CREATE OR REPLACE FUNCTION public._user_strategies_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS user_strategies_updated_at ON public.user_strategies;
CREATE TRIGGER user_strategies_updated_at
    BEFORE UPDATE ON public.user_strategies
    FOR EACH ROW EXECUTE FUNCTION public._user_strategies_set_updated_at();

-- RLS — single user can only see their own rows. SUPER_ADMIN bypasses.
ALTER TABLE public.user_strategies ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS user_strategies_own_select ON public.user_strategies;
CREATE POLICY user_strategies_own_select
    ON public.user_strategies
    FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS user_strategies_own_modify ON public.user_strategies;
CREATE POLICY user_strategies_own_modify
    ON public.user_strategies
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Strategy execution log — append-only audit of every scheduler tick
-- that emitted (or considered emitting) an order from a strategy.
CREATE TABLE IF NOT EXISTS public.strategy_executions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id     UUID NOT NULL REFERENCES public.user_strategies(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    tick_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          TEXT,
    -- 'entry' / 'exit' / 'no_action'
    decision        TEXT NOT NULL CHECK (decision IN ('entry', 'exit', 'no_action')),
    -- Engine + rule trace for debugging — which condition fired, with values
    trace           JSONB DEFAULT '{}'::jsonb,
    -- Linked trade row if decision != no_action
    trade_id        UUID,
    -- Live or paper at tick time (matches user_strategies.status)
    mode            TEXT NOT NULL CHECK (mode IN ('paper', 'live'))
);

CREATE INDEX IF NOT EXISTS strategy_executions_strategy_id_idx
    ON public.strategy_executions (strategy_id, tick_at DESC);

ALTER TABLE public.strategy_executions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS strategy_executions_own_select ON public.strategy_executions;
CREATE POLICY strategy_executions_own_select
    ON public.strategy_executions
    FOR SELECT
    USING (auth.uid() = user_id);

COMMIT;

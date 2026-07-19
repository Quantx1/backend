-- ============================================================================
-- PR 134 — AutoPilot decision columns + autopilot_enabled flag
-- ============================================================================
-- Extends ``auto_trader_runs`` (PR 69) with the structured fields the
-- F4 service needs to record per-tick:
--
--   target_weights JSONB  — { SYMBOL: weight }, post-overlay
--   diagnostics    JSONB  — { vix_level, vix_exposure_cap, applied_scale, var_95_inr, ... }
--   status         TEXT   — 'decided' | 'executed' | 'blocked' | 'failed'
--
-- Also adds the per-user enrolment + execution-pause flags AutoPilot
-- queries on every rebalance tick (Step 1 §F4 Elite enrolment + N8 user
-- kill switch).
--
-- Idempotent — safe to re-run.
-- ============================================================================

-- auto_trader_runs additions ────────────────────────────────────────────────
ALTER TABLE public.auto_trader_runs
    ADD COLUMN IF NOT EXISTS target_weights JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS diagnostics    JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS status         TEXT NOT NULL DEFAULT 'decided';

-- user_profiles enrolment + kill switch ─────────────────────────────────────
ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS autopilot_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    -- PR 155 — dry-run mode. AutoPilot decides + records on this user
    -- but never places live orders. Default TRUE on first enrolment so
    -- users always start in shadow mode.
    ADD COLUMN IF NOT EXISTS autopilot_dry_run BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS live_trading_paused BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS live_trading_paused_until TIMESTAMPTZ;

-- Find Elite + AutoPilot-enabled users in one shot.
CREATE INDEX IF NOT EXISTS user_profiles_autopilot_idx
    ON public.user_profiles (tier, autopilot_enabled)
    WHERE autopilot_enabled = TRUE;

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_29_pr134_autopilot_decisions',
    'Add target_weights/diagnostics/status to auto_trader_runs + autopilot_enabled/live_trading_paused on user_profiles — PR 134.'
) ON CONFLICT (version) DO NOTHING;

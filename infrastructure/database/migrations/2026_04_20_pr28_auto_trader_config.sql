-- ============================================================================
-- PR 28 — Auto-trader (F4) dashboard state columns
-- ============================================================================
-- /auto-trader (Elite) reads + writes four fields:
--
--   auto_trader_enabled (from PR 2) ― master on/off.
--   auto_trader_config  (JSONB, NEW) ― user-editable safety rails:
--       {
--         "risk_profile": "conservative"|"moderate"|"aggressive",
--         "max_position_pct": 7.0,     -- cap per single holding
--         "daily_loss_limit_pct": 2.0, -- circuit-break below this
--         "max_concurrent_positions": 12,
--         "allow_fno": false
--       }
--   auto_trader_last_run_at (TIMESTAMPTZ, NEW) ― last FinRL-X rebalance tick.
--   kill_switch_active (from complete_schema.sql) ― emergency pause.
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS auto_trader_config JSONB
        DEFAULT '{
            "risk_profile": "moderate",
            "max_position_pct": 7.0,
            "daily_loss_limit_pct": 2.0,
            "max_concurrent_positions": 12,
            "allow_fno": false
        }'::jsonb;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS auto_trader_last_run_at TIMESTAMPTZ;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr28_auto_trader_config',
    'Add user_profiles.auto_trader_config (JSONB) + auto_trader_last_run_at — PR 28 Elite auto-trader dashboard state.'
) ON CONFLICT (version) DO NOTHING;

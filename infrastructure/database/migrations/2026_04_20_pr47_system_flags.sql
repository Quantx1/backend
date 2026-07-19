-- ============================================================================
-- PR 47 — N9 Admin Command Center expansions
-- ============================================================================
-- Adds ``system_flags`` — a tiny key-value store for platform-wide
-- ops toggles. First flag: ``global_kill_switch`` (bool). When active,
-- every order-placing path must halt (trade execution, auto-trader,
-- scheduler retrain jobs) until an admin clears it.
--
-- Design choices:
--   * Single flag per row — simple JSON ``value`` so future flags can
--     be a boolean, number, or small config blob without a schema change.
--   * ``updated_by`` captures audit trail when admins flip something.
--   * Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.system_flags (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_by UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed the initial flag so reads never hit a missing row.
INSERT INTO public.system_flags (key, value, description)
VALUES (
    'global_kill_switch',
    '{"active": false, "reason": null}'::jsonb,
    'Halts all order-placing paths (manual live trades, AutoPilot, scheduler). Flip on in emergencies.'
) ON CONFLICT (key) DO NOTHING;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr47_system_flags',
    'Add system_flags table + seed global_kill_switch — PR 47 N9 admin command-center.'
) ON CONFLICT (version) DO NOTHING;

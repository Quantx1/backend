-- ============================================================================
-- PR 49 — Admin audit log
-- ============================================================================
-- One row per admin-initiated mutation. Every endpoint in admin_routes.py
-- that writes state (user suspend/ban/reset, ml retrain, scan trigger,
-- kite refresh, global kill switch) writes one row here via
-- ``log_admin_action(...)``.
--
-- Schema:
--   actor_id    — the admin user_id (FK to user_profiles; SET NULL on delete
--                 so we don't lose audit history when an admin's profile is
--                 removed, though their UUID stays)
--   action      — snake_case verb ("user_suspend", "tier_reset",
--                 "ml_retrain_trigger", "kill_switch_flip", …)
--   target_type — 'user' | 'tier' | 'ml_model' | 'scheduler_job' |
--                 'system_flag' | 'payment' | 'other'
--   target_id   — free-form id of the affected entity (user_id, job_id, …)
--   payload     — full JSON blob of the request body / relevant params so ops
--                 can reconstruct exactly what happened
--   ip_address  — when available from the request (nullable)
--   user_agent  — nullable
--
-- Retention:
--   Not deleted by app code. Consider a nightly prune of rows >365d if
--   storage pressure shows up later.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.admin_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    actor_id UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL,
    actor_email TEXT,                      -- denormalized for forensics
    action TEXT NOT NULL,
    target_type TEXT NOT NULL DEFAULT 'other'
        CHECK (target_type IN (
            'user','tier','ml_model','scheduler_job',
            'system_flag','payment','signal','other'
        )),
    target_id TEXT,
    payload JSONB DEFAULT '{}'::jsonb,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS admin_audit_log_actor_idx
    ON public.admin_audit_log (actor_id, created_at DESC);
CREATE INDEX IF NOT EXISTS admin_audit_log_target_idx
    ON public.admin_audit_log (target_type, target_id, created_at DESC);
CREATE INDEX IF NOT EXISTS admin_audit_log_action_idx
    ON public.admin_audit_log (action, created_at DESC);
CREATE INDEX IF NOT EXISTS admin_audit_log_recent_idx
    ON public.admin_audit_log (created_at DESC);

-- RLS: only super-admins read; service-role writes. We don't ship RLS
-- policies here because admin-only views go through the admin JWT path
-- (get_admin_user) which uses service-role for queries anyway. Enabling
-- RLS without a SELECT policy would block reads — skip for now.

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr49_admin_audit_log',
    'Add admin_audit_log table + indexes — PR 49 N9 ops forensics.'
) ON CONFLICT (version) DO NOTHING;

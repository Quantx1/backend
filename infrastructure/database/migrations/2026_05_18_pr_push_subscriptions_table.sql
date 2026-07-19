-- ============================================================================
-- PR push-subs — push_subscriptions table (2026-05-18)
-- ============================================================================
-- Backend writes/reads push_subscriptions in 6 places but the table was
-- never created in the base schema:
--
--   backend/api/push_routes.py:62, 95     (upsert / delete)
--   backend/services/realtime.py:749,     (fan-out reads + 410 cleanup)
--                                  763, 773, 785
--
-- Without this table /api/push/subscribe returns 500 from Supabase and no
-- browser push notification ever reaches a user. Web Push is a paid-tier
-- retention feature so this counts as a launch blocker.
--
-- Schema matches the column set that push_routes.upsert() writes:
--   user_id, endpoint, p256dh, auth, user_agent
-- plus id/created_at/updated_at infra columns. UNIQUE(user_id, endpoint)
-- backs the on_conflict="user_id,endpoint" upsert clause.
--
-- RLS posture (per 2026-05-12 lockdown): RLS enabled, no policies. The
-- frontend never reads this table directly — every interaction goes
-- through /api/push/* which authenticates against service_role. Service
-- role bypasses RLS so backend writes continue unaffected; anon/authed
-- clients are blocked at the row level.
--
-- Idempotent — safe to re-run.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.push_subscriptions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    endpoint    TEXT NOT NULL,
    p256dh      TEXT NOT NULL,
    auth        TEXT NOT NULL,
    user_agent  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, endpoint)
);

CREATE INDEX IF NOT EXISTS push_subscriptions_user_idx
    ON public.push_subscriptions (user_id);

CREATE INDEX IF NOT EXISTS push_subscriptions_endpoint_idx
    ON public.push_subscriptions (endpoint);

-- Reuse the shared updated_at trigger function if it exists; otherwise
-- create a minimal local one. The base schema defines
-- update_updated_at_column() but we guard against fresh-install order.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column'
    ) THEN
        CREATE FUNCTION public.update_updated_at_column()
        RETURNS TRIGGER AS $fn$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $fn$ LANGUAGE plpgsql;
    END IF;
END$$;

DROP TRIGGER IF EXISTS update_push_subscriptions_updated_at
    ON public.push_subscriptions;

CREATE TRIGGER update_push_subscriptions_updated_at
    BEFORE UPDATE ON public.push_subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

ALTER TABLE public.push_subscriptions ENABLE ROW LEVEL SECURITY;

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_05_18_pr_push_subscriptions_table',
    'Add push_subscriptions table for Web Push fan-out (push_routes + realtime).'
) ON CONFLICT (version) DO NOTHING;

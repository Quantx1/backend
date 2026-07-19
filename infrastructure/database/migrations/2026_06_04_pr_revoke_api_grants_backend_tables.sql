-- ============================================================================
-- 2026-06-04 — Revoke anon/authenticated API grants on backend-only objects.
--
-- Supabase advisor (WARN): "Public/Signed-In Users Can See Object in GraphQL
-- Schema". anon/authenticated held table-level grants on backend-only objects,
-- so they appeared in the auto-generated PostgREST/GraphQL API surface (object
-- NAMES discoverable; data already blocked by RLS).
--
-- We revoke ONLY where it is provably safe: tables that have RLS enabled AND no
-- policy permitting anon/authenticated/public — i.e. anon/authenticated already
-- get zero rows, so removing their grant changes no behaviour, it just drops
-- them from the API surface. Tables WITH an anon/authenticated policy
-- (user_profiles, strategy_catalog, strategy_backtests, user_strategy_deployments,
-- and the other intentionally-exposed tables) are left untouched — the app
-- reads those via the Supabase API. The two orphan (security_invoker) views are
-- also revoked since nothing reads them.
--
-- Dynamic + idempotent: only ever targets service-locked tables.
-- ============================================================================

DO $$
DECLARE r record;
BEGIN
    FOR r IN
        SELECT c.relname
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r' AND c.relrowsecurity
          AND NOT EXISTS (
              SELECT 1 FROM pg_policies p
              WHERE p.schemaname = 'public' AND p.tablename = c.relname
                AND p.roles && ARRAY['anon','authenticated','public']::name[]
          )
    LOOP
        EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated;', r.relname);
    END LOOP;

    -- Orphan security-invoker views (unreferenced by app code).
    EXECUTE 'REVOKE ALL ON public.v_user_stats   FROM anon, authenticated';
    EXECUTE 'REVOKE ALL ON public.v_today_signals FROM anon, authenticated';
END $$;

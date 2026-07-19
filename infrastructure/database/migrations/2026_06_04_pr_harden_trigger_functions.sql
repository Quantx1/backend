-- ============================================================================
-- 2026-06-04 — Harden trigger functions (Supabase advisor: WARN).
--
--   * function_search_path_mutable (x4): handle_new_user,
--     _user_strategies_set_updated_at, update_updated_at, update_user_stats had
--     no pinned search_path → search-path injection risk (esp. the SECURITY
--     DEFINER one). All four bodies use fully-qualified public.* refs (or none),
--     so SET search_path = '' is safe and is the recommended fix.
--   * "Public can execute SECURITY DEFINER function": handle_new_user is
--     SECURITY DEFINER and EXECUTE-able by anon/authenticated/public. It is a
--     trigger fired on auth.users INSERT — the trigger runs it regardless of
--     direct EXECUTE grants, so revoking EXECUTE is safe and closes the finding.
--     (Same for the other trigger fns — they are never called directly.)
--
-- Idempotent.
-- ============================================================================

ALTER FUNCTION public.handle_new_user()                 SET search_path = '';
ALTER FUNCTION public._user_strategies_set_updated_at() SET search_path = '';
ALTER FUNCTION public.update_updated_at()               SET search_path = '';
ALTER FUNCTION public.update_user_stats()               SET search_path = '';

REVOKE EXECUTE ON FUNCTION public.handle_new_user()                 FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION public._user_strategies_set_updated_at() FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION public.update_updated_at()               FROM anon, authenticated, PUBLIC;
REVOKE EXECUTE ON FUNCTION public.update_user_stats()               FROM anon, authenticated, PUBLIC;

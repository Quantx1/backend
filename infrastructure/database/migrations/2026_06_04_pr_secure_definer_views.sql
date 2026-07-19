-- ============================================================================
-- 2026-06-04 — Fix SECURITY DEFINER views (Supabase advisor: ERROR).
--
-- public.v_user_stats and public.v_today_signals were created as SECURITY
-- DEFINER views owned by `postgres`, and anon/authenticated hold SELECT on
-- them. A SECURITY DEFINER view runs with the OWNER's privileges, so those
-- roles could read through the views and BYPASS RLS on the underlying tables —
-- exposing every user's capital / total_pnl / win-rate (v_user_stats) and the
-- full signal feed (v_today_signals) to anyone with the anon key.
--
-- Neither view is referenced by application code (legacy orphans from
-- FULL_SETUP.sql). Switching them to SECURITY INVOKER makes them respect the
-- CALLER's RLS: anon/authenticated now see only what RLS permits (nothing,
-- for cross-user rows), while the backend (service_role) keeps full access
-- since service_role bypasses RLS regardless.
--
-- Idempotent: SET (security_invoker = on) is a safe no-op if already set.
-- ============================================================================

ALTER VIEW public.v_user_stats    SET (security_invoker = on);
ALTER VIEW public.v_today_signals SET (security_invoker = on);

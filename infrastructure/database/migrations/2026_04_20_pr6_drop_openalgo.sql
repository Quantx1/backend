-- ============================================================================
-- PR 6 — Drop OpenAlgo columns
-- ============================================================================
-- OpenAlgo was removed from the product (2026-04-18 decision reversal).
-- Direct OAuth to Zerodha + Upstox + Angel One is the canonical broker path;
-- OpenAlgo self-hosting UX is too technical for the target retail user.
--
-- This migration cleans up the two columns added in
-- 2026_04_19_pr2_v1_ai_stack.sql (user_profiles.openalgo_url /
-- openalgo_api_key). Idempotent — IF EXISTS guards let this re-run safely.
-- ============================================================================

ALTER TABLE public.user_profiles
    DROP COLUMN IF EXISTS openalgo_url;
ALTER TABLE public.user_profiles
    DROP COLUMN IF EXISTS openalgo_api_key;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr6_drop_openalgo',
    'Drop user_profiles.openalgo_url + openalgo_api_key — OpenAlgo removed from product.'
) ON CONFLICT (version) DO NOTHING;

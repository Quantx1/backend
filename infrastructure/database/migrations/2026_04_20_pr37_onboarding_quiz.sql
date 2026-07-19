-- ============================================================================
-- PR 37 — Onboarding risk-profile quiz (N5)
-- ============================================================================
-- First-login wizard scores the user on 5 axes (experience, risk
-- tolerance, horizon, loss tolerance, goal) and derives
-- ``risk_profile`` (conservative | moderate | aggressive).
--
-- ``risk_profile`` already exists on user_profiles (conservative/moderate/
-- aggressive). We add:
--   * onboarding_completed   — first-login redirect gate
--   * onboarding_completed_at
--   * risk_quiz_answers      — full JSONB snapshot for audit + re-runs
--   * recommended_tier       — product recommendation from the quiz
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS onboarding_completed_at TIMESTAMPTZ;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS risk_quiz_answers JSONB DEFAULT '{}'::jsonb;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS recommended_tier TEXT
        CHECK (recommended_tier IS NULL OR recommended_tier IN ('free', 'pro', 'elite'));

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr37_onboarding_quiz',
    'Add onboarding_completed / risk_quiz_answers / recommended_tier — PR 37 N5 risk-profile quiz.'
) ON CONFLICT (version) DO NOTHING;

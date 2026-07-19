-- ============================================================================
-- PR 42 — N12 Referral loop (virality + acquisition)
-- ============================================================================
-- Two-sided reward: referrer + referred each get +1 month Pro credit
-- the first time the referred user completes a paid upgrade.
--
-- Tables:
--   * user_profiles.referral_code      — 8-char base32 unique code
--   * user_profiles.referred_by        — fk to the referrer (nullable, set at signup)
--   * user_referrals                   — one row per invite attempt
--
-- State machine for user_referrals.status:
--     'pending'    — invite shared, not yet signed up
--     'signed_up'  — referred user created their account
--     'rewarded'   — referred user paid, both sides got 1 month credit
--     'expired'    — 90 days elapsed without signup (soft-expires via cron)
--
-- The ``claim_type`` column is forward-looking: today only ``first_paid``
-- exists; future campaigns might add ``paper_milestone``, ``streak_30d``…
--
-- Idempotent.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS referral_code TEXT UNIQUE;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS referred_by UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS referral_credit_months INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS public.user_referrals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    referrer_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    referred_user_id UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL,
    referred_email TEXT,
    claim_type TEXT NOT NULL DEFAULT 'first_paid' CHECK (claim_type IN ('first_paid')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','signed_up','rewarded','expired')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signed_up_at TIMESTAMPTZ,
    rewarded_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS user_referrals_referrer_idx
    ON public.user_referrals (referrer_id, created_at DESC);
CREATE INDEX IF NOT EXISTS user_referrals_referred_user_idx
    ON public.user_referrals (referred_user_id)
    WHERE referred_user_id IS NOT NULL;

-- RLS: a user sees only their own referrals as either side.
ALTER TABLE public.user_referrals ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    DROP POLICY IF EXISTS "users see own referrals" ON public.user_referrals;
    CREATE POLICY "users see own referrals" ON public.user_referrals
        FOR SELECT USING (
            auth.uid() = referrer_id OR auth.uid() = referred_user_id
        );
EXCEPTION WHEN OTHERS THEN NULL; END $$;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr42_referrals',
    'Add user_referrals + referral_code/referred_by/referral_credit_months on user_profiles — PR 42 N12 referral loop.'
) ON CONFLICT (version) DO NOTHING;

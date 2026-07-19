-- ============================================================================
-- PR 60 — F12 WhatsApp digest (Pro upgrade driver)
-- ============================================================================
-- Adds OTP-verification + digest-opt-in columns on top of the existing
-- whatsapp_phone / whatsapp_verified pair (shipped in PR 2).
--
-- Why a separate `whatsapp_digest_enabled` flag (not reusing
-- whatsapp_verified): verification is the mechanical "we know this
-- number is real" state; digest-enabled is the marketing consent — two
-- different user actions. Enable stays OFF by default so a verified
-- user isn't auto-subscribed.
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS whatsapp_otp TEXT;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS whatsapp_otp_expires_at TIMESTAMPTZ;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS whatsapp_otp_attempts SMALLINT DEFAULT 0;

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS whatsapp_digest_enabled BOOLEAN DEFAULT FALSE;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr60_whatsapp_digest',
    'Add whatsapp_otp + attempts + digest_enabled — PR 60 F12 WhatsApp digest.'
) ON CONFLICT (version) DO NOTHING;

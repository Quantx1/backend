-- ============================================================================
-- PR 40 — N11 Alerts Studio (granular per-event channel routing)
-- ============================================================================
-- Before: single ``notifications_enabled`` blanket flag + channel-connected
-- booleans scattered across user_profiles.
--
-- After: a single ``alert_preferences`` JSONB keyed by event-type whose
-- value is a {channel: bool} map. Backend + realtime bus consult this
-- matrix before firing anything.
--
-- Event catalogue (rows):
--   new_signal           — SwingLens / TickPulse fires a fresh signal
--   signal_triggered     — live price crosses signal entry
--   target_hit           — position closes at target
--   sl_hit               — position closes at stop-loss
--   regime_change        — RegimeIQ transitions bull/sideways/bear
--   debate_completed     — Counterpoint returns verdict
--   earnings_upcoming    — EarningsScout flags announcement ≤14d
--   weekly_review        — Sunday N10 narrative ready
--   auto_trade_executed  — AutoPilot fires an order
--   price_alert          — user-configured price crosses
--
-- Channels (columns):
--   push · telegram · whatsapp · email
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.user_profiles
    ADD COLUMN IF NOT EXISTS alert_preferences JSONB
        DEFAULT '{
            "new_signal":          {"push": true,  "telegram": true,  "whatsapp": false, "email": false},
            "signal_triggered":    {"push": true,  "telegram": false, "whatsapp": false, "email": false},
            "target_hit":          {"push": true,  "telegram": true,  "whatsapp": false, "email": true},
            "sl_hit":              {"push": true,  "telegram": true,  "whatsapp": false, "email": true},
            "regime_change":       {"push": true,  "telegram": true,  "whatsapp": false, "email": false},
            "debate_completed":    {"push": true,  "telegram": false, "whatsapp": false, "email": false},
            "earnings_upcoming":   {"push": false, "telegram": true,  "whatsapp": false, "email": false},
            "weekly_review":       {"push": false, "telegram": false, "whatsapp": true,  "email": true},
            "auto_trade_executed": {"push": true,  "telegram": true,  "whatsapp": false, "email": false},
            "price_alert":         {"push": true,  "telegram": true,  "whatsapp": false, "email": false}
        }'::jsonb;

-- Log this migration.
INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_20_pr40_alert_preferences',
    'Add alert_preferences JSONB — PR 40 N11 Alerts Studio granular event×channel matrix.'
) ON CONFLICT (version) DO NOTHING;

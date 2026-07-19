-- ============================================================================
-- PR 109 — watchlist alert debounce columns
-- ============================================================================
-- The watchlist table already stores `alert_price_above` and
-- `alert_price_below`, but no backend job evaluated them. PR 109 wires
-- a 5-minute scheduler scan that fires a `price_alert` Notification
-- when LTP crosses the threshold.
--
-- Without debounce, the same row would fire every 5 minutes forever
-- once price clears the threshold. We track the last fire time + last
-- direction so:
--   * The same direction re-arms only after the user changes the
--     threshold (alert_price_above set to NULL or a new value).
--   * Cross in the opposite direction (above → below) is treated as
--     a new event and fires immediately.
--
-- Idempotent — safe to re-run.
-- ============================================================================

ALTER TABLE public.watchlist
    ADD COLUMN IF NOT EXISTS alert_last_fired_at TIMESTAMPTZ;

ALTER TABLE public.watchlist
    ADD COLUMN IF NOT EXISTS alert_last_fired_direction TEXT
        CHECK (alert_last_fired_direction IS NULL OR alert_last_fired_direction IN ('above', 'below'));

INSERT INTO public.schema_migrations (version, description)
VALUES (
    '2026_04_25_pr109_watchlist_alert_debounce',
    'Add alert_last_fired_at + alert_last_fired_direction — PR 109 watchlist price-alert scanner.'
) ON CONFLICT (version) DO NOTHING;

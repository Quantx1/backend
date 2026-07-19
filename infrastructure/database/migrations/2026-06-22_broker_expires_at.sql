-- 2026-06-22 — broker_connections.expires_at: token expiry tracking.
-- /broker/connections + freshness checks read this; previously absent → endpoint errored.
ALTER TABLE public.broker_connections
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

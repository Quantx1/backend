-- 2026-06-22 — allow 'expired' broker status (token-refresh failure path).
ALTER TABLE public.broker_connections
    DROP CONSTRAINT IF EXISTS broker_connections_status_check;
ALTER TABLE public.broker_connections
    ADD CONSTRAINT broker_connections_status_check
    CHECK (status IN ('connected', 'disconnected', 'error', 'expired'));

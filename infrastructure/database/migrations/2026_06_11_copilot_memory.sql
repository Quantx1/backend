-- AIL v2 — per-user rolling conversation MEMORY (Copilot).
-- Idempotent. Service-role only (RLS on, no anon/auth policies).

CREATE TABLE IF NOT EXISTS public.copilot_memory (
    user_id          uuid PRIMARY KEY,
    summary          text NOT NULL DEFAULT '',
    turns_summarized integer NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.copilot_memory ENABLE ROW LEVEL SECURITY;

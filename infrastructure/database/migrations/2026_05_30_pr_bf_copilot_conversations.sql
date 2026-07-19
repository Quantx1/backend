-- PR-BF.2 — Saved Copilot conversations
--
-- The /copilot chat surface (PR-AF) renders turns from React state
-- only — refresh and the history is gone. Saving conversations
-- server-side lets users come back to a thread, share it, or have
-- the runner reference past context.
--
-- Schema:
--   copilot_conversations — one row per chat thread
--   copilot_messages      — one row per turn (user OR assistant)
--
-- Idempotent — IF NOT EXISTS on table + indexes.

CREATE TABLE IF NOT EXISTS copilot_conversations (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL,
  title       TEXT,                         -- derived from first user msg
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  -- Soft-delete so the user can restore; cleaned by a periodic job.
  archived_at TIMESTAMPTZ,
  -- For pinning + free-form tags later.
  metadata    JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_copilot_conversations_user_updated
  ON copilot_conversations(user_id, updated_at DESC)
  WHERE archived_at IS NULL;

CREATE TABLE IF NOT EXISTS copilot_messages (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL
                  REFERENCES copilot_conversations(id) ON DELETE CASCADE,
  -- One of 'user' or 'assistant'. (System prompts aren't persisted —
  -- they live in the orchestrator code and shouldn't drift in DB rows.)
  role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content         TEXT NOT NULL,
  -- For the assistant turn: which tools the orchestrator called +
  -- the intent classifier's verdict + token counts. Optional.
  tools_used      JSONB DEFAULT '[]'::jsonb,
  trace           JSONB DEFAULT '[]'::jsonb,
  intent          TEXT,
  refused         BOOLEAN DEFAULT FALSE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_copilot_messages_conversation
  ON copilot_messages(conversation_id, created_at ASC);

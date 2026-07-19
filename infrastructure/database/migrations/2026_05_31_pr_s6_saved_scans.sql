-- PR-S6 — Saved Scans + Alerts
--
-- Lets a trader save a screener configuration once, have it auto-run on
-- a schedule during market hours, and get notified when new symbols
-- match. Two tables:
--
--   saved_scans          one row per saved configuration
--   saved_scan_alerts    one row per fire — captures the symbol diff
--                        (new hits since last run) for notification

-- ─── saved_scans ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS saved_scans (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL,

  -- User-facing label, e.g. "Power Setup on Nifty 500 Banking"
  name            TEXT NOT NULL CHECK (char_length(name) BETWEEN 1 AND 120),

  -- Which scanners to combine (confluence mode) OR which single
  -- scanner to run. Single-element list = single scanner.
  scanner_ids     INTEGER[] NOT NULL CHECK (array_length(scanner_ids, 1) >= 1),

  -- Optional filters applied BEFORE confluence
  universe        TEXT NOT NULL DEFAULT 'nifty500'
                  CHECK (universe IN ('nifty50','nifty100','nifty500','nse_all')),
  sectors         TEXT[] DEFAULT '{}',         -- canonical sector names
  min_hits        INTEGER NOT NULL DEFAULT 1 CHECK (min_hits BETWEEN 1 AND 10),

  -- Schedule: how often to re-run while market is open.
  --   * 'hourly' → every hour 9:30-15:30 IST Mon-Fri
  --   * 'open_close' → 9:30 + 15:25 only (most economical)
  --   * 'every_15min' → market hours, every 15 min (most aggressive)
  --   * 'manual' → user fires it from the UI; no cron
  schedule        TEXT NOT NULL DEFAULT 'hourly'
                  CHECK (schedule IN ('hourly','open_close','every_15min','manual')),

  -- How to notify on new hits
  notify_channels TEXT[] NOT NULL DEFAULT '{push}'
                  CHECK (notify_channels <@ ARRAY['push','email','whatsapp','telegram']),

  -- Lifecycle
  enabled         BOOLEAN NOT NULL DEFAULT TRUE,
  last_run_at     TIMESTAMPTZ,
  last_hit_symbols TEXT[] DEFAULT '{}',        -- previous run's matches (for diff)
  last_hit_count  INTEGER DEFAULT 0,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saved_scans_user
  ON saved_scans(user_id, created_at DESC);

-- For the cron sweep: pick all enabled scans due for a re-run.
CREATE INDEX IF NOT EXISTS idx_saved_scans_enabled_schedule
  ON saved_scans(enabled, schedule, last_run_at)
  WHERE enabled = TRUE;


-- ─── saved_scan_alerts ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS saved_scan_alerts (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scan_id         UUID NOT NULL REFERENCES saved_scans(id) ON DELETE CASCADE,
  user_id         UUID NOT NULL,

  -- Which symbols are NEW since the last run (the diff that triggered
  -- the alert). Not all symbols matched — only the additions, so users
  -- don't get re-alerted for the same hits every hour.
  new_symbols     TEXT[] NOT NULL,
  total_match_count INTEGER NOT NULL DEFAULT 0,

  -- Notification dispatch state
  notified        BOOLEAN NOT NULL DEFAULT FALSE,
  notify_error    TEXT,

  fired_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saved_scan_alerts_user_fired
  ON saved_scan_alerts(user_id, fired_at DESC);

CREATE INDEX IF NOT EXISTS idx_saved_scan_alerts_undelivered
  ON saved_scan_alerts(notified, fired_at)
  WHERE notified = FALSE;
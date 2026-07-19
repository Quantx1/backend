-- T1.1 (2026-05-31) — universal broker-side stop tracking.
--
-- Catastrophic-gap fix: previously execution.py only attempted GTT
-- placement for Zerodha and never recorded whether the broker
-- actually accepted the stop. Upstox + Angel positions ran with NO
-- broker-side protection. Now every position carries:
--   - stop_status     'placed' | 'unsupported' | 'failed' | 'unprotected'
--   - stop_broker_id  the GTT/SL-M order id at the broker
--   - target_broker_id  for Zerodha OCO two-leg GTTs
--   - stop_method     'gtt_oco' | 'gtt_single' | 'sl_m'
--   - stop_error      diagnostic on non-'placed' status
--
-- All columns nullable + default NULL so existing positions are
-- unaffected. New positions opened by execution.py:execute_live_trade
-- populate them via stop_orchestrator.StopResult.to_position_patch().

ALTER TABLE positions
  ADD COLUMN IF NOT EXISTS stop_status      TEXT,
  ADD COLUMN IF NOT EXISTS stop_broker_id   TEXT,
  ADD COLUMN IF NOT EXISTS target_broker_id TEXT,
  ADD COLUMN IF NOT EXISTS stop_method      TEXT,
  ADD COLUMN IF NOT EXISTS stop_error       TEXT;

-- Enum-like guard. Use CHECK to keep insert paths honest without
-- needing a Postgres ENUM type (those are painful to evolve).
ALTER TABLE positions
  DROP CONSTRAINT IF EXISTS positions_stop_status_check;
ALTER TABLE positions
  ADD CONSTRAINT positions_stop_status_check
  CHECK (stop_status IS NULL OR stop_status IN (
    'placed', 'unsupported', 'failed', 'unprotected'
  ));

-- Operational index for "find all live positions without broker stop"
-- — this powers the alert that nudges users when their bot opened
-- a position the broker didn't accept the stop for.
CREATE INDEX IF NOT EXISTS positions_unprotected_idx
  ON positions (user_id, stop_status)
  WHERE is_active = TRUE AND stop_status IN ('unprotected', 'failed');

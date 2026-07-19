-- PR-AR — strategy safety columns
--
-- Adds the columns needed for:
--   1. Per-strategy day-loss circuit breaker
--   2. Audit trail for why a strategy was paused (manual / loss-breach /
--      reconciliation-error / etc.)
--   3. Daily counters that the breaker reads + the runner increments
--
-- Backward compatible (every ADD uses IF NOT EXISTS; pause_reason is
-- nullable so existing rows stay valid).

ALTER TABLE user_strategies
  ADD COLUMN IF NOT EXISTS max_daily_loss_pct DECIMAL(5, 2);
COMMENT ON COLUMN user_strategies.max_daily_loss_pct IS
  'Per-strategy day-loss breaker. When today''s realized + unrealized P&L on this strategy '
  'drops below this percent of capital deployed, the runner pauses the strategy and skips '
  'further entries. NULL = use the platform default (3%).';

ALTER TABLE user_strategies
  ADD COLUMN IF NOT EXISTS pause_reason TEXT
    CHECK (pause_reason IS NULL OR pause_reason IN
      ('manual', 'day_loss_breach', 'broker_error', 'kill_switch',
       'subscription_lapsed', 'reconciliation_failure'));

-- Reconciliation poller bookkeeping
ALTER TABLE trades
  ADD COLUMN IF NOT EXISTS last_reconciled_at TIMESTAMPTZ;
ALTER TABLE trades
  ADD COLUMN IF NOT EXISTS reconciliation_attempts INT DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_trades_pending_reconcile
  ON trades(status, last_reconciled_at)
  WHERE status IN ('pending', 'approved');

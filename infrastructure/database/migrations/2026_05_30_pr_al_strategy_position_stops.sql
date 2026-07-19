-- PR-AL — strategy_positions: persist stop_loss / target_1 / exit data
--
-- The strategy runner now sizes positions from the user's capital and
-- enforces a hard SL/target gate (price-based exit) BEFORE evaluating
-- the DSL exit condition. To remember stop/target between ticks we need
-- columns on strategy_positions to persist them at entry time, plus
-- exit_reason + exit_price for closed-position audit.
--
-- Safe to run repeatedly — uses IF NOT EXISTS guards on every ADD.

ALTER TABLE strategy_positions
  ADD COLUMN IF NOT EXISTS stop_loss   DECIMAL(15, 2);
ALTER TABLE strategy_positions
  ADD COLUMN IF NOT EXISTS target_1    DECIMAL(15, 2);
ALTER TABLE strategy_positions
  ADD COLUMN IF NOT EXISTS exit_reason TEXT
    CHECK (exit_reason IS NULL OR exit_reason IN ('stop_loss', 'target', 'dsl_exit', 'manual', 'time'));
ALTER TABLE strategy_positions
  ADD COLUMN IF NOT EXISTS exit_price  DECIMAL(15, 2);

-- Status index helps the runner's per-tick open-position lookup.
CREATE INDEX IF NOT EXISTS idx_strategy_positions_user_status
  ON strategy_positions(user_id, status)
  WHERE status = 'open';

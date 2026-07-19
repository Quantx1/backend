-- PR-AM — paper_trades.source column
--
-- Adds a ``source`` column so we can tell apart fills the user clicked
-- manually (``manual``) from fills the strategy runner placed on their
-- behalf (``user_strategy``). Useful for activity feeds + analytics
-- ("of your last 100 paper trades, 73 were strategy-fired").
--
-- Backward-compatible: column is nullable, no existing row needs a value.

ALTER TABLE paper_trades
  ADD COLUMN IF NOT EXISTS source TEXT;

CREATE INDEX IF NOT EXISTS idx_paper_trades_user_source_time
  ON paper_trades(user_id, source, executed_at DESC);

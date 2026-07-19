-- PR-AT — paper options trading tables
--
-- Multi-leg option positions don't fit the existing paper_positions
-- shape (single symbol, single avg_price). A bull call spread is
-- ONE position to the user (one P&L, one set of risk metrics) but
-- TWO legs in the broker's view (long ATM call + short OTM call).
--
-- Schema:
--   paper_option_positions  — one row per multi-leg position
--   paper_option_legs       — one row per leg (FK to position)
--   paper_option_trades     — entry + exit audit (combined position)
--
-- Both tables get user_id + created_at + status indexes for the
-- runner's open-position sweep and the /strategies/deployed query.

CREATE TABLE IF NOT EXISTS paper_option_positions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL,
  strategy_id     UUID,                        -- nullable: ad-hoc deploys allowed
  template_slug   TEXT,                         -- e.g. 'bull_call_spread'
  underlying      TEXT NOT NULL,                -- 'NIFTY', 'BANKNIFTY', 'RELIANCE', ...
  expiry_date     DATE NOT NULL,
  -- Net cost (paid for debit spreads, received for credit spreads).
  -- Positive = we paid premium; negative = we collected.
  net_premium     DECIMAL(15, 2) NOT NULL,
  -- For max loss / margin display on the deployed panel.
  max_profit      DECIMAL(15, 2),
  max_loss        DECIMAL(15, 2),
  -- Pre-trade BS-derived risk metrics; stored for the UI to show
  -- without recomputing every render.
  estimated_margin DECIMAL(15, 2),
  -- Mark-to-market refreshed by the MTM sweep.
  current_value   DECIMAL(15, 2),
  unrealized_pnl  DECIMAL(15, 2),
  realized_pnl    DECIMAL(15, 2),
  status          TEXT NOT NULL DEFAULT 'open'
                  CHECK (status IN ('open', 'closed', 'expired')),
  exit_reason     TEXT CHECK (exit_reason IS NULL OR exit_reason IN
                  ('target', 'stop_loss', 'manual', 'expiry', 'dsl_exit', 'time')),
  entry_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_marked_at  TIMESTAMPTZ,
  closed_at       TIMESTAMPTZ,
  metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_paper_option_positions_user_status
  ON paper_option_positions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_paper_option_positions_strategy
  ON paper_option_positions(strategy_id) WHERE strategy_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS paper_option_legs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  position_id     UUID NOT NULL
                  REFERENCES paper_option_positions(id) ON DELETE CASCADE,
  side            TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
  option_type     TEXT NOT NULL CHECK (option_type IN ('CE', 'PE')),
  strike          DECIMAL(15, 2) NOT NULL,
  expiry_date     DATE NOT NULL,
  lots            INT NOT NULL CHECK (lots > 0),
  lot_size        INT NOT NULL CHECK (lot_size > 0),
  entry_price     DECIMAL(15, 2) NOT NULL,      -- premium per share at open
  current_price   DECIMAL(15, 2),                -- premium per share at last mark
  exit_price      DECIMAL(15, 2),                -- premium per share at close
  metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_paper_option_legs_position
  ON paper_option_legs(position_id);

CREATE TABLE IF NOT EXISTS paper_option_trades (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL,
  position_id     UUID NOT NULL
                  REFERENCES paper_option_positions(id) ON DELETE CASCADE,
  -- 'open' = position created; 'close' = position closed.
  action          TEXT NOT NULL CHECK (action IN ('open', 'close')),
  -- For close trades, the realized PnL of the entire combined position.
  pnl             DECIMAL(15, 2),
  pnl_pct         DECIMAL(8, 4),
  -- 'manual' | 'user_strategy' | 'mtm_sweep_expired' | 'mtm_sweep_target' | ...
  source          TEXT DEFAULT 'manual',
  executed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_paper_option_trades_user_time
  ON paper_option_trades(user_id, executed_at DESC);

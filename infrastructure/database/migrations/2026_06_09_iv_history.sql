-- IV history — daily ATM implied vol per F&O underlying, accumulated forward
-- from option-chain snapshots. Feeds IV Rank / IV Percentile (Volatility
-- Intelligence). Reference/derived data; RLS-locked to the service role.
CREATE TABLE IF NOT EXISTS public.iv_history (
    symbol      TEXT NOT NULL,
    trade_date  DATE NOT NULL,
    atm_iv      NUMERIC NOT NULL,
    source      TEXT DEFAULT 'kite_snapshot',
    updated_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_iv_history_symbol_date
    ON public.iv_history (symbol, trade_date DESC);

ALTER TABLE public.iv_history ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.iv_history FROM anon, authenticated;

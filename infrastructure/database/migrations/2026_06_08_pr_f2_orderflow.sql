-- F2: order-flow EOD (free nselib sources, backend-only). Idempotent.
CREATE TABLE IF NOT EXISTS public.participant_oi_eod (
    date DATE NOT NULL, participant TEXT NOT NULL,          -- client|pro|fii|dii
    fut_long BIGINT, fut_short BIGINT,
    opt_call_long BIGINT, opt_call_short BIGINT,
    opt_put_long BIGINT, opt_put_short BIGINT,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, participant)
);
CREATE INDEX IF NOT EXISTS idx_participant_oi_date ON public.participant_oi_eod (date DESC);

CREATE TABLE IF NOT EXISTS public.fii_dii_flow_eod (
    date DATE NOT NULL, segment TEXT NOT NULL DEFAULT 'CASH',  -- CASH|FNO
    fii_buy NUMERIC, fii_sell NUMERIC, fii_net NUMERIC,
    dii_buy NUMERIC, dii_sell NUMERIC, dii_net NUMERIC,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, segment)
);

CREATE TABLE IF NOT EXISTS public.bulk_block_deals (
    date DATE NOT NULL, symbol TEXT NOT NULL,
    deal_type TEXT NOT NULL DEFAULT 'BULK',                 -- BULK|BLOCK
    client_name TEXT NOT NULL DEFAULT '', buy_sell TEXT NOT NULL DEFAULT '',
    qty BIGINT NOT NULL DEFAULT 0, price NUMERIC,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, symbol, deal_type, client_name, buy_sell, qty)
);
CREATE INDEX IF NOT EXISTS idx_bulk_block_date ON public.bulk_block_deals (date DESC, symbol);

CREATE TABLE IF NOT EXISTS public.short_selling (
    date DATE NOT NULL, symbol TEXT NOT NULL, qty BIGINT,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS public.fno_ban (
    date DATE NOT NULL, symbol TEXT NOT NULL,
    source TEXT DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (date, symbol)
);

ALTER TABLE public.participant_oi_eod ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fii_dii_flow_eod ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bulk_block_deals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.short_selling ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fno_ban ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.participant_oi_eod, public.fii_dii_flow_eod, public.bulk_block_deals, public.short_selling, public.fno_ban FROM anon, authenticated;

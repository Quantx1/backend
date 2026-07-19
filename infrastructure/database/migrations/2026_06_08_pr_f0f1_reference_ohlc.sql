-- F0+F1: reference spine + durable EOD OHLC store (free-sourced, backend-only).
-- Part B of complete_schema.sql auto-regenerates from this; do not hand-edit Part B.

-- F0.1 instrument master ------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.instruments (
    symbol TEXT NOT NULL, exchange TEXT NOT NULL DEFAULT 'NSE',
    instrument_type TEXT NOT NULL DEFAULT 'EQ',
    isin TEXT, series TEXT, name TEXT, sector TEXT, mcap_category TEXT,
    lot_size INT, tick_size NUMERIC,
    -- sentinels (NOT NULL DEFAULT) so the composite PK uses plain columns —
    -- Postgres forbids expressions (COALESCE) in a PRIMARY KEY, and the upsert
    -- on_conflict target must match these columns verbatim. Equities omit
    -- expiry/strike on insert → DEFAULTs apply → conflict resolves cleanly.
    strike NUMERIC NOT NULL DEFAULT 0,
    expiry DATE NOT NULL DEFAULT '1900-01-01',
    listing_date DATE, face_value NUMERIC, status TEXT DEFAULT 'active',
    source TEXT DEFAULT 'nselib', updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, exchange, instrument_type, expiry, strike)
);
CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON public.instruments (symbol);

-- F0.2 index constituents -----------------------------------------------------
CREATE TABLE IF NOT EXISTS public.index_constituents (
    index_name TEXT NOT NULL, symbol TEXT NOT NULL, weight NUMERIC,
    industry TEXT, source TEXT DEFAULT 'niftyindices', updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (index_name, symbol)
);

-- F0.3 corporate actions ------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.corporate_actions (
    symbol TEXT NOT NULL, ex_date DATE NOT NULL,
    action_type TEXT NOT NULL,
    ratio_from NUMERIC, ratio_to NUMERIC, value NUMERIC, details JSONB,
    source TEXT DEFAULT 'nselib',
    PRIMARY KEY (symbol, ex_date, action_type)
);
CREATE INDEX IF NOT EXISTS idx_corpactions_symbol ON public.corporate_actions (symbol, ex_date DESC);

-- F1 evolve orphan candles -> partitioned daily store-of-record ---------------
-- Idempotent + DATA-PRESERVING. The first version renamed candles aside and then
-- DROPPED it unconditionally, destroying every existing row (prod had 31,200).
-- This version: skip if already partitioned; otherwise rename the existing
-- (non-partitioned) table aside, build the partitioned store, COPY every row
-- forward, then drop the legacy copy. Safe to re-run.
DO $$
BEGIN
    -- Already partitioned (re-run / fresh install already migrated): nothing to do.
    IF EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'candles' AND c.relkind = 'p'
    ) THEN
        RAISE NOTICE 'candles already partitioned; skipping F1 migration';
        RETURN;
    END IF;

    -- Preserve any existing (non-partitioned) candles by renaming it aside.
    IF EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'candles'
    ) THEN
        ALTER TABLE public.candles RENAME TO candles_legacy;
    END IF;

    CREATE TABLE public.candles (
        stock_symbol TEXT NOT NULL, exchange TEXT NOT NULL DEFAULT 'NSE',
        interval TEXT NOT NULL DEFAULT '1d', timestamp TIMESTAMPTZ NOT NULL,
        open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
        delivery_qty BIGINT, delivery_pct NUMERIC, adj_close NUMERIC,
        source TEXT NOT NULL DEFAULT 'nselib', created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (stock_symbol, interval, timestamp)
    ) PARTITION BY RANGE (timestamp);
    CREATE TABLE IF NOT EXISTS public.candles_y2019 PARTITION OF public.candles FOR VALUES FROM ('2019-01-01') TO ('2020-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2020 PARTITION OF public.candles FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2021 PARTITION OF public.candles FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2022 PARTITION OF public.candles FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2023 PARTITION OF public.candles FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2024 PARTITION OF public.candles FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2025 PARTITION OF public.candles FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2026 PARTITION OF public.candles FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
    CREATE TABLE IF NOT EXISTS public.candles_y2027 PARTITION OF public.candles FOR VALUES FROM ('2027-01-01') TO ('2028-01-01');
    -- Catch-all: an out-of-range timestamp in the legacy copy (or a future
    -- ingest) lands here instead of failing the INSERT below.
    CREATE TABLE IF NOT EXISTS public.candles_default PARTITION OF public.candles DEFAULT;
    CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON public.candles (stock_symbol, timestamp DESC);

    -- Carry every existing row forward (legacy lacks the delivery and adj_close
    -- columns -> NULL; legacy 'id' PK is dropped for the composite key). De-dup
    -- via the new composite PK.
    IF EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'candles_legacy'
    ) THEN
        INSERT INTO public.candles
            (stock_symbol, exchange, interval, timestamp, open, high, low, close, volume, source, created_at)
        SELECT stock_symbol, exchange, interval, timestamp, open, high, low, close, volume, source, created_at
        FROM public.candles_legacy
        ON CONFLICT (stock_symbol, interval, timestamp) DO NOTHING;
        DROP TABLE public.candles_legacy;
    END IF;
END $$;

-- backend-only RLS (no PostgREST exposure) ------------------------------------
ALTER TABLE public.instruments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.index_constituents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.corporate_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.instruments, public.index_constituents, public.corporate_actions, public.candles FROM anon, authenticated;
-- RLS on the partition CHILDREN too: RLS on the partitioned parent does NOT
-- cover direct access to partitions, and Supabase exposes each partition to
-- PostgREST/GraphQL — without this they were anon/authenticated-readable.
ALTER TABLE public.candles_y2019   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2020   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2021   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2022   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2023   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2024   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2025   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2026   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_y2027   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candles_default ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.candles_y2019, public.candles_y2020, public.candles_y2021,
  public.candles_y2022, public.candles_y2023, public.candles_y2024,
  public.candles_y2025, public.candles_y2026, public.candles_y2027,
  public.candles_default FROM anon, authenticated;

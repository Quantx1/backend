-- 2026-07-21 — allow 'momentum' in signals.signal_type.
--
-- The v2 style engines (momentum + swing) bridge their daily books into
-- public.signals (sync_signals_table) so the signals page / detail /
-- debate surfaces stay live. The v1-era check constraint only allowed
-- ('swing','positional','intraday','btst') and rejected the momentum book.
ALTER TABLE public.signals DROP CONSTRAINT IF EXISTS signals_signal_type_check;
ALTER TABLE public.signals ADD CONSTRAINT signals_signal_type_check
    CHECK (signal_type IN ('swing', 'positional', 'intraday', 'btst', 'momentum'));

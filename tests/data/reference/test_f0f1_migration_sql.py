from pathlib import Path

SQL = Path("infrastructure/database/migrations/2026_06_08_pr_f0f1_reference_ohlc.sql").read_text()


def test_migration_defines_core_objects():
    for token in [
        "CREATE TABLE", "public.instruments", "public.index_constituents",
        "public.corporate_actions", "PARTITION BY RANGE", "candles",
    ]:
        assert token in SQL, f"missing {token}"


def test_candles_evolution_uses_composite_pk_and_partitions():
    assert "candles_legacy" in SQL
    assert "PRIMARY KEY (stock_symbol, interval, timestamp)" in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL

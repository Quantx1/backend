# tests/data/reference/test_f2_migration_sql.py
from pathlib import Path
SQL = Path("infrastructure/database/migrations/2026_06_08_pr_f2_orderflow.sql").read_text()

def test_defines_orderflow_tables():
    for t in ["public.participant_oi_eod", "public.fii_dii_flow_eod",
              "public.bulk_block_deals", "public.short_selling", "public.fno_ban"]:
        assert f"CREATE TABLE IF NOT EXISTS {t}" in SQL, t

def test_idempotent_and_no_expression_pk_and_rls():
    assert "COALESCE(" not in SQL                       # no expression PKs (F0+F1 lesson)
    assert SQL.count("ENABLE ROW LEVEL SECURITY") >= 5  # backend-only
    assert "REVOKE ALL" in SQL

from backend.data.reference.nse_fundamentals import map_fundamentals_row


def test_row_has_all_columns():
    r = map_fundamentals_row("X", {"fundamentals": {}, "promoter_holding": {}}, "2026-06-08")
    for col in ["snapshot_date", "symbol", "pe", "roe", "roce", "market_cap_cr",
                "book_value", "dividend_yield", "current_price", "debt_to_equity",
                "eps", "sales_growth", "profit_growth", "promoter_pct", "source"]:
        assert col in r

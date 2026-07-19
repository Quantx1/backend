from backend.data.reference.nse_fundamentals import map_fundamentals_row


def _data():
    # Mirrors the REAL screener_in.get_fundamentals() shape: fundamentals holds the
    # headline ratios; growth holds PERIOD-SUFFIXED keys (sales_growth_3_years, ...).
    return {
        "fundamentals": {"pe": 28.5, "roe": 18.2, "roce": 22.1, "market_cap_cr": 1850000,
                         "book_value": 1200, "dividend_yield": 0.4, "current_price": 2905},
        "growth": {"sales_growth_10_years": 11.0, "sales_growth_5_years": 12.5,
                   "sales_growth_3_years": 14.0, "sales_growth_ttm": 16.0,
                   "profit_growth_3_years": 18.0},
        "promoter_holding": {"promoter_pct": 50.3},
        "source": "screener.in",
    }


def test_map_fundamentals_row():
    r = map_fundamentals_row("RELIANCE", _data(), "2026-06-08")
    assert r["snapshot_date"] == "2026-06-08" and r["symbol"] == "RELIANCE"
    assert r["pe"] == 28.5 and r["roe"] == 18.2 and r["roce"] == 22.1
    assert r["market_cap_cr"] == 1850000 and r["book_value"] == 1200
    assert r["promoter_pct"] == 50.3
    assert r["source"] == "screener.in"
    # growth must actually land from the period-suffixed keys (regression guard)
    assert r["sales_growth"] == 14.0    # picks 3_years
    assert r["profit_growth"] == 18.0


def test_map_fundamentals_honest_empty_fields():
    r = map_fundamentals_row("X", {"fundamentals": {}, "promoter_holding": {}}, "2026-06-08")
    assert r["symbol"] == "X" and r["pe"] is None and r["promoter_pct"] is None


def test_map_fundamentals_none_data():
    assert map_fundamentals_row("X", None, "2026-06-08") is None

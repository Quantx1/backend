from backend.platform.scheduler import fundamentals_to_row


def test_fundamentals_to_row():
    data = {"fundamentals": {"pe": 10.0}, "promoter_holding": {"promoter_pct": 40.0}}
    r = fundamentals_to_row("TCS", data, "2026-06-08")
    assert r["symbol"] == "TCS" and r["pe"] == 10.0 and r["promoter_pct"] == 40.0

from backend.platform.scheduler import scan_universe_and_rows
from backend.services.intraday_scanner.scanner import IntradayMatch


def _match(sym):
    return IntradayMatch(symbol=sym, setup_id="orb_long", direction="bullish",
                         detected_at="t", timeframe="15m", entry=1, stop=0.9,
                         target=1.3, risk_reward=3.0, last_price=1.0,
                         volume_ratio=2.0, confidence="high", reason="r")


def test_scan_universe_and_rows_maps_matches_to_rows():
    matches = [_match("RELIANCE"), _match("TCS")]
    rows = scan_universe_and_rows(matches)
    assert len(rows) == 2
    assert {r["symbol"] for r in rows} == {"RELIANCE", "TCS"}
    assert all(r["signal_type"] == "intraday" and r["engine_name"] == "Intraday" for r in rows)

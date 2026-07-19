from backend.services.intraday_scanner.scanner import IntradayMatch
from backend.services.intraday_scanner.signal_mapper import (
    match_to_ws_payload, match_to_signal_row, CONFIDENCE_INT,
)


def _match(direction="bullish", confidence="high"):
    return IntradayMatch(
        symbol="RELIANCE", setup_id="vwap_bounce", direction=direction,
        detected_at="2026-06-08T10:05:00+05:30", timeframe="5m",
        entry=100.0, stop=99.0, target=103.0, risk_reward=3.0,
        last_price=100.2, volume_ratio=1.8, confidence=confidence,
        reason="Bounced off session VWAP with volume",
    )


def test_ws_payload_shape():
    p = match_to_ws_payload(_match())
    assert p["symbol"] == "RELIANCE"
    assert p["setup_id"] == "vwap_bounce"
    assert p["direction"] == "LONG"
    assert p["confidence"] == CONFIDENCE_INT["high"]
    assert p["entry"] == 100.0 and p["stop"] == 99.0 and p["target"] == 103.0
    assert p["reason"].startswith("Bounced")


def test_signal_row_shape():
    r = match_to_signal_row(_match(direction="bearish", confidence="medium"))
    assert r["signal_type"] == "intraday"
    assert r["engine_name"] == "Intraday"
    assert r["direction"] == "SHORT"
    assert r["confidence"] == CONFIDENCE_INT["medium"]
    assert r["status"] == "active"
    assert r["exchange"] == "NSE" and r["segment"] == "EQUITY"
    # setup_id is carried in raw_scores, not as a top-level signals column
    assert r["raw_scores"]["setup_id"] == "vwap_bounce"


def test_neutral_maps_to_neutral_direction():
    assert match_to_ws_payload(_match(direction="neutral"))["direction"] == "NEUTRAL"

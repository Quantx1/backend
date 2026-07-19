"""Guard: the intraday mappers stay consistent with the scanner dataclass."""
from backend.services.intraday_scanner.scanner import IntradayMatch
from backend.services.intraday_scanner.signal_mapper import match_to_ws_payload, match_to_signal_row


def test_mapper_handles_all_directions_and_confidences():
    for direction in ("bullish", "bearish", "neutral"):
        for conf in ("high", "medium", "low"):
            m = IntradayMatch(symbol="X", setup_id="s", direction=direction,
                              detected_at="t", timeframe="5m", entry=1, stop=0.9,
                              target=1.3, risk_reward=3.0, last_price=1.0,
                              volume_ratio=1.0, confidence=conf, reason="r")
            assert match_to_ws_payload(m)["confidence"] in (40, 60, 80)
            assert match_to_signal_row(m)["signal_type"] == "intraday"

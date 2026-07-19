"""Smart Alerts — pure rule helpers (#8)."""
from backend.services.news.live_alerts import volume_alert, oi_alert, breakout_alert, iv_alert


def test_volume_alert():
    assert volume_alert("X", 3.5)["type"] == "volume"
    assert volume_alert("X", 2.0) is None
    assert volume_alert("X", None) is None


def test_oi_alert():
    assert oi_alert("X", 20)["type"] == "oi"
    assert oi_alert("X", -18)["message"].startswith("Futures OI fell 18")
    assert oi_alert("X", 5) is None


def test_breakout_alert():
    assert breakout_alert("X", 105, 100)["type"] == "breakout"
    assert breakout_alert("X", 99, 100) is None
    assert breakout_alert("X", None, 100) is None


def test_iv_alert():
    assert iv_alert("X", 85)["type"] == "iv"
    assert iv_alert("X", 50) is None

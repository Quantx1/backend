"""Volume Intelligence — pure compute (#9)."""
from backend.services.market.volume_intelligence import compute_volume_intel, _drivers


def test_accumulation_signal():
    vols = [100.0] * 20 + [300.0]      # 3× average today
    delivs = [40.0] * 20 + [70.0]      # delivery jumps
    v = compute_volume_intel(vols, delivs)
    assert v["x_avg"] == 3.0
    assert v["delivery_today"] == 70.0
    assert v["signal"] == "accumulation"
    assert any("accumulation" in d for d in _drivers(v))


def test_churn_signal():
    vols = [100.0] * 20 + [300.0]
    delivs = [40.0] * 20 + [25.0]      # heavy volume, weak delivery
    assert compute_volume_intel(vols, delivs)["signal"] == "churn"


def test_quiet_and_too_short():
    assert compute_volume_intel([100.0] * 20 + [30.0])["signal"] == "quiet"
    assert compute_volume_intel([1.0, 2.0])["signal"] == "normal"

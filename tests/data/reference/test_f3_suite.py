from backend.data.reference.nse_derivatives import build_derivatives_metrics


def test_metrics_honest_empty_on_no_options():
    assert build_derivatives_metrics([]) == []


def test_pcr_none_when_no_ce_oi():
    rows = [{"date": "d", "symbol": "X", "expiry": "e", "strike": 100,
             "option_type": "PE", "oi": 50, "volume": 1}]
    m = build_derivatives_metrics(rows)[0]
    assert m["pcr_oi"] is None and m["total_pe_oi"] == 50

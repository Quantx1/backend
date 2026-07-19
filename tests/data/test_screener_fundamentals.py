"""screener.in fundamentals scraper (Portfolio Doctor F7).

The audit found run_finrobot_doctor was called with empty fundamentals, so the
CoT agents graded ROE/growth/promoter from empty JSON. These tests pin the
scraper's parse contract (faithful screener.in HTML — ₹/comma/%/'Cr.' values,
ranges tables, shareholding) and the honest-empty failure path. The network
fetch is mocked so they're deterministic offline.
"""
from __future__ import annotations

import pytest

import backend.data.fundamentals.screener_in as si


# Faithful slice of a screener.in consolidated page.
SCREENER_HTML = """
<html><body>
<ul id="top-ratios">
  <li class="flex"><span class="name">Market Cap</span>
      <span class="value"><span class="number">17,64,237</span> Cr.</span></li>
  <li class="flex"><span class="name">Current Price</span>
      <span class="value">₹ <span class="number">1,304</span></span></li>
  <li class="flex"><span class="name">High / Low</span>
      <span class="value">₹1,612/1,290</span></li>
  <li class="flex"><span class="name">Stock P/E</span>
      <span class="value"><span class="number">22.7</span></span></li>
  <li class="flex"><span class="name">Book Value</span>
      <span class="value">₹ <span class="number">668</span></span></li>
  <li class="flex"><span class="name">ROCE</span>
      <span class="value"><span class="number">10.3</span> %</span></li>
  <li class="flex"><span class="name">ROE</span>
      <span class="value"><span class="number">8.91</span> %</span></li>
</ul>
<table class="ranges-table"><tbody>
  <tr><th colspan="2">Compounded Sales Growth</th></tr>
  <tr><td>10 Years:</td><td>15%</td></tr>
  <tr><td>3 Years:</td><td>6%</td></tr>
</tbody></table>
<table class="ranges-table"><tbody>
  <tr><th colspan="2">Compounded Profit Growth</th></tr>
  <tr><td>5 Years:</td><td>12%</td></tr>
</tbody></table>
<div class="pros"><ul><li>Company has strong cash flows.</li></ul></div>
<div class="cons"><ul><li>Low return on equity of 8.77% over last 3 years.</li></ul></div>
<section id="shareholding"><table><tbody>
  <tr><td class="text">Promoters+</td><td>50.10%</td><td>50.00%</td></tr>
  <tr><td class="text">FIIs+</td><td>11.2%</td><td>11.5%</td></tr>
</tbody></table></section>
</body></html>
"""


class _Resp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status


@pytest.fixture(autouse=True)
def _clear_cache():
    si._cache.clear()
    yield
    si._cache.clear()


def test_num_parser():
    assert si._num("₹17,64,237Cr.") == 1764237.0
    assert si._num("22.7") == 22.7
    assert si._num("8.91%") == 8.91
    assert si._num("15%") == 15.0
    assert si._num("₹ 668") == 668.0
    assert si._num(None) is None
    assert si._num("-") is None


def test_get_fundamentals_parses_real_format(monkeypatch):
    monkeypatch.setattr(si, "_fetch_html", lambda sym: SCREENER_HTML)
    d = si.get_fundamentals("RELIANCE")

    assert d["available"] is True
    assert d["source"] == "screener.in"
    f = d["fundamentals"]
    assert f["market_cap_cr"] == 1764237.0
    assert f["pe"] == 22.7
    assert f["roe"] == 8.91
    assert f["roce"] == 10.3
    assert f["book_value"] == 668.0
    assert "high_low" not in f and "High / Low" not in f   # range skipped
    # growth ranges
    assert d["growth"]["sales_growth_10_years"] == 15.0
    assert d["growth"]["profit_growth_5_years"] == 12.0
    # qualitative + promoter
    assert d["cons"] and "return on equity" in d["cons"][0].lower()
    assert d["promoter_holding"]["promoter_pct"] == 50.00   # latest period


def test_get_fundamentals_honest_empty_on_failure(monkeypatch):
    monkeypatch.setattr(si, "_fetch_html", lambda sym: None)
    d = si.get_fundamentals("NOSUCH")
    assert d["available"] is False
    assert d["source"] == "unavailable"
    assert d["fundamentals"] == {}              # never synthetic
    assert d["last_error"]


def test_fetch_html_rejects_non_200(monkeypatch):
    """A 404/blocked page (no top-ratios) must not be treated as data."""
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp("<html>blocked</html>", 403))
    assert si._fetch_html("RELIANCE") is None


def test_to_doctor_inputs_shape(monkeypatch):
    monkeypatch.setattr(si, "_fetch_html", lambda sym: SCREENER_HTML)
    di = si.to_doctor_inputs("RELIANCE")
    assert set(di) == {"fundamentals", "promoter_holding", "peers"}
    # growth folded into fundamentals; pros/cons attached
    assert di["fundamentals"]["sales_growth_10_years"] == 15.0
    assert di["fundamentals"]["roe"] == 8.91
    assert "cons" in di["fundamentals"]
    assert di["promoter_holding"]["promoter_pct"] == 50.0


def test_to_doctor_inputs_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(si, "_fetch_html", lambda sym: None)
    di = si.to_doctor_inputs("NOSUCH")
    assert di["fundamentals"] == {}             # honest-empty, agents see nothing fake
    assert di["peers"] == []


def test_transient_failure_caches_short_real_data_caches_long(monkeypatch):
    """A screener.in blip must NOT pin honest-empty for 6h — failures use a
    short TTL so they self-heal; only real data is cached long."""
    monkeypatch.setattr(si, "_fetch_html", lambda sym: None)
    si.get_fundamentals("FLAKY")
    assert si._cache["FLAKY"][2] == si._TTL_FAIL    # transient → self-heals fast

    si._cache.clear()
    monkeypatch.setattr(si, "_fetch_html", lambda sym: SCREENER_HTML)
    si.get_fundamentals("GOOD")
    assert si._cache["GOOD"][2] == si._TTL_OK        # real data → cached 6h
    assert si._TTL_FAIL < si._TTL_OK

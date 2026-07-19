"""Alpha Factory unified Scores block — pure shaping tests (no DB/network).

`build_scores` is exercised directly with fabricated inputs; `scores()` is
exercised with the four `_read_*` readers monkeypatched.
"""
import backend.services.scanners.stock_scores as ss
from backend.services.scanners.stock_scores import _rank_to_pct, build_scores


# ── _rank_to_pct ─────────────────────────────────────────────────────────


def test_rank_to_pct_endpoints_and_midpoint():
    # rank #1 of 101 -> 100; last -> 0; middle -> 50
    assert _rank_to_pct(1, 101) == 100.0
    assert _rank_to_pct(101, 101) == 0.0
    assert _rank_to_pct(51, 101) == 50.0


def test_rank_to_pct_honest_none_on_bad_inputs():
    assert _rank_to_pct(None, 100) is None
    assert _rank_to_pct(5, None) is None
    assert _rank_to_pct(1, 1) is None      # universe too small to rank against
    assert _rank_to_pct(0, 100) is None    # nonsense rank
    assert _rank_to_pct(101, 100) is None  # rank beyond universe


# ── build_scores: full set ───────────────────────────────────────────────


def _full_inputs():
    alpha = {"rank": 5, "universe": 101, "trade_date": "2026-06-10"}
    factor_pcts = {"momentum": 80.0, "trend": 60.0, "low_volatility": 40.0}
    mood = {"score": 0.42, "headlines": 7, "trade_date": "2026-06-11"}
    iv = {"iv_rank": 63.2, "iv_percentile": 71.0, "days": 40, "current_iv": 0.22}
    return alpha, factor_pcts, mood, iv


def test_full_set_composite_is_mean_of_cross_sectional_pcts_only():
    out = build_scores(*_full_inputs())
    by_key = {e["key"]: e for e in out["scores"]}
    assert set(by_key) == {"alpha", "momentum", "trend", "low_volatility", "mood", "iv_rank"}

    # alpha: rank 5 of 101 -> 96 pct; value carries the raw rank
    assert by_key["alpha"]["pct"] == 96.0
    assert by_key["alpha"]["value"] == 5
    assert "#5 of 101" in by_key["alpha"]["note"]

    # factors carry their percentile, no raw value (factor_rank exposes pcts only)
    assert by_key["momentum"]["pct"] == 80.0
    assert by_key["momentum"]["value"] is None

    # mood + iv_rank are NOT cross-sectional -> pct None, value carries the score
    assert by_key["mood"]["pct"] is None
    assert by_key["mood"]["value"] == 0.42
    assert by_key["iv_rank"]["pct"] is None
    assert by_key["iv_rank"]["value"] == 63.2

    # composite = mean(96, 80, 60, 40) — mood/iv excluded (no pct)
    assert out["composite"] == round((96.0 + 80.0 + 60.0 + 40.0) / 4, 2)


# ── build_scores: sparse / honest-empty ──────────────────────────────────


def test_sparse_fewer_than_two_pcts_means_composite_none():
    # only one cross-sectional pct (alpha) + a mood score -> composite None
    out = build_scores(
        {"rank": 10, "universe": 101, "trade_date": "2026-06-10"},
        {}, {"score": -0.3, "headlines": 2, "trade_date": "2026-06-11"}, None,
    )
    keys = [e["key"] for e in out["scores"]]
    assert keys == ["alpha", "mood"]
    assert out["composite"] is None


def test_missing_sources_are_honestly_omitted():
    # nothing available at all -> empty scores, composite None
    out = build_scores(None, {}, None, None)
    assert out == {"scores": [], "composite": None}

    # a factor whose pct is None is dropped, not zero-filled
    out2 = build_scores(None, {"momentum": 70.0, "trend": None}, None, None)
    keys = [e["key"] for e in out2["scores"]]
    assert keys == ["momentum"]
    assert out2["composite"] is None  # only 1 pct present


def test_alpha_without_universe_keeps_row_but_no_pct():
    out = build_scores({"rank": 3, "universe": None, "trade_date": None},
                       {"momentum": 50.0, "low_volatility": 90.0}, None, None)
    by_key = {e["key"]: e for e in out["scores"]}
    assert by_key["alpha"]["pct"] is None       # can't percentile without n
    assert by_key["alpha"]["value"] == 3
    # composite from the two factor pcts alone
    assert out["composite"] == 70.0


def test_no_fabricated_quality_or_smart_money_rows():
    out = build_scores(*_full_inputs())
    keys = {e["key"] for e in out["scores"]}
    assert "quality" not in keys
    assert "smart_money" not in keys


# ── scores(): readers monkeypatched ──────────────────────────────────────


def test_scores_wires_readers_and_canonicalizes_symbol(monkeypatch):
    seen = {}

    def fake_alpha(sym):
        seen["alpha"] = sym
        return {"rank": 1, "universe": 51, "trade_date": "2026-06-10"}

    monkeypatch.setattr(ss, "_read_alpha", fake_alpha)
    monkeypatch.setattr(ss, "_read_factor_pcts", lambda s: {"momentum": 25.0})
    monkeypatch.setattr(ss, "_read_mood", lambda s: None)
    monkeypatch.setattr(ss, "_read_iv", lambda s: None)

    out = ss.scores("reliance.ns")
    assert seen["alpha"] == "RELIANCE"          # .NS stripped + uppercased
    assert out["symbol"] == "RELIANCE"
    by_key = {e["key"]: e for e in out["scores"]}
    assert by_key["alpha"]["pct"] == 100.0
    assert by_key["momentum"]["pct"] == 25.0
    assert out["composite"] == 62.5             # mean(100, 25)


def test_scores_empty_symbol_is_honest_empty(monkeypatch):
    # readers must never be called for an empty symbol
    def boom(_):
        raise AssertionError("reader called for empty symbol")
    for r in ("_read_alpha", "_read_factor_pcts", "_read_mood", "_read_iv"):
        monkeypatch.setattr(ss, r, boom)
    out = ss.scores("  ")
    assert out["scores"] == [] and out["composite"] is None

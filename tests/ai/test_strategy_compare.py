"""Strategy compare — pure head-to-head extraction + ranking."""
from backend.ai.strategy.compare import compare_strategies


def _strat(sid, name, *, sharpe=None, consistency=None, holdout=None, dd=None, trades=None, status="paper"):
    oos = {}
    if sharpe is not None:
        oos["oos_mean_sharpe"] = sharpe
    if consistency is not None:
        oos["oos_consistency"] = consistency
    if holdout is not None:
        oos["holdout_return_pct"] = holdout
    if dd is not None:
        oos["oos_worst_drawdown_pct"] = dd
    if trades is not None:
        oos["oos_trades"] = trades
    return {
        "id": sid, "name": name, "status": status,
        "last_backtest": {"out_of_sample": oos} if oos else {},
    }


def test_winner_per_metric_higher_better():
    a = _strat("a", "A", sharpe=1.2, trades=40)
    b = _strat("b", "B", sharpe=0.8, trades=80)
    out = compare_strategies([a, b])
    assert out["winners"]["oos_sharpe"] == "a"   # higher sharpe wins
    assert out["winners"]["oos_trades"] == "b"   # more trades wins


def test_drawdown_winner_is_lower():
    a = _strat("a", "A", sharpe=1.0, dd=30.0)
    b = _strat("b", "B", sharpe=1.0, dd=12.0)
    out = compare_strategies([a, b])
    assert out["winners"]["oos_worst_drawdown_pct"] == "b"  # shallower DD wins


def test_best_overall_prefers_gate_passers():
    # 'a' has the higher Sharpe but fails the gate (too few trades / no holdout);
    # 'b' clears the gate → best_overall should be the gate-passer.
    a = _strat("a", "A", sharpe=2.0, trades=5, consistency=0.2, holdout=-3.0, dd=10.0)
    b = _strat("b", "B", sharpe=1.0, trades=40, consistency=0.8, holdout=4.0, dd=15.0)
    out = compare_strategies([b, a])
    cards = {c["id"]: c for c in out["strategies"]}
    assert cards["a"]["gate_pass"] is False
    assert cards["b"]["gate_pass"] is True
    assert out["best_overall"] == "b"


def test_missing_metric_is_honest_none():
    a = _strat("a", "A")            # no backtest at all
    b = _strat("b", "B", sharpe=1.0, trades=40, consistency=0.8, holdout=2.0, dd=20.0)
    out = compare_strategies([a, b])
    cards = {c["id"]: c for c in out["strategies"]}
    assert cards["a"]["has_backtest"] is False
    assert cards["a"]["metrics"]["oos_sharpe"] is None
    # best_overall falls to the only scored strategy
    assert out["best_overall"] == "b"

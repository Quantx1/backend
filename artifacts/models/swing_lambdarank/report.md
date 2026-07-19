# Training report — swing_lambdarank

**Verdict:** NOT shippable — ic=-0.003500108901918351 icir=-0.1468241023990875 dsr=0.0 pbo=0.5 below gate

## Headline metrics

| metric | value |
| --- | --- |
| rank_ic_mean | -0.0035 |
| rank_icir | -0.1468 |
| decile_spread_mean | 0.0004 |
| deflated_sharpe | 0.0 |
| probability_backtest_overfitting | 0.5 |
| n_features | 61 |
| n_folds | 5 |
| n_rows | 344595 |
| n_symbols | 310 |
| n_dates | 1232 |
| train_seconds | 1042.4 |

## Per-fold rank-IC

`[0.0234, -0.0374, -0.0206, 0.0224, -0.0054]`

## Top features (gain)

| feature | gain |
| --- | --- |
| realized_vol_21 | 6928.5 |
| adx_14 | 2836.8 |
| corr_index_63 | 2690.0 |
| chronos_uncert | 2464.5 |
| beta_index_63 | 2441.0 |
| chronos_fwd_ret | 2426.4 |
| tsfm_fwd_ret | 2425.9 |
| atr_pct_14 | 2410.7 |
| ens_fwd_ret | 2174.2 |
| kronos_fwd_ret | 1943.7 |
| realized_vol_10 | 1703.3 |
| sma20_slope_10 | 1606.8 |
| ret_63d | 1514.9 |
| ret_42d | 1223.9 |
| parkinson_vol_10 | 1126.8 |

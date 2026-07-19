# Training report — positional_lambdarank

**Verdict:** NOT shippable — ic=-0.06128304274685373 icir=-1.0925025522319636 dsr=0.0 pbo=0.5 below gate

## Headline metrics

| metric | value |
| --- | --- |
| rank_ic_mean | -0.0613 |
| rank_icir | -1.0925 |
| decile_spread_mean | -0.0013 |
| deflated_sharpe | 0.0 |
| probability_backtest_overfitting | 0.5 |
| n_features | 52 |
| n_folds | 4 |
| n_rows | 308595 |
| n_symbols | 310 |
| n_dates | 1080 |
| train_seconds | 2216.6 |

## Per-fold rank-IC

`[-0.1124, -0.1211, -0.0164, 0.0049]`

## Top features (gain)

| feature | gain |
| --- | --- |
| realized_vol_252 | 100010.5 |
| corr_index_252 | 19631.6 |
| amihud_illiq_63 | 16244.0 |
| max_drawdown_126 | 14726.5 |
| beta_index_252 | 14322.3 |
| ulcer_index_126 | 13499.6 |
| turnover_63 | 10030.9 |
| realized_vol_126 | 9255.1 |
| downside_vol_126 | 8555.2 |
| xs_rank_ret_252 | 8431.7 |
| mom_consistency_252 | 5620.1 |
| sma_200_slope_63 | 5121.0 |
| xs_rank_rs_index_252 | 4711.9 |
| sharpe_252 | 4432.4 |
| realized_vol_63 | 4245.4 |

# Training report — positional_lambdarank

**Verdict:** NOT shippable — ic=-0.06953535922731967 icir=-1.0972668037587943 dsr=0.0 pbo=0.5 below gate

## Headline metrics

| metric | value |
| --- | --- |
| rank_ic_mean | -0.0695 |
| rank_icir | -1.0973 |
| decile_spread_mean | -0.019 |
| deflated_sharpe | 0.0 |
| probability_backtest_overfitting | 0.5 |
| n_features | 52 |
| n_folds | 4 |
| n_rows | 308515 |
| n_symbols | 310 |
| n_dates | 1077 |
| train_seconds | 887.3 |

## Per-fold rank-IC

`[-0.1372, -0.1153, -0.0513, 0.0256]`

## Top features (gain)

| feature | gain |
| --- | --- |
| realized_vol_252 | 63744.8 |
| corr_index_252 | 18490.6 |
| amihud_illiq_63 | 17968.5 |
| ulcer_index_126 | 14721.0 |
| max_drawdown_126 | 13829.4 |
| turnover_63 | 11992.8 |
| realized_vol_126 | 11934.5 |
| beta_index_252 | 10840.2 |
| xs_rank_ret_252 | 8147.0 |
| downside_vol_126 | 6740.3 |
| mom_consistency_252 | 5731.3 |
| sma_100_slope_63 | 4923.9 |
| sma_200_slope_63 | 4591.7 |
| xs_rank_rs_index_252 | 3812.8 |
| realized_vol_63 | 3492.0 |

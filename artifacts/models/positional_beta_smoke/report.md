# Training report — positional_lambdarank

**Verdict:** NOT shippable — ic=-0.031797573569539495 icir=-0.9133916438335975 dsr=0.0 pbo=0.5 below gate

## Headline metrics

| metric | value |
| --- | --- |
| rank_ic_mean | -0.0318 |
| rank_icir | -0.9134 |
| decile_spread_mean | -0.0053 |
| deflated_sharpe | 0.0 |
| probability_backtest_overfitting | 0.5 |
| n_features | 52 |
| n_folds | 4 |
| n_rows | 41208 |
| n_symbols | 40 |
| n_dates | 1077 |
| train_seconds | 15.3 |

## Per-fold rank-IC

`[-0.0696, -0.0328, 0.0241, -0.0489]`

## Top features (gain)

| feature | gain |
| --- | --- |
| realized_vol_252 | 11865.2 |
| max_drawdown_126 | 9195.3 |
| ulcer_index_126 | 7459.2 |
| beta_index_252 | 7173.8 |
| turnover_63 | 6971.2 |
| amihud_illiq_63 | 6831.3 |
| corr_index_252 | 6589.7 |
| mom_consistency_252 | 4972.1 |
| vol_ratio_63_252 | 4865.1 |
| realized_vol_126 | 4322.0 |
| sma_200_slope_63 | 3828.7 |
| downside_vol_126 | 3669.0 |
| sma_100_slope_63 | 3533.3 |
| pct_days_above_sma200_126 | 3361.5 |
| mom_consistency_126 | 2088.8 |

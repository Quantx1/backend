# Training report — positional_lambdarank

**Verdict:** NOT shippable — ic=-0.015184340348079792 icir=-0.215090348325656 dsr=0.0 pbo=0.5 below gate

## Headline metrics

| metric | value |
| --- | --- |
| rank_ic_mean | -0.0152 |
| rank_icir | -0.2151 |
| decile_spread_mean | -0.0208 |
| deflated_sharpe | 0.0 |
| probability_backtest_overfitting | 0.5 |
| n_features | 52 |
| n_folds | 4 |
| n_rows | 308595 |
| n_symbols | 310 |
| n_dates | 1080 |
| train_seconds | 1205.3 |

## Per-fold rank-IC

`[0.0369, 0.0494, -0.0177, -0.1294]`

## Top features (gain)

| feature | gain |
| --- | --- |
| realized_vol_252 | 8049.9 |
| turnover_63 | 6941.3 |
| max_drawdown_126 | 6831.6 |
| amihud_illiq_63 | 6694.9 |
| ulcer_index_126 | 5709.0 |
| beta_index_252 | 4617.2 |
| downside_vol_126 | 4529.4 |
| corr_index_252 | 4446.8 |
| realized_vol_63 | 3814.4 |
| vol_ratio_63_252 | 3639.4 |
| mom_consistency_252 | 3563.3 |
| realized_vol_126 | 3497.3 |
| sma_200_slope_63 | 3072.1 |
| pct_days_above_sma200_126 | 2480.6 |
| sma_100_slope_63 | 1928.6 |

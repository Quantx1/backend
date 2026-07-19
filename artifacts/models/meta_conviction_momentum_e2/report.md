# Training report — meta_conviction_momentum

**Verdict:** NOT shippable — auc=0.5083 fold_aucs=[0.501, 0.498, 0.51, 0.525] tercile_lift=0.0059 brier_calibrated=0.2498 (climatology 0.2485) below gate

## Headline metrics

| metric | value |
| --- | --- |
| n_features | 20 |
| n_folds | 4 |
| n_rows | 226344 |
| n_symbols | 310 |
| n_dates | 756 |
| train_seconds | 128.4 |

## Top features (gain)

| feature | gain |
| --- | --- |
| amihud_illiq_21 | 17537.3 |
| score_dispersion | 17227.7 |
| days_since_switch | 15728.9 |
| realized_vol_63 | 15157.6 |
| dist_high_252 | 14799.8 |
| mkt_rv21 | 13616.1 |
| beta_index_63 | 8788.6 |
| score_z | 6925.2 |
| tsfm_fwd_ret | 5625.0 |
| mkt_dist_high_63 | 5445.9 |
| mkt_ret_21 | 5023.0 |
| score_pct | 3937.2 |
| tsfm_kronos_spread | 3913.3 |
| ens_fwd_ret | 3571.2 |
| kronos_fwd_ret | 3558.9 |

# Training report — meta_conviction_momentum

**Verdict:** NOT shippable — auc=0.5754 fold_aucs=[0.53, 0.529, 0.7, 0.542] tercile_lift=0.0216 brier=0.2708 (climatology 0.2497) below gate

## Headline metrics

| metric | value |
| --- | --- |
| n_features | 20 |
| n_folds | 4 |
| n_rows | 226344 |
| n_symbols | 310 |
| n_dates | 756 |
| train_seconds | 124.4 |

## Top features (gain)

| feature | gain |
| --- | --- |
| score_dispersion | 355082.4 |
| days_since_switch | 347478.0 |
| mkt_rv21 | 233279.9 |
| regime_bull | 136345.5 |
| mkt_dist_high_63 | 127937.3 |
| regime_bear | 94544.3 |
| amihud_illiq_21 | 53573.9 |
| mkt_ret_21 | 39492.5 |
| dist_high_252 | 21674.3 |
| beta_index_63 | 9891.3 |
| realized_vol_63 | 5052.3 |
| regime_confidence | 3268.5 |
| tsfm_kronos_spread | 3110.4 |
| ens_fwd_ret | 2285.3 |
| score_z | 1536.1 |

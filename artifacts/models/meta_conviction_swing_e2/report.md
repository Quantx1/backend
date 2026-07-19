# Training report — meta_conviction_swing

**Verdict:** NOT shippable — auc=0.5037 fold_aucs=[0.501, 0.506, 0.513, 0.495] tercile_lift=0.0042 brier_calibrated=0.2489 (climatology 0.2487) below gate

## Headline metrics

| metric | value |
| --- | --- |
| n_features | 22 |
| n_folds | 4 |
| n_rows | 228929 |
| n_symbols | 310 |
| n_dates | 749 |
| train_seconds | 129.5 |

## Top features (gain)

| feature | gain |
| --- | --- |
| days_since_switch | 12380.1 |
| mkt_rv21 | 11308.4 |
| chronos_uncert | 11056.9 |
| score_dispersion | 9021.5 |
| beta_index_63 | 8876.9 |
| tsfm_fwd_ret | 7646.5 |
| realized_vol_21 | 6674.5 |
| chronos_fwd_ret | 6535.6 |
| ens_fwd_ret | 6254.2 |
| mkt_dist_high_63 | 5839.4 |
| tsfm_kronos_spread | 5034.0 |
| kronos_fwd_ret | 4900.6 |
| mkt_ret_21 | 4570.2 |
| pullback_from_high_21 | 4038.4 |
| score_z | 3865.8 |

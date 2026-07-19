# Training report — meta_conviction_swing

**Verdict:** NOT shippable — auc=0.5368 fold_aucs=[0.51, 0.449, 0.7, 0.487] tercile_lift=0.0573 brier=0.2671 (climatology 0.2484) below gate

## Headline metrics

| metric | value |
| --- | --- |
| n_features | 22 |
| n_folds | 4 |
| n_rows | 228929 |
| n_symbols | 310 |
| n_dates | 749 |
| train_seconds | 136.3 |

## Top features (gain)

| feature | gain |
| --- | --- |
| days_since_switch | 50809.9 |
| score_dispersion | 46876.2 |
| mkt_dist_high_63 | 46559.3 |
| mkt_rv21 | 39059.7 |
| mkt_ret_21 | 29583.5 |
| regime_bull | 15609.6 |
| beta_index_63 | 10245.7 |
| chronos_uncert | 8078.1 |
| regime_confidence | 7658.0 |
| regime_bear | 7144.4 |
| realized_vol_21 | 5272.5 |
| chronos_fwd_ret | 4910.6 |
| kronos_fwd_ret | 4691.2 |
| pullback_from_high_21 | 4598.9 |
| ens_fwd_ret | 4304.5 |

# E-013 GOLD# historical data quality (H-021, diagnostic)

Outcome: **supported**

supported: VR_day P0 0.4341 P1 0.932 (delta 0.4979, CI [0.405, 0.5979]), VR_week delta 0.4577 (CI [0.2796, 0.6539]); RV level P1/P0 0.4653, VR-adjusted 0.999; change months {'spread_zero_share': '2018-10', 'h1_bars_per_day': '2018-04', 'm1_zero_return_share': '2020-03'}; findings: M1 spread field: confirmed_defect; H1 spread field: confirmed_defect; D1 spread field: confirmed_defect; session length: confirmed_collection_change; RV level shift: intraday_reversal_excess_consistent_with_quote_noise; external cross-check: undetermined

Diagnostic only: no model is trained, no strategy evaluated, and nothing here re-establishes any past out-of-sample edge. M1/H1/D1 agreement and RV_H1/RV_15m are low-discrimination (same stored M1).

## Findings (pre-declared classes)

| item | class | evidence |
|---|---|---|
| M1 spread field | confirmed_defect | SPREAD = 0 on > 50% of bars in [2013, 2014, 2015, 2016, 2017, 2018] (missing cost field, not a price defect) |
| H1 spread field | confirmed_defect | SPREAD = 0 on > 50% of bars in [2013, 2014, 2015, 2016, 2017, 2018] (missing cost field, not a price defect) |
| D1 spread field | confirmed_defect | SPREAD = 0 on > 50% of bars in [2013, 2014, 2015, 2016, 2017, 2018] (missing cost field, not a price defect) |
| session length | confirmed_collection_change | median H1 bars/day [24.0] before vs [23.0] after; extra hours [0] |
| RV level shift | intraday_reversal_excess_consistent_with_quote_noise | VR_day P0 0.4341 (CI [0.3905, 0.4861]) vs P1 0.932 (CI [0.8535, 1.0227]), delta CI [0.405, 0.5979], week delta 0.4577; real intraday mean reversion is not excluded |
| external cross-check | undetermined | no independent gold feed reachable; a feed-wide level difference that is internally consistent cannot be excluded |

## Primary statistics

{'vr_day_P0': 0.4341, 'vr_day_P1': 0.932, 'delta_vr_day': 0.4979, 'delta_vr_day_ci95': [0.405, 0.5979], 'vr_day_P0_ci95': [0.3905, 0.4861], 'vr_day_P1_ci95': [0.8535, 1.0227], 'delta_vr_week': 0.4577, 'delta_vr_week_ci95': [0.2796, 0.6539], 'rv_level_ratio_P1_over_P0': 0.4653, 'rv_level_ratio_vr_adjusted': 0.999, 'cross_residual_ratio_2017H2_over_2018_2023': np.float64(12.322), 'spike_rate_ratio_P0_over_P1': 1.459}

## By period

{'P0': {'days': 910, 'mean_rv_h1': 0.00017821597303496934, 'vr_day': 0.4341, 'vr_week': 0.45, 'roll_noise_share_h1': 0.00017, 'ratio_rv_h1_over_rv_15m': 1.4139, 'ratio_rv_h1_over_parkinson': 2.296, 'm1_zero_return_share': 0.0627, 'spikes_per_1000_days': 202.2}, 'P1': {'days': 1515, 'mean_rv_h1': 8.292527070769637e-05, 'vr_day': 0.932, 'vr_week': 0.9077, 'roll_noise_share_h1': 0.00126, 'ratio_rv_h1_over_rv_15m': 1.0227, 'ratio_rv_h1_over_parkinson': 1.0175, 'm1_zero_return_share': 0.0474, 'spikes_per_1000_days': 138.61}, 'P2': {'days': 731, 'mean_rv_h1': 0.00017250389630438847, 'vr_day': 1.0689, 'vr_week': 0.8357, 'roll_noise_share_h1': 0.00018, 'ratio_rv_h1_over_rv_15m': 1.0202, 'ratio_rv_h1_over_parkinson': 0.9526, 'm1_zero_return_share': 0.0093, 'spikes_per_1000_days': 134.06}}

## By year (daily means; vr_day per year)

| key | days | rv_h1 | rv_1m | rv_15m | rv_60m | park | w2 | spikes | m1_bars | zero_return_share | vr_day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2014 | 140 | 0.000155345 | 0.000125182 | 0.00012258 | 0.000120517 | 9.28982e-05 | -1.90002e-10 | 0.235714 | 1350.95 | 0.0767509 | 0.669395 |
| 2015 | 257 | 0.000163194 | 0.000130349 | 0.000122802 | 0.000119528 | 8.07321e-05 | 1.62933e-09 | 0.2607 | 1369.11 | 0.0699454 | 0.449174 |
| 2016 | 257 | 0.000272628 | 0.000186436 | 0.000182368 | 0.00018278 | 0.000102961 | -2.09972e-10 | 0.163424 | 1372.53 | 0.0482945 | 0.350635 |
| 2017 | 256 | 0.000111023 | 8.14644e-05 | 7.46642e-05 | 7.35729e-05 | 4.07042e-05 | 9.48659e-10 | 0.164062 | 1372.91 | 0.062248 | 0.355791 |
| 2018 | 250 | 5.49968e-05 | 5.03816e-05 | 4.66571e-05 | 4.49912e-05 | 4.0325e-05 | 8.93939e-10 | 0.068 | 1372.9 | 0.0871472 | 0.685714 |
| 2019 | 249 | 5.61869e-05 | 5.64589e-05 | 5.2635e-05 | 5.28173e-05 | 5.598e-05 | 1.10144e-09 | 0.156627 | 1375.76 | 0.111939 | 0.925625 |
| 2020 | 250 | 0.000150817 | 0.000192369 | 0.000156077 | 0.000143876 | 0.000158786 | 8.92573e-09 | 0.12 | 1377.61 | 0.0209346 | 0.976266 |
| 2021 | 251 | 8.17809e-05 | 8.46583e-05 | 8.04193e-05 | 7.40902e-05 | 7.87836e-05 | 1.63156e-09 | 0.159363 | 1377.97 | 0.0230461 | 0.880012 |
| 2022 | 258 | 8.75709e-05 | 9.48072e-05 | 8.64344e-05 | 8.56788e-05 | 8.79576e-05 | 1.18818e-09 | 0.162791 | 1373.79 | 0.0211507 | 0.994341 |
| 2023 | 257 | 6.64102e-05 | 6.37168e-05 | 6.44452e-05 | 6.25864e-05 | 6.72533e-05 | 2.30119e-11 | 0.163424 | 1373.7 | 0.0222321 | 1.02314 |
| 2024 | 258 | 7.75589e-05 | 8.02352e-05 | 7.67824e-05 | 7.59059e-05 | 8.25105e-05 | 8.56568e-10 | 0.174419 | 1372.17 | 0.0160437 | 1.16331 |
| 2025 | 253 | 0.000131258 | 0.000133343 | 0.000128109 | 0.0001232 | 0.000136783 | 9.15035e-10 | 0.0948617 | 1374.45 | 0.00740549 | 1.1131 |
| 2026 | 220 | 0.000331281 | 0.00033973 | 0.000324446 | 0.000301004 | 0.000347671 | 2.23722e-10 | 0.131818 | 1375.99 | 0.00355319 | 1.02277 |

## Hour-of-day share of squared H1 returns

{'hours_present_before_2018_only': [0], 'P0': {'share_by_hour': {0: 0.2931, 1: 0.2897, 2: 0.0079, 3: 0.015, 4: 0.0176, 5: 0.0101, 6: 0.0106, 7: 0.007, 8: 0.011, 9: 0.0169, 10: 0.0187, 11: 0.0114, 12: 0.0119, 13: 0.0122, 14: 0.0169, 15: 0.0575, 16: 0.053, 17: 0.0407, 18: 0.0236, 19: 0.0182, 20: 0.0226, 21: 0.0178, 22: 0.0119, 23: 0.0048}, 'share_of_extra_hours': 0.2931}, 'P1': {'share_by_hour': {0: 0.0253, 1: 0.065, 2: 0.0194, 3: 0.0316, 4: 0.0349, 5: 0.015, 6: 0.0127, 7: 0.0142, 8: 0.0245, 9: 0.0367, 10: 0.0401, 11: 0.0316, 12: 0.0327, 13: 0.0285, 14: 0.0388, 15: 0.12, 16: 0.1156, 17: 0.1149, 18: 0.0536, 19: 0.0336, 20: 0.0358, 21: 0.0402, 22: 0.0251, 23: 0.01}, 'share_of_extra_hours': 0.0253}, 'P2': {'share_by_hour': {0: 0.0036, 1: 0.0725, 2: 0.0327, 3: 0.046, 4: 0.0656, 5: 0.0304, 6: 0.0155, 7: 0.0332, 8: 0.0421, 9: 0.0291, 10: 0.0314, 11: 0.0349, 12: 0.0204, 13: 0.0315, 14: 0.0382, 15: 0.0862, 16: 0.0838, 17: 0.1051, 18: 0.0498, 19: 0.0349, 20: 0.0292, 21: 0.0366, 22: 0.0354, 23: 0.0117}, 'share_of_extra_hours': 0.0036}}

## Change months

{'spread_zero_share': '2018-10', 'h1_bars_per_day': '2018-04', 'm1_zero_return_share': '2020-03'}

## Bar integrity M1

| key | ohlc_inconsistent | flat | spread_zero | spread_median_nonzero | duplicate_or_backward_time |
|---|---|---|---|---|---|
| 2013 | 0 | 0.000650195 | 1 | nan | 0 |
| 2014 | 0 | 0.0360364 | 1 | nan | 0 |
| 2015 | 0 | 0.0196424 | 1 | nan | 0 |
| 2016 | 0 | 0.00636595 | 1 | nan | 0 |
| 2017 | 0 | 0.00311045 | 1 | nan | 0 |
| 2018 | 0 | 0.0136607 | 0.780682 | 22 | 0 |
| 2019 | 0 | 0.0178698 | 2.55567e-05 | 22 | 0 |
| 2020 | 0 | 0.00151868 | 2.82283e-06 | 23 | 0 |
| 2021 | 0 | 0.000997857 | 5.65358e-05 | 25 | 0 |
| 2022 | 0 | 0.000648913 | 0 | 18 | 0 |
| 2023 | 0 | 0.000767614 | 0 | 13 | 0 |
| 2024 | 0 | 0.000397208 | 0 | 18 | 0 |
| 2025 | 0 | 5.38138e-05 | 0 | 17 | 0 |
| 2026 | 0 | 6.5371e-06 | 0 | 25 | 0 |

## Bar integrity H1

| key | ohlc_inconsistent | flat | spread_zero | spread_median_nonzero | duplicate_or_backward_time |
|---|---|---|---|---|---|
| 2013 | 0 | 0.000650195 | 1 | nan | 0 |
| 2014 | 0 | 0.000651466 | 1 | nan | 0 |
| 2015 | 0 | 0.000651148 | 1 | nan | 0 |
| 2016 | 0 | 0 | 1 | nan | 0 |
| 2017 | 0 | 0 | 1 | nan | 0 |
| 2018 | 0 | 0 | 0.78225 | 17 | 0 |
| 2019 | 0 | 0.00017001 | 0.00034002 | 16 | 0 |
| 2020 | 0 | 0 | 0.000169147 | 23 | 0 |
| 2021 | 0 | 0 | 0.000169434 | 25 | 0 |
| 2022 | 0 | 0 | 0 | 18 | 0 |
| 2023 | 0 | 0 | 0 | 12 | 0 |
| 2024 | 0 | 0 | 0 | 17 | 0 |
| 2025 | 0 | 0 | 0 | 16 | 0 |
| 2026 | 0 | 0 | 0 | 24 | 0 |

## Bar integrity D1

| key | ohlc_inconsistent | flat | spread_zero | spread_median_nonzero | duplicate_or_backward_time |
|---|---|---|---|---|---|
| 2013 | 0 | 0.0151515 | 1 | nan | 0 |
| 2014 | 0 | 0.003861 | 1 | nan | 0 |
| 2015 | 0 | 0 | 1 | nan | 0 |
| 2016 | 0 | 0 | 1 | nan | 0 |
| 2017 | 0 | 0 | 1 | nan | 0 |
| 2018 | 0 | 0 | 0.77907 | 14 | 0 |
| 2019 | 0 | 0 | 0.00775194 | 14 | 0 |
| 2020 | 0 | 0 | 0.003861 | 23 | 0 |
| 2021 | 0 | 0 | 0.00387597 | 25 | 0 |
| 2022 | 0 | 0 | 0 | 18 | 0 |
| 2023 | 0 | 0 | 0 | 12 | 0 |
| 2024 | 0 | 0 | 0 | 17 | 0 |
| 2025 | 0 | 0 | 0 | 16 | 0 |
| 2026 | 0 | 0 | 0 | 24 | 0 |

## Aggregation mismatch rate by year (low discrimination)

| key | H1_vs_M1 | D1_vs_H1 |
|---|---|---|
| 2013 | 0 | 0 |
| 2014 | 0 | 0 |
| 2015 | 0 | 0 |
| 2016 | 0 | 0 |
| 2017 | 0 | 0 |
| 2018 | 0 | 0 |
| 2019 | 0 | 0 |
| 2020 | 0 | 0 |
| 2021 | 0 | 0 |
| 2022 | 0 | 0 |
| 2023 | 0 | 0 |
| 2024 | 0 | 0 |
| 2025 | 0 | 0 |
| 2026 | 0 | 0 |

## Sessions

{2013: {'h1_bars_per_day_median': 24.0, 'hours_missing_on_full_days_over_50pct': [], 'weekly_open_hour_mode': 0}, 2014: {'h1_bars_per_day_median': 24.0, 'hours_missing_on_full_days_over_50pct': [], 'weekly_open_hour_mode': 0}, 2015: {'h1_bars_per_day_median': 24.0, 'hours_missing_on_full_days_over_50pct': [], 'weekly_open_hour_mode': 0}, 2016: {'h1_bars_per_day_median': 24.0, 'hours_missing_on_full_days_over_50pct': [], 'weekly_open_hour_mode': 0}, 2017: {'h1_bars_per_day_median': 24.0, 'hours_missing_on_full_days_over_50pct': [], 'weekly_open_hour_mode': 0}, 2018: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2019: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2020: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2021: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2022: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2023: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2024: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2025: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}, 2026: {'h1_bars_per_day_median': 23.0, 'hours_missing_on_full_days_over_50pct': [0], 'weekly_open_hour_mode': 1}}

## Return shape

| key | z_kurtosis | z_abs_gt5 | atr14_pct | median_tick_volume |
|---|---|---|---|---|
| 2013 | 3.31377 | 0 | 0.312809 | 1578.5 |
| 2014 | 2.86099 | 0 | 0.251903 | 1335 |
| 2015 | 2.59241 | 0 | 0.288422 | 2337 |
| 2016 | 2.41674 | 0 | 0.347418 | 4402.5 |
| 2017 | 2.8612 | 0 | 0.237473 | 7987.5 |
| 2018 | 1.54614 | 0 | 0.196296 | 3912 |
| 2019 | 1.82121 | 0 | 0.201761 | 1592.5 |
| 2020 | 1.45017 | 0 | 0.344738 | 2658.5 |
| 2021 | 2.12066 | 0 | 0.247552 | 4008.5 |
| 2022 | 1.68896 | 0 | 0.26725 | 4270 |
| 2023 | 2.24684 | 0 | 0.221381 | 4118 |
| 2024 | 1.73176 | 0 | 0.25208 | 6201.5 |
| 2025 | 1.15552 | 0 | 0.328925 | 9358 |
| 2026 | 1.64106 | 0 | 0.49122 | 14352 |

## Cross-symbol residual (GOLD# - EURUSD - XAUEUR#; XM's own gold, not external)

{'2017_08_12': {'bars': 1901, 'var_ratio': 1.4891, 'corr_g_minus_e_vs_x': 0.3249}, '2018': {'bars': 5565, 'var_ratio': 0.5507, 'corr_g_minus_e_vs_x': 0.7259}, '2019': {'bars': 5603, 'var_ratio': 0.1118, 'corr_g_minus_e_vs_x': 0.9442}, '2020': {'bars': 5633, 'var_ratio': 0.542, 'corr_g_minus_e_vs_x': 0.7287}, '2021': {'bars': 5626, 'var_ratio': 0.081, 'corr_g_minus_e_vs_x': 0.9591}, '2022': {'bars': 5644, 'var_ratio': 0.1299, 'corr_g_minus_e_vs_x': 0.9351}, '2023': {'bars': 5627, 'var_ratio': 0.0028, 'corr_g_minus_e_vs_x': 0.9986}, '2024': {'bars': 5669, 'var_ratio': 0.0018, 'corr_g_minus_e_vs_x': 0.9991}, '2025': {'bars': 5636, 'var_ratio': 0.0021, 'corr_g_minus_e_vs_x': 0.999}, '2026': {'bars': 4883, 'var_ratio': 0.0003, 'corr_g_minus_e_vs_x': 0.9998}}
# E-009 volatility-managed sizing on fixed H1 C trades (H-018)

Outcome: **underpowered**

underpowered: 2024+ 274 trades / 33 months with trades; Sharpe EWMA_RISK 1.291 vs FIXED_RISK 1.087 (delta 0.204, 95% month-block CI [-0.29, 0.617]); vol-matched MDD -5.401% vs -4.877%, CVaR5 -91.025 vs -93.608 bp; alpha on FIXED+market {'alpha_bp': 0.625, 'nw_t': 1.41, 'betas': [0.7317, -0.0121]}; delta Sharpe spread x2 0.202, quoted swap x1.5 0.207; descriptive EWMA_TARGET vs FIXED_RISK {'delta_sharpe': 0.032, 'ci': [-0.278, 0.344]}, vs CONSTANT_NOTIONAL {'delta_sharpe': 0.111, 'ci': [-0.412, 0.621]}; hour R^2 of weight ratios {'EWMA_RISK': 0.072, 'EWMA_TARGET': 0.065, 'CONSTANT_NOTIONAL': 0.081}

Primary: EWMA_RISK vs FIXED_RISK, oos_2024_on / base. in_sample_2018_2023 is descriptive (H1 C signal fit on 2014-2023). EWMA_TARGET and CONSTANT_NOTIONAL are descriptive. The delta-Sharpe CI is a percentile month-block bootstrap over about 34 months, slightly narrow (about 5% per tail instead of 2.5% in the critic's simulation).
Evidence class: development (2024+ prices were inspected in earlier experiments).

| window | scenario | arm | trades | days | net_mean_bp | annual_pct | sharpe | max_dd_pct | calmar | vol_matched_max_dd_pct | vol_matched_cvar5_bp | worst_day_bp | avg_w | max_w | return_per_unit_notional_bp | cost_bp_total | top10pct_trades_abs_pnl_share | month_pnl_herfindahl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in_sample_2018_2023 | base | FIXED_RISK | 363 | 1548 | 6.148 | 15.49 | 2.629 | -4.36 | 3.557 | -4.431 | -63.807 | -157.619 | 1.129 | 3.133 | 23.218 | -450.26 | 0.138 | 0.0261 |
| in_sample_2018_2023 | base | EWMA_RISK | 363 | 1548 | 7.339 | 18.49 | 2.718 | -4.42 | 4.182 | -3.896 | -60.102 | -157.21 | 1.366 | 5.0 | 22.911 | -535.43 | 0.212 | 0.0305 |
| in_sample_2018_2023 | base | EWMA_TARGET | 363 | 1548 | 5.439 | 13.71 | 2.257 | -4.86 | 2.821 | -4.798 | -71.287 | -280.386 | 1.075 | 1.903 | 21.585 | -422.03 | 0.191 | 0.0274 |
| in_sample_2018_2023 | base | CONSTANT_NOTIONAL | 363 | 1548 | 5.02 | 12.65 | 1.876 | -8.99 | 1.407 | -8.09 | -72.397 | -388.363 | 0.986 | 0.986 | 21.699 | -404.44 | 0.22 | 0.0306 |
| in_sample_2018_2023 | spread_x2 | FIXED_RISK | 364 | 1548 | 5.932 | 14.95 | 2.553 | -4.38 | 3.41 | -4.461 | -63.678 | -156.543 | 1.128 | 3.134 | 22.368 | -922.38 | 0.138 | 0.0254 |
| in_sample_2018_2023 | spread_x2 | EWMA_RISK | 364 | 1548 | 7.147 | 18.01 | 2.651 | -4.45 | 4.045 | -3.903 | -59.758 | -156.139 | 1.363 | 5.0 | 22.298 | -1079.74 | 0.212 | 0.0299 |
| in_sample_2018_2023 | spread_x2 | EWMA_TARGET | 364 | 1548 | 5.279 | 13.3 | 2.198 | -4.86 | 2.737 | -4.784 | -70.39 | -280.363 | 1.073 | 1.903 | 20.926 | -859.34 | 0.191 | 0.027 |
| in_sample_2018_2023 | spread_x2 | CONSTANT_NOTIONAL | 364 | 1548 | 4.78 | 12.05 | 1.799 | -9.0 | 1.339 | -8.099 | -72.421 | -388.298 | 0.986 | 0.986 | 20.608 | -846.64 | 0.22 | 0.0291 |
| in_sample_2018_2023 | swap_quoted_x1.5 | FIXED_RISK | 363 | 1548 | 5.459 | 13.76 | 2.364 | -5.04 | 2.732 | -5.117 | -65.086 | -161.596 | 1.129 | 3.133 | 20.615 | -1517.09 | 0.14 | 0.0264 |
| in_sample_2018_2023 | swap_quoted_x1.5 | EWMA_RISK | 363 | 1548 | 6.518 | 16.43 | 2.463 | -5.15 | 3.19 | -4.564 | -62.627 | -161.166 | 1.366 | 5.0 | 20.349 | -1805.86 | 0.207 | 0.0303 |
| in_sample_2018_2023 | swap_quoted_x1.5 | EWMA_TARGET | 363 | 1548 | 4.734 | 11.93 | 1.987 | -4.9 | 2.437 | -4.83 | -73.081 | -280.386 | 1.075 | 1.903 | 18.786 | -1513.84 | 0.189 | 0.0274 |
| in_sample_2018_2023 | swap_quoted_x1.5 | CONSTANT_NOTIONAL | 363 | 1548 | 4.373 | 11.02 | 1.646 | -9.07 | 1.215 | -8.119 | -73.527 | -388.363 | 0.986 | 0.986 | 18.906 | -1404.69 | 0.22 | 0.0313 |
| oos_2024_on | base | FIXED_RISK | 274 | 740 | 2.922 | 7.36 | 1.087 | -4.88 | 1.51 | -4.877 | -93.608 | -153.329 | 0.752 | 2.021 | 10.489 | -165.35 | 0.145 | 0.0621 |
| oos_2024_on | base | EWMA_RISK | 274 | 740 | 2.645 | 6.66 | 1.291 | -4.14 | 1.612 | -5.401 | -91.025 | -158.641 | 0.566 | 2.581 | 12.625 | -124.49 | 0.234 | 0.0825 |
| oos_2024_on | base | EWMA_TARGET | 274 | 740 | 2.928 | 7.38 | 1.119 | -5.56 | 1.328 | -5.776 | -97.504 | -240.632 | 0.676 | 1.33 | 11.703 | -148.54 | 0.191 | 0.0722 |
| oos_2024_on | base | CONSTANT_NOTIONAL | 274 | 740 | 4.182 | 10.54 | 1.009 | -8.1 | 1.301 | -5.421 | -97.799 | -280.72 | 0.986 | 0.986 | 11.448 | -216.64 | 0.22 | 0.0799 |
| oos_2024_on | spread_x2 | FIXED_RISK | 274 | 740 | 2.511 | 6.33 | 0.942 | -4.88 | 1.296 | -4.882 | -93.454 | -153.337 | 0.752 | 2.021 | 9.011 | -330.69 | 0.146 | 0.059 |
| oos_2024_on | spread_x2 | EWMA_RISK | 274 | 740 | 2.315 | 5.83 | 1.144 | -4.14 | 1.41 | -5.423 | -91.097 | -158.641 | 0.566 | 2.581 | 11.049 | -248.98 | 0.234 | 0.0799 |
| oos_2024_on | spread_x2 | EWMA_TARGET | 274 | 740 | 2.606 | 6.57 | 1.003 | -5.56 | 1.182 | -5.765 | -97.167 | -240.613 | 0.676 | 1.33 | 10.417 | -297.06 | 0.192 | 0.069 |
| oos_2024_on | spread_x2 | CONSTANT_NOTIONAL | 274 | 740 | 3.777 | 9.52 | 0.914 | -8.11 | 1.173 | -5.404 | -97.265 | -281.487 | 0.986 | 0.986 | 10.342 | -433.25 | 0.221 | 0.077 |
| oos_2024_on | swap_quoted_x1.5 | FIXED_RISK | 274 | 740 | 2.402 | 6.05 | 0.894 | -5.64 | 1.073 | -5.682 | -95.941 | -154.999 | 0.752 | 2.021 | 8.623 | -550.19 | 0.146 | 0.0601 |
| oos_2024_on | swap_quoted_x1.5 | EWMA_RISK | 274 | 740 | 2.254 | 5.68 | 1.101 | -4.84 | 1.173 | -6.367 | -93.59 | -158.641 | 0.566 | 2.581 | 10.758 | -413.81 | 0.23 | 0.0803 |
| oos_2024_on | swap_quoted_x1.5 | EWMA_TARGET | 274 | 740 | 2.439 | 6.15 | 0.935 | -6.13 | 1.002 | -6.365 | -99.743 | -240.632 | 0.676 | 1.33 | 9.748 | -510.4 | 0.187 | 0.0705 |
| oos_2024_on | swap_quoted_x1.5 | CONSTANT_NOTIONAL | 274 | 740 | 3.456 | 8.71 | 0.836 | -8.34 | 1.045 | -5.553 | -99.618 | -280.72 | 0.986 | 0.986 | 9.463 | -753.22 | 0.217 | 0.0773 |

By regime (2024+, base): {'low': {'days': 139, 'EWMA_RISK': 2.212, 'FIXED_RISK': 2.132}, 'mid': {'days': 217, 'EWMA_RISK': 1.318, 'FIXED_RISK': 1.266}, 'high': {'days': 375, 'EWMA_RISK': 0.79, 'FIXED_RISK': 0.759}}
By year (2024+, base): {2024: {'EWMA_RISK': 1.629, 'FIXED_RISK': 1.501}, 2025: {'EWMA_RISK': 1.703, 'FIXED_RISK': 1.544}, 2026: {'EWMA_RISK': 0.223, 'FIXED_RISK': 0.403}}
Leverage cap binding (2024+, base): {'FIXED_RISK': 0, 'EWMA_RISK': 0, 'EWMA_TARGET': 0, 'CONSTANT_NOTIONAL': 0}
Calibration (2018-2023 signal bars, entry info only): {'sigma_target': 0.008783908914697672, 'constant_w': 0.9864782244556155, 'sigma_ref': 0.008904311009546147, 'bars': 1263}
Swap per night (fraction of notional): account 0.000000, quoted -0.000197